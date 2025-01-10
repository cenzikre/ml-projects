# Preprocessing the dataset (label, tabular features, sequence features)

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.nn.utils.rnn import pad_sequence
import torch


datasets = {
    'train': {},
    'test': {},
    'valid': {},
}
modelsets = {'train', 'test', 'valid'}



# Label

query = """
sql query to get the data
"""
label_df = pd.read_sql(query, conn)

train_y = label_df.loc[label_df['SETS']=='TRAIN', ['LOAN_KEY', 'DPD90M12']].sort_values('LOAN_KEY', ignore_index=True)
test_y = label_df.loc[label_df['SETS']=='TEST', ['LOAN_KEY', 'DPD90M12']].sort_values('LOAN_KEY', ignore_index=True)
valid_y = label_df.loc[label_df['SETS']=='VALID', ['LOAN_KEY', 'DPD90M12']].sort_values('LOAN_KEY', ignore_index=True)
labels = [train_y, test_y, valid_y]

for modelset, label in zip(modelsets, labels):
    datasets[modelset]['Label'] = label


# Features

query = """
sql query to get the data
"""
df = pd.read_sql(query, conn)



# RVLR Pattern

def unique_values(x, substr='-', res=None):
    if not res:
        res = set()
    res.update(set(x[x.str.contains('-')].unique()))
    return res

rvlr_map = {
    'rvlr1': ['RVLR14', 'RVLR15', 'RVLR16'],
    'rvlr2': ['RVLR17', 'RVLR18', 'RVLR19'],
    'rvlr3': ['RVLR20', 'RVLR21', 'RVLR22'],
    'rvlr4': ['RVLR23', 'RVLR24', 'RVLR25'],
    'rvlr5': ['RVLR26', 'RVLR27', 'RVLR28'],
}
rvlr_list = [x for value in rvlr_map.values() for x in value]

rvlrs = (df.set_index('CUR_LOAN_KEY'))[rvlr_list]
rvlrs = rvlrs.apply(lambda x: x.str.replace(' ', 'X')).fillna('')
to_replace = None
for rvlr in rvlr_list:
    to_replace = unique_values(rvlrs[rvlr], substr='-', res=to_replace)
rvlrs = rvlrs.replace(list(to_replace), 'Z')

for key, values in rvlr_map.items():
    rvlrs[key] = rvlrs[values[0]] + rvlrs[values[1]] + rvlrs[values[2]]
rvlrs = rvlrs.drop(columns=rvlr_list)
rvlrs = rvlrs.map(lambda x: x[::-1])


max_pad_len = 24
preprocessors = {}
rti_cd = ['I', 'R', 'T', 'X', 'Z']
rti_cd_encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unkown_value=-1,
    # encoded_missing_value=-2,
)
rti_cd_encoder.fit(pd.DataFrame.from_dict({'RTI_CODE':rti_cd}))

for modelset, label in zip(modelsets, labels):
    _rvlrs = rvlrs.reindex(label['LOAN_KEY'])
    attention_mask = torch.from_numpy(_rvlrs.apply(lambda x: ~(x=='ZZZ') & x.notna()).values)
    encoded_padded_list = []
    seq_mask_list = []

    for i, col in enumerate(_rvlrs.columns):
        _data = _rvlrs[col].fillna('ZZZ').apply(list).explode().to_frame()
        _data.columns = ['RTI_CODE']
        _data['ENCODED'] = rti_cd_encoder.transform(_data[['RTI_CODE']]) + 1

        encoded_list = [torch.from_numpy(seq['ENCODED'].values).float() for _, seq in _data.groupby(level=0)]
        encoded_list = [tensor[(tensor.size(0) - max_pad_len):, :] if tensor.size(0) > max_pad_len else tensor for tensor in encoded_list]

        seq_lens = [tensor.size(0) for tensor in encoded_list]
        encoded_padded = pad_sequence(encoded_list, batch_first=True)
        seq_mask = torch.zeros(encoded_padded.size(0), encoded_padded.size(1), dtype=bool)
        for i, length in enumerate(seq_lens):
            seq_mask[i, :length] = 1

        encoded_padded_list.append(encoded_padded)
        seq_mask_list.append(seq_mask)

    seq_encoded_padded_stacked = torch.stack(encoded_padded_list, dim=2)
    seq_mask_stacked = torch.stack(seq_mask_list, dim=2)

    datasets[modelset]['TU_RTI_Sequences'] = {
        'tensor': seq_encoded_padded_stacked,
        'seq_mask': seq_mask_stacked,
        'att_mask': attention_mask,
        'encoders': rti_cd_encoder
    }



# TU Features

tu_feature_names = [x for x in df.columns if 'RVLR' not in x][2:]
tu_features = (df.set_index('CUR_LOAN_KEY'))[['CHANNEL'] + tu_feature_names]

channels = {
    ('STORE', 'WEB', 'LBP'): None,
    ('STORE',): None,
    ('WBE', 'LBP'): None,
}

for modelset, label in zip(modelsets, labels):
    scaled_feature_list = []
    attention_mask_list = []
    for segm, scaler in channels.items():

        _channel = tu_features[tu_features['CHANNEL'].isin(segm)].drop(columns=['CHANNEL'])
        _channel = _channel.reindex(label['LOAN_KEY'])
        attention_mask = _channel.apply(lambda x: x.isna()).sum(axis=1)==0
        attention_mask = torch.from_numpy(attention_mask.values)

        if not scaler:
            scaler = StandardScaler()
            scaler.fit(_channel)
            channels[segm] = scaler
        _channel = pd.DataFrame(scaler.transform(_channel), columns=scaler.feature_names_in_, index=_channel.index)
        scaled_feature_list.append(torch.from_numpy(_channel.values).float())
        attention_mask_list.append(attention_mask)

    scaled_features = torch.stack(scaled_feature_list, dim=2)
    attention_masks = torch.stack(attention_mask_list, dim=2)

    datasets[modelset]['TU_Features'] = {
        'tensor': scaled_features,
        'att_mask': attention_masks,
        'scalers': channels
    }




# Transaction Sequence

query = """

"""
df = pd.read_sql(query, conn)
df = df.sort_values(['CUR_LOAN_KEY', 'LOAN_PAYMENT_KEY'], ignore_index=True)


max_seq_len = 512
preprocessors = {}
trans_cd = [
    0, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14,
    ....
]
pmtmthd_cd = [-1, 0, 1, 2, 3, 4, ....]

for modelset, label in zip(modelsets, labels):
    # filter the overall dataset based on labels for each model set
    filtered_df = pd.merge(
        label[['LOAN_KEY']],
        df[['CUR_LOAN_KEY', 'LOAN_PAYMENT_KEY', 'DAYS_TO_APP', 'PAID_AMT', 'PAYMENT_METHOD', 'TRANS_CODE']],
        how='left',
        left_on='LOAN_KEY',
        right_on='CUR_LOAN_KEY',
    ).sort_values(['LOAN_KEY', 'LOAN_PAYMENT_KEY'], ignore_index=True).drop(columns=['CUR_LOAN_KEY', 'LOAN_PAYMENT_KEY'])

    # create mask for loans with missing sequence info
    masked_df = filtered_df.groupby('LOAN_KEY').apply(lambda x: not x.isna().all(axis=None), include_groups=False)
    attention_mask = torch.from_numpy(masked_df.values)

    # truncate the sequences to 512 in max length
    caped_df = filtered_df.groupby('LOAN_KEY').apply(lambda x: x.tail(512), include_groups=False).reset_index(level=1, drop=True)

    # normalize 'DAYS_TO_APP', 'PAID_AMT'
    if modelset == 'train':
        scaler = StandardScaler()
        caped_df[['DAYS_TO_APP', 'PAID_AMT']] = scaler.fit_transform(caped_df[['DAYS_TO_APP', 'PAID_AMT']])
        preprocessors['scaler'] = scaler
    else:
        caped_df[['DAYS_TO_APP', 'PAID_AMT']] = preprocessors['scaler'].transform(caped_df[['DAYS_TO_APP', 'PAID_AMT']])
    caped_df[['DAYS_TO_APP', 'PAID_AMT']] = caped_df[['DAYS_TO_APP', 'PAID_AMT']].fillna(0)

    # encode 'PAYMENT_METHOD'
    if modelset == 'train':
        pmtmthd_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unkown_values=-1,
            encoded_missing_value=-2,
        )
        pmtmthd_encoder.fit(pd.DataFrame.from_dict({'PAYMENT_METHOD': pmtmthd_cd}))
        preprocessors['pmtmthd_encoder'] = pmtmthd_encoder
    caped_df['PAYMENT_METHOD'] = preprocessors['pmtmthd_encoder'].transform(caped_df[['PAYMENT_METHOD']]) + 2

    # encode 'TRANS_CODE'
    if modelset == 'train':
        transcd_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        transcd_encoder.fit(pd.DataFrame.from_dict({'TRANS_CODE': trans_cd}))
        preprocessors['transcd_encoder'] = transcd_encoder
    caped_df['TRANS_CODE'] = preprocessors['transcd_encoder'].transform(caped_df[['TRANS_CODE']]) + 2

    # pad the processed sequences
    encoded_list = [torch.from_numpy(seq.values).float() for _, seq in caped_df.groupby(level=0)]
    encoded_padded = pad_sequence(encoded_list, batch_first=True)

    # create sequence padding mask
    seq_lens = [tensor.size(0) for tensor in encoded_list]
    seq_mask = torch.zeros(encoded_padded.size(0), encoded_padded.size(1), dtype=bool)
    for i, length in enumerate(seq_lens):
        seq_mask[i, :length] = 1

    datasets[modelset]['SP_Sequence'] = {
        'tensor': encoded_padded,
        'att_mask': attention_mask,
        'seq_mask': seq_mask,
        'preprocessors': preprocessors,
    }
