# Preprocessing the dataset (label, tabular features, sequence features)


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

