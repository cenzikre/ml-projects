import functools
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

log_file_path = 'training_log.txt'

def log_epoch_performance_chckpnt(log_file_path):
    first_run = True

    def decorator(epoch_func):
        @functools.wraps(epoch_func)
        def wrapper(*args, **kwargs):
            nonlocal first_run
            # clear the log file only on the first run
            if first_run:
                with open(log_file_path, 'w') as log_file:
                    log_file.write("Epoch Performance Log\n")
                first_run = False

            # call the original epoch function and get results
            start_time = time.time()
            losses, scores, checkpoint = epoch_func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60

            # log the performance for this epoch
            with open(log_file_path, 'a') as log_file:
                epoch_number = len(losses) # Assuming each call is a new epoch
                epoch_loss_train, epoch_loss_valid = losses[-1]
                train_auc, valid_auc = scores[-1]

                # log_entry = (
                #     f"Epoch: {epoch_number}, "
                #     f"Train Loss: {epoch_loss_train:.3f}, Valid Loss: {epoch_loss_valid:.3f}, "
                #     f"Train AUC: {train_auc:.3f}, Valid AUC: {valid_auc:.3f}\n"
                # )
                log_entry = f"Epoch {epoch_number}, Time: {elapsed_time:.1f} min, Loss: {epoch_loss_train:.3f} vs {epoch_loss_valid:.3f}, AUC: {train_auc:.3f} vs {valid_auc:.3f}\n"

                log_file.write(log_entry)
                print(f"Logged - {log_entry.strip()})")

            return losses, scores, checkpoint
        return wrapper
    return decorator

def log_epoch_performance(log_file_path):
    first_run = True

    def decorator(epoch_func):
        @functools.wraps(epoch_func)
        def wrapper(*args, **kwargs):
            nonlocal first_run
            # clear the log file only on the first run
            if first_run:
                with open(log_file_path, 'w') as log_file:
                    log_file.write("Epoch Performance Log\n")
                first_run = False

            # call the original epoch function and get results
            start_time = time.time()
            losses, scores = epoch_func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60

            # log the performance for this epoch
            with open(log_file_path, 'a') as log_file:
                epoch_number = len(losses) # Assuming each call is a new epoch
                epoch_loss_train, epoch_loss_valid = losses[-1]
                train_auc, valid_auc = scores[-1]

                # log_entry = (
                #     f"Epoch: {epoch_number}, "
                #     f"Train Loss: {epoch_loss_train:.3f}, Valid Loss: {epoch_loss_valid:.3f}, "
                #     f"Train AUC: {train_auc:.3f}, Valid AUC: {valid_auc:.3f}\n"
                # )
                log_entry = f"Epoch {epoch_number}, Time: {elapsed_time:.1f} min, Loss: {epoch_loss_train:.3f} vs {epoch_loss_valid:.3f}, AUC: {train_auc:.3f} vs {valid_auc:.3f}\n"

                log_file.write(log_entry)
                print(f"Logged - {log_entry.strip()})")

            return losses, scores
        return wrapper
    return decorator



class FeatureSequenceDataset(Dataset):
    def __init__(self, modelset: str, label: dict, features: dict, sequences: list):
        mask = features[modelset]['att_mask'][:, 0]
        self.label = torch.from_numpy(label[modelset]['DPD90M12'].values)[mask]
        self.features = features[modelset]['tensor'][:, :, 0][mask]
        self.sequences = [seq[modelset]['tensor'][mask] for seq in sequences]
        self.sequence_masks = [seq[modelset]['seq_mask'][mask] for seq in sequences]
        attention_masks = [features[modelset]['att_mask'][:, 0].reshape(-1, 1)] + [seq[modelset]['att_mask'].reshape(-1, 1) for seq in sequences]
        self.attention_mask = torch.cat(attention_masks, dim=1)[mask]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'label': self.label[idx],
            'features': self.features[idx],
            'sequences': [tensor[idx] for tensor in self.sequences],
            'sequence_masks': [tensor[idx] for tensor in self.sequence_masks],
            'attention_mask': self.attention_mask[idx],
        }



class FeatureSequenceNet(nn.Module):
    def __init__(self, feature_dim, feature_n_layers, n_seq, seq_embedding_dim, hidden_dim, seq_n_layers, dropout=0.2, n_heads=2, n_transformer_layers=2):
        super(FeatureSequenceNet, self).__init__()

        # MLP Extractor
        feature_MLP = [
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
        for i in range(feature_n_layers-1):
            _layers = [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ]
            feature_MLP += _layers
        self.feature_MLP = nn.Sequential(*feature_MLP)

        # sequence embeddings
        self.embeddings = nn.ModuleList([
            nn.Linear(1, seq_embedding_dim),
            nn.Linear(1, seq_embedding_dim),
            nn.Embedding(num_embeddings=15, embedding_dim=seq_embedding_dim),
            nn.Embedding(num_embeddings=60, embedding_dim=seq_embedding_dim),
        ])
        self.embedding_dropout = nn.Dropout(dropout)

        # self.pre_lstm = nn.Sequential(
        #     nn.Linear(seq_embedding_dim * 4, seq_embedding_dim),
        #     nn.LayerNorm(seq_embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(dropout),
        # )
        # self.lstms = nn.ModuleList([
        #     nn.LSTM(
        #         input_size=seq_embedding_dim * 4,
        #         hidden_size=hidden_size,
        #         num_layers=seq_n_layers,
        #         batch_first=True,
        #         dropout=dropout,
        #         bidirectional=False
        #     )
        #     for _ in range(n_seq)
        # ])

        # sequence extractor
        self.lstm = nn.LSTM(
            input_size=seq_embedding_dim*4,
            hidden_size=hidden_dim,
            num_layers=seq_n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # classifier
        self.output = nn.Linear(hidden_dim, 2)

        # dynamic gating, feature-level vs modality-level
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)

        # simple qkv attention
        self.query = nn.Parameter(torch.randn(hidden_dim))

        # dynamic qkv attention
        self.query_fc = nn.Linear(hidden_dim, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)

        # multihead attention
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

        # dynamic fc attention
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # transformer attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

    def seq_embedding(self, x):
        xs = []
        for i, layer in enumerate(self.embeddings):
            if i in (0, 1):
                _x = x[:, :, i].unsqueeze(-1) # continous features
                xs.append(layer(_x))
            else:
                _x = x[:, :, i].long()        # categorical features
                xs.append(layer(_x))
        x = torch.cat(xs, dim=-1) # Concatenate along the feature dimension
        x = self.embedding_dropout(x)

        return x

    def masked_mean_pooling(self, x, mask):
        x = x.masked_fill((~mask).unsqueeze(-1), float('nan'))
        pooled = torch.nanmean(x, dim=1)
        pooled[pooled.isnan()] = torch.tensor(0.0, device=x.device)
        return pooled

    # def masked_mean_pooling(self, input_tensor, mask):
    #     mask = mask.unsqueeze(-1).float()
    #     masked_input = input_tensor * mask
    #     sum_features = masked_input.sum(dim=1)
    #     num_available = mask.sum(dim=1)
    #     num_available_safe = num_available.clone()
    #     num_available_safe[num_available_safe==0] = 1.0
    #     mean_features = sum_features / num_available_safe
    #     mean_features[num_available.squeeze(-1)==0] = 0.0
    #     return mean_features

    def pass_lstm(self, x, mask, lstm):
        # sort
        seq_lengths = mask.sum(dim=1).cpu()
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]
        mask = mask[perm_idx]

        # pack, lstm, unpack
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        packed_output, (hn, cn) = lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # unsort
        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]
        seq_lengths = seq_lengths[unperm_idx]

        # pooling - last time step
        batch_size = x.size(0)
        hidden_dim = output.size(2)

        idx = (seq_lengths - 1).unsqueeze(1).expand(batch_size, hidden_dim).unsqueeze(1).to(output.device)
        last_outputs = output.gather(1, idx).squeeze(1)

        return last_outputs

    def gating(self, x, mask):
        gates = torch.sigmoid(self.gate_fc(x))
        gated_features = x * gates
        mask = mask.unsqueeze(-1)
        gated_features = gated_features * mask.float()
        return gated_features

    def fixed_qkv_attention_pooling(self, x, mask=None):
        batch_size, token_size, feature_size = x.shape
        query = self.query.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        key = x
        value = x
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(feature_size)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        pooled_x = torch.matmul(attention_weights, value).squeeze(1)
        return pooled_x

    def dynamic_qkv_attention_pooling(self, x, mask=None):
        batch_size, token_size, feature_size = x.shape
        query = self.qery_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)

        query = query.mean(dim=1, keepdim=True)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(feature_size)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        pooled_x = torch.matmul(attention_weights, value).squeeze(1)
        return pooled_x

    def dynamic_fc_attention_pooling(self, x, mask=None):
        batch_size, token_size, feature_size = x.shape
        x_flat = x.view(-1, feature_size)

        # compute attention score based on x
        attention_scores = self.attention_fc(x_flat)
        attention_scores = attention_scores.view(batch_size, token_size)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        pooled_x = torch.sum(x * attention_weights, dim=-2)
        return pooled_x

    def transformer_attention_pooling(self, x, mask=None):
        batch_size, token_size, feature_size = x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if mask is not None:
            transformer_mask = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        else:
            transformer_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        cls_x = x[:, 0, :]
        return cls_x

    def forward(self, feature_tensor, sequence_tensors, sequence_masks, attention_masks):
        # feature MLP extractor
        x = [self.feature_MLP(feature_tensor)]

        # sequence embedding & LSTM extractor
        for i in range(len(sequence_tensor)):
            xi = sequence_tensors[i]
            maski = sequence_masks[i]

            xi = self.embedding(xi)
            xi = self.pass_lstm(xi, maski, self.lstm)
            x.append(xi)

        # feature stacking and gating
        x = torch.stack(x, dim=1)
        x = self.gating(x, attention_masks)

        # pooling
        # x = self.masked_mean_pooling(x, attention_masks)
        # x = self.fixed_qkv_attention_pooling(x, attention_masks)
        # x = self.dynamic_qkv_attention_pooling(x, attention_masks)
        # x = self.dynamic_fc_attention_pooling(x, attention_masks)
        x = self.transformer_attention_pooling(x, attention_masks)

        # classification
        x = self.output(x)

        return x


def training(batch, model, loss_fn, optimizer):
    model.train()
    labels = batch['label'].long().to(device)
    features = batch['features'].to(device)
    sequences = [tensor.to(device) for tensor in batch['sequence']]
    sequence_masks = [tensor.to(device) for tensor in batch['sequence_masks']]
    attention_mask = batch['attention_mask'].to(device)
    
    optimizer.zero_grad()
    outputs = model(features, sequences, sequence_masks, attention_mask)
    
    if torch.isnan(outputs).any():
        raise RuntimeError("NaN detected in model outputs. Stopping training.")
    
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item(), outputs.detach().cpu(), labels.cpu()

def validation(batch, model, loss_fn):
    model.eval()
    with torch.no_grad():
        labels = batch['label'].long().to(device)
        features = batch['features'].to(device)
        sequences = [tensor.to(device) for tensor in batch['sequence']]
        sequence_masks = [tensor.to(device) for tensor in batch['sequence_masks']]
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(features, sequences, sequence_masks, attention_mask)
        loss = loss_fn(outputs, labels)

        return loss.item(), outputs.cpu(), labels.cpu()

def seg_test_auc(test_data, model, device='cpu'):
    model.to(device)
    labels = test_data['label'].long().to(device)
    features = test_data['features'].to(device)
    sequences = [tensor.to(device) for tensor in test_data['sequence']]
    sequence_masks = [tensor.to(device) for tensor in test_data['sequence_masks']]
    attention_mask = test_data['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(features, sequences, sequence_masks, attention_mask).cpu()

    return roc_auc_score(labels, outputs.softmax(dim=1)[:, 1])


@log_epoch_performance_chckpnt(log_file_path)
def epoch_model_update(model, loss_fn, optimizer, train_loader, valid_loader, losses=None, scores=None):
    losses = [] if losses is None else losses
    scores = [] if scores is None else scores

    epoch_loss_train = 0
    epoch_loss_valid = 0
    train_outputs, train_labels = [], []
    valid_outputs, valid_labels = [], []

    for train_batch in train_loader:
        batch_loss_train, batch_outputs_train, batch_labels_train = training(train_batch, model, loss_fn, optimizer)
        epoch_loss_train += batch_loss_train

        train_outputs.append(batch_outputs_train)
        train_labels.append(batch_labels_train)

    for valid_batch in valid_loader:
        batch_loss_valid, batch_outputs_valid, batch_labels_valid = validation(valid_batch, model, loss_fn)
        epoch_loss_valid += batch_loss_valid

        valid_outputs.append(batch_outputs_valid)
        valid_labels.append(batch_labels_valid)

    train_outputs, train_labels = torch.cat(train_outputs), torch.cat(train_labels)
    valid_outputs, valid_labels = torch.cat(valid_outputs), torch.cat(valid_labels)

    train_auc = roc_auc_score(train_labels, train_outputs.softmax(dim=1)[:, 1])
    valid_auc = roc_auc_score(valid_labels, valid_outputs.softmax(dim=1)[:, 1])

    scores.append((train_auc, valid_auc))
    epoch_loss_train /= len(train_loader)
    epoch_loss_valid /= len(valid_loader)
    losses.append((epoch_loss_train, epoch_loss_valid))

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': (epoch_loss_train, epoch_loss_valid),
        'score': (train_auc, valid_auc),
    }
    return losses, scores, checkpoint








train_dataset = FeatureSequenceDataset('train', label_comp, features_comp, [seq1, seq2, seq3])
test_dataset = FeatureSequenceDataset('test', label_comp, features_comp, [seq1, seq2, seq3])
valid_dataset = FeatureSequenceDataset('valid', label_comp, features_comp, [seq1, seq2, seq3])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = FeatureSequenceNet(
    feature_dim=440,
    feature_n_layers=2,
    n_seq=3,
    seq_embedding_dim=16,
    hidden_dim=128,
    seq_n_layers=2,
    dropout=0.2,
    n_heads=2,
    n_transformer_layers=2
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 20
losses = []
scores = []

for epoch in range(num_epochs):
    losses, scores, checkpoint epoch_model_update(model, loss_fn, optimizer, train_loader, test_loader, losses, scores)
    torch.save(checkpoint, f"Feature-Sequence/model_epoch_{epoch+1}.pth")
