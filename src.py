import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset,Dataset
from torch import nn, Tensor
from torch.nn import Transformer
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# import warnings
# warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def plotmea(day, mea1,mea2, df, duration=1):
    """
    Plots measurements from a specified range of days in the provided dataframe.
    'day' specifies the starting day (1-based index).
    'mea1' and 'mea2' are the column names for the data to plot.
    'df' is the dataframe containing the data with a datetime index.
    'duration' specifies the number of days to include in the plot, default is 1.
    """
    unique_dates = pd.unique(df.index.date)
    total_days = len(unique_dates)

    if day - 1 + duration > total_days:
        print(f"Requested range exceeds the available data. Total available days: {total_days}")
        return

    start_date = unique_dates[day - 1]
    end_date = unique_dates[day - 1 + duration - 1]
    selected_date = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    fig, ax1 = plt.subplots(figsize=(13, 3))

    ax1.scatter(selected_date.index, selected_date[mea2], color='r', label=mea2, s=10)
    ax2 = ax1.twinx()
    ax2.plot(selected_date.index, selected_date[mea1], color='b', label=mea1)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_title(f'{mea1} and {mea2} from {start_date} to {end_date}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(mea2)
    ax2.set_ylabel(mea1)

    plt.tight_layout()
    plt.show()


def confusionmetrix(df,mea1,mea2):
    # Confusion Matrix
    conf_matrix = confusion_matrix(df[mea1], df[mea2])
    print("Confusion Matrix:\n", conf_matrix)
    # Accuracy
    accuracy = accuracy_score(df[mea1], df[mea2])
    print("Accuracy:", accuracy)
    # Precision
    precision = precision_score(df[mea1], df[mea2])
    print("Precision:", precision)
    # Recall
    recall = recall_score(df[mea1], df[mea2])
    print("Recall:", recall)
    # F1 Score
    f1 = f1_score(df[mea1], df[mea2])
    print("F1 Score:", f1)
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(df[mea1], df[mea2])
    auc = roc_auc_score(df[mea1], df[mea2])
    print("AUC:", auc)
    # Plotting the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def evaluation(df,mea1,mea2):
    confusionmetrix(df,mea1,mea2)
    plotmea(1, mea1,mea2, df, 3)  
 



def preprocessingdf(df):
    # Combine 'Date' and 'Time' into a DatetimeIndex
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)

    # Drop the original 'Date' and 'Time' columns
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    # Convert 'Room_Occupancy_Count' to a binary occupancy column
    df['realoccupancy'] = (df['Room_Occupancy_Count'] > 0.5).astype(int)
    df.drop(['Room_Occupancy_Count','S5_CO2_Slope'], axis=1, inplace=True)

    start_date = '2017-12-21'
    end_date = '2017-12-28'

    df = df.loc[start_date:end_date]

    # To be consistent with the othjer dataset, Combine data of the same type.
    df['Pirstatus'] = ((df['S6_PIR'] == 1) | (df['S7_PIR'] == 1)).astype(int)
    df.drop(['S6_PIR', 'S7_PIR'], axis=1, inplace=True)

    temperature_columns = [col for col in df.columns if 'Temp' in col]
    df['Temperature'] = df[temperature_columns].mean(axis=1)

    co2_columns = [col for col in df.columns if 'CO2' in col]
    df['Co2'] = df[co2_columns].mean(axis=1)

    co2_columns = [col for col in df.columns if 'Light' in col]
    df['Light'] = df[co2_columns].mean(axis=1)

    co2_columns = [col for col in df.columns if 'Sound' in col]
    df['Sound'] = df[co2_columns].mean(axis=1)


    df_features = df[['Temperature', 'Co2','Sound','Light','realoccupancy', 'Pirstatus']]
    float_cols = df_features.select_dtypes(include=['float64','float32']).columns
    int_cols = df_features.select_dtypes(include=['int32', 'int64']).columns


    agg_funcs = {col: 'mean' for col in float_cols}
    agg_funcs.update({col: 'last' for col in int_cols})
    resampled_data = df_features.resample('30s').agg(agg_funcs)
    df_features = resampled_data.interpolate(method='linear')
    df_features['Pirstatus'] = df_features['Pirstatus'].astype(int)
    df_features['realoccupancy'] = df_features['realoccupancy'].astype(int)
    return df_features






############## pytorch dataloader and model ######################class TimeSeriesDataset(Dataset):
PAD_IDX = 99
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col, seq_length):
        self.data = data
        self.target_col = target_col
        self.seq_length = seq_length
        

    def __len__(self):
        return len(self.data) - self.seq_length + 1


    def __getitem__(self, idx):
        window = self.data.iloc[idx:idx+self.seq_length]
        if not self.is_time_continuous(window.index):
            return None 
        
        X = window.drop([self.target_col], axis=1).iloc[:self.seq_length].values
        y = window[self.target_col].iloc[:self.seq_length].values
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def is_time_continuous(self, times):
        expected_interval = pd.Timedelta(minutes=10)
        return all((times[i+1] - times[i]) <= expected_interval for i in range(len(times) - 1))
    

    
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TransformerModelEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_length):
        super(TransformerModelEncoder, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 2,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(seq_length * d_model, seq_length)

    def forward(self, x):
        x = x.permute(1, 0, 2) 

        x = self.positional_encoding(self.embedding(x)) # (seq_length, batch_size, d_model)

        x = self.transformer_encoder(x)
     
        # x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        x = x.flatten(start_dim=1)  # (batch_size, seq_length * d_model)
        out = self.fc(x)  # (batch_size, horizon)
        return out
    

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Linear(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    

def generate_square_subsequent_mask(sz,DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt,DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src[:,:,0] == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask