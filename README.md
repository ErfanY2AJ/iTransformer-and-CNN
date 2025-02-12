in the conversional Transformers we tend to embedd every sentenses or a sequense of time series one by one . Howrver , on the Inverted Transformer we tend to the model learn all of the series and labeld them as a Single Token 


the following figure show you what are the difference between these to Model !
![Screenshot 2025-02-05 221437](https://github.com/user-attachments/assets/6cad2778-2dc2-48a4-849b-cf8308fa3502)

we procces by loading the Exchange-rate dataset , the data contain 8 columns 
![image](https://github.com/user-attachments/assets/f513530a-dd10-478a-bc47-86d502394fde)

next step i create a sequesnse of data as a input to data loader 


# Custom Dataset Class

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return sequence, label
```


# Create Sequences for Training

```python
def create_sequences(data, input_length=96, horizon=96):
    sequences = []
    data_size = len(data)
    for i in range(data_size - input_length - horizon):
        input_seq = torch.tensor(data[i:i+input_length], dtype=torch.float32)
        target_seq = torch.tensor(data[i+input_length:i+input_length+horizon], dtype=torch.float32)
        sequences.append((input_seq, target_seq))
    return sequences
```

on the Next step we going to preprocess and load the data 

# Load and Preprocess the Data
```python

def load_and_preprocess_data(file_path):
    with gzip.open(file_path, 'rt') as file:
        df = pd.read_csv(file)
    df = df.ffill()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    return data_scaled, scaler

```

# Create sequences with input length 96 and the current horizon
```python
        sequences = create_sequences(data_scaled, input_length=96, horizon=horizon)
        train_size = int(len(sequences) * 0.7)
        train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]

        # Create DataLoaders
        train_dataset = TimeSeriesDataset(train_sequences)
        test_dataset = TimeSeriesDataset(test_sequences)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

now its the time move forward into iTransformer block , these transformer like old transformer consist of Attention , Encoding Layer , forward network and positional Encoding  with inverted Embeding 


# Positional Encoding Module
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1).to(device)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
```


# Attention part without predefining function

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Transform Q, K, V
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.softmax(scores)

        # Attention application on V
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final output layer
        output = self.output_layer(context)
        return output
```


# Custom Embedding Layer
```python
class DataEmbedding_inverted(nn.Module):
    def __init__(self, time_steps, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(time_steps, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.value_embedding(x)
        x = x.unsqueeze(1)
        return self.dropout(x)
```

# Transformer Layer
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.attention(src, src, src)
        src = self.norm1(src + src2)
        src2 = self.ff(src)
        src = self.norm2(src + src2)
        return src
```

and the resault is obtianed as bellow . as we see , there is a magnificant Loss decreasing and good performance . 

![image](https://github.com/user-attachments/assets/00adaac8-45b8-4e1d-94bd-1146d7b356f1)


![image](https://github.com/user-attachments/assets/b4270d3e-6390-422e-a994-f189c9a89c13)


![image](https://github.com/user-attachments/assets/332fc35d-bea8-496d-b312-5e15552ade6f)


![image](https://github.com/user-attachments/assets/2a89453b-7bb8-4a17-a6c6-a8f191d96d53)


![image](https://github.com/user-attachments/assets/a259f86f-5486-4d5c-a380-55bd8cce933e)


Actual vs prediction for a Batch of test dataset 

![image](https://github.com/user-attachments/assets/747f9510-23d8-4d53-9828-88b19d83f7e7)


![image](https://github.com/user-attachments/assets/d54ddce5-2744-4483-a1b7-8aefe64c584f)


![image](https://github.com/user-attachments/assets/1482ebc2-ee58-4fdd-9992-e58ce315f2df)



![image](https://github.com/user-attachments/assets/8fe4bb7f-8bd5-4efd-b1da-e379af92a8dd)



![image](https://github.com/user-attachments/assets/abe2c371-3c11-418e-a5ca-655f09fb515c)


![image](https://github.com/user-attachments/assets/8c5a633c-c54f-4bd0-8a62-1588ec8b4e81)


![image](https://github.com/user-attachments/assets/787ad0c2-ab04-4ef3-84b3-bed2f600af22)


![image](https://github.com/user-attachments/assets/80f9b184-f9f8-4aa6-a104-4bc6654864b5)



