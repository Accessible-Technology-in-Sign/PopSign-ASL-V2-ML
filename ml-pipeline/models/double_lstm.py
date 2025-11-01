import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Double LSTM models    
class DoubleLSTM_A(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs, num_layers):
        super(DoubleLSTM_A, self).__init__()

        # First bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=512,  # Increased hidden size
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(512 * 2, num_signs)

        # Batch Normalization layer to improve training stability
        self.bn = nn.BatchNorm1d(512 * 2)  # Batch norm on concatenated outputs

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm(x)

        # Only take the last output for classification
        x = x[:, -1, :]

        # Apply Batch Normalization
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x
    
class DoubleLSTM2(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM2, self).__init__()

        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=256 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=256 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(256 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        x, _ = self.lstm2(x)

        # Third LST layer
        x, _ = self.lstm3(x)

        # Only take the last output for classification
        x = x[:, -1, :]

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x


class DoubleLSTM3(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM3, self).__init__()

        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=512,  # Increased hidden size
            batch_first=True,
            bidirectional=True
        )

        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(512 * 2, num_signs)

        # Batch Normalization layer to improve training stability
        self.bn = nn.BatchNorm1d(512 * 2)  # Batch norm on concatenated outputs

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        x, _ = self.lstm2(x)

        # Third LSTM layer
        x, _ = self.lstm3(x)

        # Only take the last output for classification
        x = x[:, -1, :]

        # Apply Batch Normalization
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x



class DoubleLSTM3FIXED(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM3, self).__init__()

        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=512,  # Increased hidden size
            batch_first=True,
            bidirectional=True
        )

        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(512 * 2, num_signs)

        # Batch Normalization layer to improve training stability
        self.bn = nn.BatchNorm1d(512 * 2)  # Batch norm on concatenated outputs

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        x, _ = self.lstm2(x)

        # Third LSTM layer
        x, _ = self.lstm3(x)

        # Only take the last output for classification
        x = x[:, -1, :]

        # Apply Batch Normalization
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x

class DoubleLSTM4(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM4, self).__init__()

        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=256,  # Increased hidden size
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=256 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = nn.Linear(256 * 2, num_signs)

        # Batch Normalization layer to improve training stability
        self.bn = nn.BatchNorm1d(256 * 2)  # Batch norm on concatenated outputs

    def forward(self, x, lengths):
        # Pack the padded sequence (without padding)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        # x, _ = self.lstm2(x)

        # Third LSTM layer
        x, _ = self.lstm3(x)

        # Unpack the sequence (so we get all timesteps back)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Only take the last output for classification
        # x = x[range(x.size(0)), lengths - 1, :]  # Get the last non-padded element
        x = x[torch.arange(x.size(0)), lengths - 1, :]


        # Apply Batch Normalization
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x


class DoubleLSTM3VARIABLE(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM3VARIABLE, self).__init__()

        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=512,  # Increased hidden size
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(512 * 2, num_signs)

        # Batch Normalization layer to improve training stability
        self.bn = nn.BatchNorm1d(512 * 2)  # Batch norm on concatenated outputs

    def forward(self, x, lengths):
        # Pack the padded sequence (without padding)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        # x, _ = self.lstm2(x)

        # Third LSTM layer
        x, _ = self.lstm3(x)

        # Unpack the sequence (so we get all timesteps back)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Only take the last output for classification
        # x = x[range(x.size(0)), lengths - 1, :]  # Get the last non-padded element
        x = x[torch.arange(x.size(0)), lengths - 1, :]


        # Apply Batch Normalization
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x


class DoubleLSTM4_FIXED(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM4_FIXED, self).__init__()

        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=256,  # Increased hidden size
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=256 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = nn.Linear(256 * 2, num_signs)

        # Batch Normalization layer to improve training stability
        self.bn = nn.BatchNorm1d(256 * 2)  # Batch norm on concatenated outputs

    def forward(self, x):
        # # Pack the padded sequence (without padding)
        # x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        # x, _ = self.lstm2(x)

        # Third LSTM layer
        x, _ = self.lstm3(x)

        # Unpack the sequence (so we get all timesteps back)
        # x, _ = pad_packed_sequence(x, batch_first=True)

        # Only take the last output for classification
        # x = x[range(x.size(0)), lengths - 1, :]  # Get the last non-padded element
        # x = x[torch.arange(x.size(0)), lengths - 1, :]


        # Apply Batch Normalization
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Linear layer to calculate attention scores
        self.attn = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        # Calculate attention scores (shape: [batch_size, seq_len, 1])
        attn_weights = torch.tanh(self.attn(x))  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Apply softmax to get weights (batch_size, seq_len, 1)

        # Compute weighted sum of LSTM outputs (shape: [batch_size, hidden_size*2])
        weighted_sum = torch.sum(attn_weights * x, dim=1)
        return weighted_sum, attn_weights

class DoubleLSTM3WithAttention(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM3WithAttention, self).__init__()

        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_features * num_coords,
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Third bidirectional LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=512 * 2,  # 512 hidden units * 2 for bidirection
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.attn = Attention(512)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(512 * 2, num_signs)

        # Batch Normalization layer
        self.bn = nn.BatchNorm1d(512 * 2)

    def forward(self, x):
        # Pass through the first LSTM layer
        x, _ = self.lstm1(x)

        # Pass through the second LSTM layer
        x, _ = self.lstm2(x)

        # Pass through the third LSTM layer
        x, _ = self.lstm3(x)

        # Apply attention mechanism to the output of the last LSTM
        x, attn_weights = self.attn(x)

        # Apply batch normalization to the output of attention
        x = self.bn(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc(x)

        return x, attn_weights

    
if __name__ == '__main__':
    print("Running lstm.py")
    #model = TransformerClassifier()
    print(model)
