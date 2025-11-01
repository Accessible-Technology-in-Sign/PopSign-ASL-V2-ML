import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(SimpleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=128 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    

class ComplexLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(SimpleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=128 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    

class DoubleLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=256, # 256 -> 512
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=256 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=256, # 256 -> 512
            batch_first=True,
            bidirectional=True
        )

        self.lstm3 = nn.LSTM(
            input_size=256 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=256, # 256 -> 512
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(256 * 2, num_signs) # 256 -> 512 

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)

        x, _ = self.lstm3(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    
class ScaledLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(ScaledLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=290,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=290 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=290,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(290 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    


class TransformerClassifier(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        model_dim = input_features * num_coords  # Embedding and model dimension
        n_heads = 14  # Number of attention heads
        num_encoder_layers = 4  # Number of transformer layers
        num_classes = num_signs  # Number of output classes
        max_seq_length = max_frames  # Maximum sequence length
        super(TransformerClassifier, self).__init__()
        
        # self.embedding = nn.Embedding(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.randn(max_seq_length, model_dim))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=n_heads,
                dim_feedforward=num_signs,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        self.fc = nn.Linear(model_dim * 20, num_classes)
    
    def forward(self, x):
        # Input shape (batch_size, seq_length)
        
        # Embed the input tokens and add positional encodings
        # x = x + self.position_encoding[:x.size(1), :]
        
        # Reshape to (seq_length, batch_size, model_dim) for transformer input
        # x = x.permute(1, 0, 2)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        
        # Get the representation of the last token (or alternatively the mean across tokens)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, model_dim)
        
        # Pass through the final classification layer
        x = self.fc(x)   
        
        return x
    
class ComplexTransformerClassifier(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        model_dim = input_features * num_coords  # Embedding and model dimension
        n_heads = 14  # Number of attention heads
        num_encoder_layers = 10  # Number of transformer layers
        num_classes = num_signs  # Number of output classes
        max_seq_length = max_frames  # Maximum sequence length
        super(ComplexTransformerClassifier, self).__init__()
        
        # Embedding and positional encoding
        # self.embedding = nn.Embedding(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.randn(max_seq_length, model_dim))

        # Efficient transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=n_heads,
                dim_feedforward=num_signs,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        # Reduce the sequence by pooling
        # self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence length
        
        # Final classification layer
        self.fc = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        # x = x + self.position_encoding[:x.size(1), :]
        # x = x.permute(1, 0, 2)  # Transformer requires (seq_len, batch, model_dim)
        x = self.transformer_encoder(x)
        
        # Pool across the sequence dimension (seq_len -> 1)
        x = x[:,-1,:]
        # x = self.pool(x.squeeze(-1))  # Now (batch_size, model_dim)
        
        # Classification layer
        logits = self.fc(x)
        return logits
    

class TransformerCNNClassifier(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        model_dim = input_features * num_coords  # Embedding and model dimension
        n_heads = 14  # Number of attention heads
        num_encoder_layers = 4  # Number of transformer layers
        num_classes = num_signs  # Number of output classes
        max_seq_length = max_frames  # Maximum sequence length
        super(TransformerCNNClassifier, self).__init__()
        
        # self.embedding = nn.Embedding(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.randn(max_seq_length, model_dim))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=n_heads,
                dim_feedforward=num_signs,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5,1), padding='same')
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # # Second Convolution Block
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding='same')
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # # self.residual = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding='same')  # To match dimensions for residual
        
        # # Third Convolution Block with Residual Connection
        # # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, input_features * num_coords), padding='same')
        
        # # Calculate the flattened size dynamically
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, 1, max_frames, input_features * num_coords)
        #     flattened_size = self._get_flattened_size(dummy_input)

        # Fully Connected Layers
        self.fc1 = nn.Linear(42, 512)
        # self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_signs)


    def _get_flattened_size(self, x):
        x = self.pool1(F.relu((self.conv1(x))))
        # residual = self.residual(x)

        x = F.relu(self.pool2(self.conv2(x)))
        
        # Third block with residual connection
        # x = (F.relu((self.conv3(x))) + residual)
        
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.view(x.shape[0], 1, *x.shape[1:])


        # Forward pass through Conv and Pooling layers
        # x = self.pool1(F.relu(self.conv1(x)))
        # # residual = self.residual(x)

        # # x = self.pool2(F.relu(self.conv2(x)))
        # # x = self.pool3(F.relu(self.conv3(x)))


        # x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layers
        # x = x.view(x.size(0), -1)  # Maintain batch dimension
        
        # Fully connected layers with Dropout and Softmax
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)

        return x
   


class ProjectedLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(ProjectedLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=563,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            proj_size=256,
        )
        
        # Second bidirectional LSTM layer
        # self.lstm2 = nn.LSTM(
        #     input_size=256 * 2,  # 128 hidden units * 2 for bidirection
        #     hidden_size=1024,
        #     batch_first=True,
        #     bidirectional=True
        # )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        # x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x

class WorldTestLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs, cnn_out_channels=64, kernel_size=3):
        super(WorldTestLSTM, self).__init__()
        
        # 2D Convolutional layers
        self.cnn1 = nn.Conv2d(
            in_channels=1,  # Treat the input as single-channel 2D data
            out_channels=cnn_out_channels,
            kernel_size=(kernel_size, 1),  # Convolve across sequence length only
            stride=(1, 1)
        )
        self.cnn2 = nn.Conv2d(
            in_channels=cnn_out_channels,
            out_channels=cnn_out_channels,
            kernel_size=(kernel_size, 1),
            stride=(1, 1)
        )
        
        self.cnn_out_channels = cnn_out_channels
        self.kernel_size = kernel_size
        self.input_features = input_features
        self.num_coords = num_coords

        # LSTM layers
        lstm_input_size = cnn_out_channels * input_features * num_coords
        self.lstm1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=max_frames,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=max_frames * 2,
            hidden_size=max_frames,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm3 = nn.LSTM(
            input_size=max_frames * 2,
            hidden_size=max_frames,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(max_frames * 2, num_coords * input_features * num_signs)
        self.fc2 = nn.Linear(num_coords * input_features * num_signs, num_signs)

    def forward(self, x):
        # Input shape: [batch, seq_len, num_features]
        batch_size, seq_len, num_features = x.size()

        # Reshape input for CNNs: [batch, 1, seq_len, num_features]
        x = x.unsqueeze(1)

        # Apply CNN layers
        x = self.cnn1(x)  # [batch, cnn_out_channels, seq_len - kernel_size + 1, num_features]
        x = torch.relu(x)
        x = self.cnn2(x)  # [batch, cnn_out_channels, seq_len - 2 * (kernel_size - 1), num_features]
        x = torch.relu(x)

        # Reshape for LSTM: [batch, seq_len', cnn_out_channels * num_features]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len', cnn_out_channels, num_features]
        x = x.view(batch_size, x.size(1), -1)   # Flatten last two dimensions

        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)

        # Third LSTM layer
        x, _ = self.lstm3(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x



class SpatioTemporalTransformerClassifier(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        model_dim = 512
        n_heads = 128
        num_encoder_layers = 2
        num_classes = num_signs
        super(SpatioTemporalTransformerClassifier, self).__init__()
        self.num_coords = num_coords
        self.input_features = input_features
        self.max_frames = max_frames

        self.input_map = nn.Sequential(
            nn.Linear(input_features * num_coords, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(0.1),
        )
        
        self.spatial_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=n_heads,
                dim_feedforward=model_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        self.temporal_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=n_heads,
                dim_feedforward=model_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        self.fc = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        
        # x = x.view(x.size(0), -1, self.max_frames)
        x = self.input_map(x)
        x = self.spatial_transformer_encoder(x)
        x = self.temporal_transformer_encoder(x)

        x = x.sum(1) / x.shape[1]
        
        
        x = self.fc(x)   
        return x


if __name__ == '__main__':
    print("Running lstm.py")
    model = TransformerClassifier()
    print(model)