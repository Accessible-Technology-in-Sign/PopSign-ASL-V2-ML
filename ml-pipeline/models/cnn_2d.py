import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN2D(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(SimpleCNN2D, self).__init__()
        
        # First Conv Layer with "SAME" padding and ReLU activation
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same')
        
        # First MaxPooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Second Conv Layer with ReLU activation
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='valid')
        
        # Second MaxPooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Third Conv Layer with ReLU activation
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='valid')
        
        # Second MaxPooling layer
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # # Third Conv Layer with ReLU activation
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='valid')
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, max_frames, input_features * num_coords)  # batch size of 1
            flattened_size = self._get_flattened_size(dummy_input)

        print("======> Size:", flattened_size)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_signs)
    
    def _get_flattened_size(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1).size(1)  # Flatten with batch dimension kept

    def forward(self, x):
        # Forward pass through Conv and Pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Maintain batch dimension
        
        # Fully connected layers with Dropout and Softmax
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x  # Logits returned for CrossEntropyLoss
    

class ComplexCNN2D(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(ComplexCNN2D, self).__init__()
        
        # First Convolution Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, input_features * num_coords), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, input_features * num_coords), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1))
        
        # Third Convolution Block with Residual Connection
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, input_features * num_coords), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1))
        
        self.residual = nn.Conv2d(64, 128, kernel_size=1, stride=1)  # To match dimensions for residual
        
        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, input_features * num_coords), padding='valid')
        # self.bn4 = nn.BatchNorm2d(256)
        # self.pool4 = nn.MaxPool2d(kernel_size=(3, 1))
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, max_frames, input_features * num_coords)
            flattened_size = self._get_flattened_size(dummy_input)

        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, num_signs)

    def _get_flattened_size(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block with residual connection
        residual = self.residual(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))) + residual)
        
        x = (F.relu((self.conv4(x))))
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block with residual connection
        residual = self.residual(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))) + residual)
        
        # Fourth block
        x = (F.relu((self.conv4(x))))
        
        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class WhisperInspiredClassifier(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(WhisperInspiredClassifier, self).__init__()

        cnn_out_channels = 256
        
        # 1D CNN to process the temporal dimension
        self.cnn = nn.Conv1d(
            in_channels=num_coords * input_features,  # Treat 63x3 as combined input channels
            out_channels=cnn_out_channels,
            kernel_size=7,
            stride=1,
            padding=3  # Preserve input size
        )
        self.cnn_bn = nn.BatchNorm1d(cnn_out_channels)
        self.cnn_activation = nn.ReLU()

        # 1D CNN to process the temporal dimension
        self.cnn2 = nn.Conv1d(
            in_channels=cnn_out_channels,  # Treat 63x3 as combined input channels
            out_channels=cnn_out_channels * 2,
            kernel_size=5,
            stride=1,
            padding=2  # Preserve input size
        )
        self.cnn_bn2 = nn.BatchNorm1d(cnn_out_channels * 2)
        self.cnn_activation2 = nn.ReLU()
        
        self.fc1 = nn.Linear(cnn_out_channels * 2 * max_frames, num_signs * num_coords)  # Flattened CNN output
        self.fc2 = nn.Linear(num_signs * num_coords, num_signs)

    def forward(self, x):
        # Input shape: [batch_size, sequence_length, feature_dim, channels]
        batch_size, seq_len, feature_dim, channels = x.size()
        x = x.permute(0, 3, 2, 1).reshape(batch_size, channels * feature_dim, seq_len)  # [batch, input_channels * feature_dim, sequence_length]
        
        # CNN
        x = self.cnn(x)  # [batch, cnn_out_channels, sequence_length]
        x = self.cnn_bn(x)
        x = self.cnn_activation(x)

        x = self.cnn2(x)  # [batch, cnn_out_channels, sequence_length]
        x = self.cnn_bn2(x)
        x = self.cnn_activation2(x)

        x = x.view(x.size(0), -1)  # Flatten to [batch_size, cnn_out_channels * 2 * sequence_length]
        
        # Fully connected layers
        x = self.fc1(x)  # [batch_size, 1024]
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

    
if __name__ == '__main__':
    print("Running cnn_2d.py")
    model = SimpleCNN2D(60, 21, 2, 563)
    print(model)