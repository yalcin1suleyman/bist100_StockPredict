import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
import math

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from importlib.machinery import SourceFileLoader
data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN1D(nn.Module):
    def __init__(self, in_channels, seq_len):
        super(CNN1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # Calculate flattened size
        conv_out_len = seq_len - 2 + 1
        pool_out_len = conv_out_len // 2
        self.fc_input_dim = 64 * pool_out_len
        
        self.fc1 = nn.Linear(self.fc_input_dim, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [batch, channels, seq]
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, input_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CNNLSTM(nn.Module):
    def __init__(self, in_channels, seq_len):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # Output of pool will be fed to LSTM. 
        # Output shape of pool: [batch, 64, pool_out_len]
        # We need to permute back to [batch, seq, channels] for LSTM -> [batch, pool_out_len, 64]
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [batch, channels, seq]
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1) # [batch, seq, channels]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model gracefully if it happens (though usually it's even)
        pe[:, 1::2] = torch.cos(position * div_term)[:pe[:, 1::2].size(0)] 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_len):
        super(TransformerModel, self).__init__()
        # If input_size is not even, we can project it to an even dimension for simplicity
        self.d_model = 32
        self.input_projection = nn.Linear(input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global average pooling
        x = self.fc(x)
        return x

def plot_horizontal_bar(data, metric_name, file_name, title):
    ascending = False if metric_name == 'R²' else True
    data_sorted = data.sort_values(by=metric_name, ascending=ascending).head(15)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(data_sorted))
    bars = plt.barh(data_sorted['Model'], data_sorted[metric_name], color=colors)
    plt.xlabel(metric_name)
    plt.ylabel('Derin Öğrenme Modelleri')
    plt.title(title)
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    early_stopping = EarlyStopping(patience=10)
    
    best_model_state = None
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def train_and_evaluate_dl_models(file_path):
    print("--- 05: DERİN ÖĞRENME MODELLERİ (DL) [PyTorch] ---")
    
    preprocessor = DataPreprocessor(file_path=file_path, window_size=10)
    X_train_dl, y_train_dl, X_test_dl, y_test_dl, y_test_unscaled, scaler_y, feature_cols = preprocessor.get_dl_data(scaler_type='minmax')
    
    seq_len = X_train_dl.shape[1]
    input_size = X_train_dl.shape[2]
    
    # 10% validation split from training data
    val_size = int(len(X_train_dl) * 0.1)
    
    X_train_t = torch.tensor(X_train_dl[:-val_size], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_dl[:-val_size], dtype=torch.float32)
    X_val_t = torch.tensor(X_train_dl[-val_size:], dtype=torch.float32)
    y_val_t = torch.tensor(y_train_dl[-val_size:], dtype=torch.float32)
    X_test_t = torch.tensor(X_test_dl, dtype=torch.float32)
    
    batch_size = 32
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    models = {
        'MLP': MLP(input_dim=seq_len * input_size),
        'CNN': CNN1D(in_channels=input_size, seq_len=seq_len),
        'LSTM': LSTMModel(input_size=input_size),
        'GRU': GRUModel(input_size=input_size),
        'BiLSTM': BiLSTMModel(input_size=input_size),
        'CNN-LSTM': CNNLSTM(in_channels=input_size, seq_len=seq_len),
        'Transformer': TransformerModel(input_size=input_size, seq_len=seq_len)
    }
    
    all_results = []
    predictions_dict = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    for name, model in models.items():
        print(f"\n[{name}] modeli eğitiliyor...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)
        end_time = time.time()
        
        # Prediction
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_t.to(device)).cpu().numpy()
            
        y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled).ravel()
        
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
        mape = calculate_mape(y_test_unscaled, y_pred_unscaled)
        r2 = r2_score(y_test_unscaled, y_pred_unscaled)
        
        all_results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'R²': r2,
            'Time (s)': end_time - start_time
        })
        predictions_dict[name] = y_pred_unscaled
        print(f"[{name}] tamamlandı. R²: {r2:.4f}, RMSE: {rmse:.4f}")

    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("Tablo 5.3. Derin Öğrenme Modelleri Performansı")
    print("="*60)
    print(results_df[['Model', 'R²', 'RMSE', 'MAE']].set_index('Model').round(4))
    print("="*60)
    
    results_df.to_csv("outputs/Table_5_3_DL_Performance.csv", index=False)
    
    # Grafik Çizimleri
    plot_horizontal_bar(results_df, 'R²', 'outputs/Fig_5_13_R2_Comparison_DL.png', 'Derin Öğrenme Modelleri R² Karşılaştırması')
    plot_horizontal_bar(results_df, 'RMSE', 'outputs/Fig_5_14_RMSE_Comparison_DL.png', 'Derin Öğrenme Modelleri RMSE Karşılaştırması')
    
    # Save predictions for statistical tests (e.g. Diebold-Mariano)
    pred_df = pd.DataFrame(predictions_dict)
    pred_df['Actual'] = y_test_unscaled
    pred_df.to_csv("outputs/dl_predictions.csv", index=False)

if __name__ == "__main__":
    file_path = "data/bist100_data_interpolate.csv"
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı.")
    else:
        train_and_evaluate_dl_models(file_path)
