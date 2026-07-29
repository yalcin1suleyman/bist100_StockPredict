import numpy as np
import pandas as pd
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from importlib.machinery import SourceFileLoader

data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor
MinMaxScaler = data_module.MinMaxScaler

dl_module = SourceFileLoader("dl_models", "05_train_dl_models.py").load_module()
MLP = dl_module.MLP
CNN1D = dl_module.CNN1D
LSTMModel = dl_module.LSTMModel
GRUModel = dl_module.GRUModel
BiLSTMModel = dl_module.BiLSTMModel
CNNLSTM = dl_module.CNNLSTM
TransformerModel = dl_module.TransformerModel
EarlyStopping = dl_module.EarlyStopping
calculate_mape = dl_module.calculate_mape

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    early_stopping = EarlyStopping(patience=3)
    best_model_state = None
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
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

file_path = "data/bist100_data_interpolate.csv"
scenarios = ['raw', 'technical', 'all']
all_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

for scenario in scenarios:
    print(f"\n{'='*50}\nDL SENARYO: {scenario.upper()}\n{'='*50}")
    
    preprocessor = DataPreprocessor(file_path=file_path, window_size=10)
    
    df = preprocessor.load_and_clean_data()
    
    price_cols_to_drop = ['Close', 'Open', 'High', 'Low']
    preprocessor.feature_cols = [c for c in preprocessor.feature_cols if c not in price_cols_to_drop]
    raw_cols = ['Volume', 'USD_TRY', 'VIX', 'Gunluk_Getiri']
    
    if scenario == 'raw':
        preprocessor.feature_cols = [c for c in preprocessor.feature_cols if c in raw_cols]
    elif scenario == 'technical':
        preprocessor.feature_cols = [c for c in preprocessor.feature_cols if c not in raw_cols]
        
    X_train_dl_list, y_train_dl_list, X_test_dl_list, y_test_dl_list, y_test_unscaled_list = [], [], [], [], []
    train_df, test_df = preprocessor.time_series_split(df)
    
    preprocessor.scaler_X = MinMaxScaler()
    preprocessor.scaler_y = MinMaxScaler()
    
    X_train_full = preprocessor.scaler_X.fit_transform(train_df[preprocessor.feature_cols])
    y_train_full = preprocessor.scaler_y.fit_transform(train_df[[preprocessor.target_col]])
    X_test_full = preprocessor.scaler_X.transform(test_df[preprocessor.feature_cols])
    y_test_full = preprocessor.scaler_y.transform(test_df[[preprocessor.target_col]])
    
    for hisse in df['Hisse_Kodu'].unique():
        train_mask = (train_df['Hisse_Kodu'] == hisse).values
        test_mask = (test_df['Hisse_Kodu'] == hisse).values
        if train_mask.sum() <= preprocessor.window_size or test_mask.sum() <= preprocessor.window_size:
            continue
            
        X_tr, y_tr = preprocessor.create_sliding_window(X_train_full[train_mask], y_train_full[train_mask])
        X_te, y_te = preprocessor.create_sliding_window(X_test_full[test_mask], y_test_full[test_mask])
        
        X_train_dl_list.append(X_tr)
        y_train_dl_list.append(y_tr)
        X_test_dl_list.append(X_te)
        y_test_dl_list.append(y_te)
        y_test_unscaled_list.append(test_df[test_mask][[preprocessor.target_col]].values[preprocessor.window_size:].ravel())
        
    X_train_dl = np.vstack(X_train_dl_list)
    y_train_dl = np.vstack(y_train_dl_list)
    X_test_dl = np.vstack(X_test_dl_list)
    y_test_dl = np.vstack(y_test_dl_list)
    y_test_unscaled = np.concatenate(y_test_unscaled_list)
    scaler_y = preprocessor.scaler_y
    
    seq_len = X_train_dl.shape[1]
    input_size = X_train_dl.shape[2]
    
    val_size = int(len(X_train_dl) * (15 / 85))
    X_train_t = torch.tensor(X_train_dl[:-val_size], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_dl[:-val_size], dtype=torch.float32)
    X_val_t = torch.tensor(X_train_dl[-val_size:], dtype=torch.float32)
    y_val_t = torch.tensor(y_train_dl[-val_size:], dtype=torch.float32)
    X_test_t = torch.tensor(X_test_dl, dtype=torch.float32)
    
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    
    models = {
        'MLP': MLP(input_dim=seq_len * input_size),
        'CNN': CNN1D(in_channels=input_size, seq_len=seq_len),
        'LSTM': LSTMModel(input_size=input_size),
        'GRU': GRUModel(input_size=input_size),
        'BiLSTM': BiLSTMModel(input_size=input_size),
        'CNN-LSTM': CNNLSTM(in_channels=input_size, seq_len=seq_len),
        'Transformer': TransformerModel(input_size=input_size, seq_len=seq_len)
    }
    
    for name, model in models.items():
        print(f"[{name}] egitiliyor ({scenario})...")
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        start_time = time.time()
        model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=8)
        end_time = time.time()
        
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_t.to(device)).cpu().numpy()
            
        y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled).ravel()
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
        mape = calculate_mape(y_test_unscaled, y_pred_unscaled)
        r2 = r2_score(y_test_unscaled, y_pred_unscaled)
        
        all_results.append({
            'Scenario': scenario,
            'Model': name,
            'Model_Scenario': f"{name} ({scenario})",
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'R²': r2,
            'Time (s)': end_time - start_time
        })

results_df = pd.DataFrame(all_results)
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/Table_5_3_DL_Performance_Scenarios.csv", index=False)
print("Bitti!")
