import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import io
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib 
import os
import time
import yfinance as yf

# ====================================================
# 1. CONFIG & STYLE
# ====================================================
st.set_page_config(
    page_title="NeuroStock - AI Stock Analysis",
    layout="wide",
    page_icon="favicon.png"
)

def add_custom_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #f8f9fa 0%, #e3f2fd 100%);
            border-right: 1px solid #d0d7de;
        }
        header[data-testid="stHeader"] {
            background-image: linear-gradient(90deg, #ffffff 0%, #e1f5fe 100%);
            border-bottom: 1px solid #d0d7de;
        }
        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# ====================================================
# 2. HELPER FUNCTIONS (MATH & LOGIC)
# ====================================================

def evaluate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Filter NaN/Infinity sebelum hitung
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'Accuracy': 0}

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # MAPE
    mask_nonzero = y_true_clean != 0
    y_true_mape = y_true_clean[mask_nonzero]
    y_pred_mape = y_pred_clean[mask_nonzero]
    
    try:
        if len(y_true_mape) > 0:
            mape = mean_absolute_percentage_error(y_true_mape, y_pred_mape) * 100
        else:
            mape = 0
    except:
        mape = 0
        
    accuracy = 100.0 - mape
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Accuracy': accuracy}

def create_dataset(X_scaled, y_scaled, time_step=60):
    """
    Membuat sequence data untuk LSTM/CNN.
    X diambil dari scaler_X (6 fitur), y diambil dari scaler_y (1 fitur).
    """
    dataX, dataY = [], []
    for i in range(len(X_scaled) - time_step):
        # Ambil window 60 hari dari X (6 fitur)
        dataX.append(X_scaled[i:(i + time_step), :])
        # Ambil target hari ke-61 dari y (1 fitur - Close)
        dataY.append(y_scaled[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ====================================================
# 3. FORECASTING LOGIC & MODELS
# ====================================================
def run_predictions_menu1(models, processed_data):
    scaler_y = processed_data["scaler_y"] # Scaler khusus Close Price
    preds = {}
    metrics = []
    
    for name, model in models.items():
        # 1. Predict (Output 1 nilai skala 0-1)
        p_train_scaled = model.predict(processed_data["X_train"], verbose=0)
        p_test_scaled = model.predict(processed_data["X_test"], verbose=0)
        
        # 2. Inverse Transform
        p_train_inv = scaler_y.inverse_transform(p_train_scaled)
        p_test_inv = scaler_y.inverse_transform(p_test_scaled)
        
        # 3. Simpan
        preds[name] = {"train": p_train_inv, "test": p_test_inv}
        
        # 4. Metrics
        m = evaluate_metrics(processed_data["y_test_true"], p_test_inv)
        m['Model'] = name
        metrics.append(m)
        
    return preds, metrics

def generate_future_forecast(model, last_sequence, n_days, scaler_y):
    forecast_scaled = []
    current_step = last_sequence.copy() 
    
    # Deteksi Otomatis Jumlah Fitur
    n_features = current_step.shape[2] 
    
    # Tentukan di mana posisi kolom 'Close'
    if n_features > 1:
        close_col_idx = 3 # Multivariate (Menu 2)
    else:
        close_col_idx = 0 # Univariate (Menu 1)
    
    for _ in range(n_days):
        pred = model.predict(current_step, verbose=0) 
        forecast_scaled.append(pred[0, 0])
        
        next_step_features = current_step[:, -1, :].copy()
        next_step_features[0, close_col_idx] = pred[0, 0]
        next_step_reshaped = next_step_features.reshape(1, 1, n_features)
        current_step = np.append(current_step[:, 1:, :], next_step_reshaped, axis=1)

    forecast_arr = np.array(forecast_scaled).reshape(-1, 1)
    forecast_final = scaler_y.inverse_transform(forecast_arr)
    return forecast_final.flatten()

# --- DEFINISI ARSITEKTUR MODEL (3 JENIS) ---

def build_lstm_model(input_shape):
    model = Sequential()
    # LSTM Layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1)) # Output
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    # CNN Layer
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1)) # Output
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Hybrid tetap sama (jangan dihapus)
def build_general_hybrid_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=96, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def process_upload_and_train(df_raw, col_map):
    # 1. Rename & Mapping (Sama seperti sebelumnya)
    rename_dict = {
        col_map['Date']: 'Date', col_map['Open']: 'Open', col_map['High']: 'High',
        col_map['Low']: 'Low', col_map['Close']: 'Close', col_map['Volume']: 'Volume'
    }
    try:
        df = df_raw.rename(columns=rename_dict)[list(rename_dict.values())]
    except KeyError as e:
        st.error(f"Error Mapping: {e}"); return None
    
    # 2. Cleaning (Sama seperti kode "Smart Cleaning" sebelumnya)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    features_raw = ['Open', 'High', 'Low', 'Close', 'Volume']
    def clean_number_str(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().replace('"', '').replace("'", "").replace('Rp', '').replace(' ', '')
        if s == '-' or s == '': return np.nan
        return s

    for c in features_raw:
        df[c] = df[c].apply(clean_number_str)

    # Deteksi Format Angka
    sample_close = df['Close'].dropna().head(50).astype(str).tolist()
    vote_us = 0; vote_id = 0
    for s in sample_close:
        if ',' in s and '.' in s:
            if s.find(',') < s.find('.'): vote_us += 1
            elif s.find('.') < s.find(','): vote_id += 1
        elif '.' in s: vote_us += 0.5 
    
    use_indo_format = vote_id > vote_us
    for c in features_raw:
        if use_indo_format: df[c] = df[c].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        else: df[c] = df[c].str.replace(',', '', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['Volume'] = df['Volume'].fillna(0)
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    for c in ['Open', 'High', 'Low']: df[c] = df[c].fillna(df['Close'])

    if len(df) < 100: st.error("Data terlalu sedikit."); return None

    # 3. RSI & Scaling
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean(); avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss; df['RSI'] = 100 - (100 / (1 + rs))
    df = df.iloc[14:].reset_index(drop=True)

    features_final = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_X.fit_transform(df[features_final].values)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(df[['Close']].values)

    time_step = 60
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), :])
        y.append(scaled_data[i + time_step, 3]) # 3 is Close
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # =========================================================
    # 4. TRAINING KOMPETITIF (3 MODEL: CNN, LSTM, HYBRID)
    # =========================================================
    
    # Dictionary untuk menyimpan hasil semua model
    trained_models = {}
    preds = {}
    metrics_list = []
    
    # Callback ProgressBar
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress_bar, start_progress, end_progress, epochs):
            self.p_bar = progress_bar
            self.start = start_progress
            self.end = end_progress
            self.epochs = epochs
        def on_epoch_end(self, epoch, logs=None):
            p = self.start + ((epoch + 1) / self.epochs) * (self.end - self.start)
            self.p_bar.progress(min(p, 1.0))

    # Definisi Pipeline
    model_configs = [
        ("CNN", build_cnn_model),
        ("LSTM", build_lstm_model),
        ("Hybrid", build_general_hybrid_model)
    ]
    
    main_progress = st.progress(0)
    total_steps = len(model_configs)
    
    for idx, (name, build_fn) in enumerate(model_configs):
        with st.spinner(f"⏳ ({idx+1}/{total_steps}) Melatih Model {name}..."):
            
            # Build & Train
            model = build_fn((time_step, 6))
            
            # Progress bar logic (0.0 - 0.33, 0.33 - 0.66, 0.66 - 1.0)
            start_p = idx / total_steps
            end_p = (idx + 1) / total_steps
            
            model.fit(
                X_train, y_train, validation_data=(X_test, y_test),
                epochs=15, # Epoch dikurangi sedikit agar tidak terlalu lama menunggu 3 model
                batch_size=32, verbose=0,
                callbacks=[StreamlitCallback(main_progress, start_p, end_p, 15)]
            )
            
            # Predict & Evaluasi
            p_train = model.predict(X_train, verbose=0)
            p_test = model.predict(X_test, verbose=0)
            
            p_train_inv = scaler_y.inverse_transform(p_train)
            p_test_inv = scaler_y.inverse_transform(p_test)
            y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            m = evaluate_metrics(y_test_inv, p_test_inv)
            m['Model'] = name # Tagging nama model
            
            # Simpan Hasil
            trained_models[name] = model
            metrics_list.append(m)
            preds[name] = {"train": p_train_inv, "test": p_test_inv}
            
    c_done, c_space = st.columns([1, 1])
    c_done.success("🎉 Selesai! Semua model berhasil dilatih.")
    
    return {
        "processed_data": {
            "dates_full": df['Date'], 
            "dates_test": df['Date'].iloc[time_step+train_size:],
            "y_full_true": df['Close'].values,
            "y_test_true": scaler_y.inverse_transform(y_test.reshape(-1, 1)),
            "rsi_values": df['RSI'].values,
            "split_date": df['Date'].iloc[int(len(df)*0.8)],
            "scaler_y": scaler_y, 
            "last_sequence": scaled_data[-time_step:].reshape(1, time_step, 6),
            "last_date": df['Date'].iloc[-1]
        },
        "metrics": metrics_list, # List of dicts (CNN, LSTM, Hybrid)
        "preds": preds,          # Dict of dicts
        "model": trained_models  # Dict of models
    }
    
    try:
        df = df_raw.rename(columns=rename_dict)[list(rename_dict.values())]
    except KeyError as e:
        st.error(f"Error Mapping Kolom: {e}. Pastikan semua kolom terpilih.")
        return None
    
    # 2. Cleaning Tanggal
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # 3. CLEANING DATA PINTAR (Handle Kutip, Koma, dan Strip -)
    cols_price = ['Open', 'High', 'Low', 'Close']
    col_vol = 'Volume'
    
    # Fungsi pembersih angka string
    def clean_number_str(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        # Hapus karakter pengganggu: kutip, Rp, spasi
        s = s.replace('"', '').replace("'", "").replace('Rp', '').replace(' ', '')
        # Handle tanda strip (-)
        if s == '-' or s == '': return np.nan
        return s

    # Terapkan pembersih string dasar
    for c in cols_price + [col_vol]:
        df[c] = df[c].apply(clean_number_str)

    # Logika deteksi format angka (US vs Indo)
    # Kita cek sampel data dari kolom Close untuk menentukan format
    sample_close = df['Close'].dropna().head(50).astype(str).tolist()
    
    # Hitung vote format
    vote_us = 0 # Format 1,200.00
    vote_id = 0 # Format 1.200,00
    
    for s in sample_close:
        if ',' in s and '.' in s:
            # Jika koma muncul sebelum titik (1,234.56) -> US
            if s.find(',') < s.find('.'): vote_us += 1
            # Jika titik muncul sebelum koma (1.234,56) -> Indo
            elif s.find('.') < s.find(','): vote_id += 1
        elif '.' in s: # Cuma ada titik (1234.56 atau 1.234)
            # Asumsi default US kalau cuma titik, kecuali ribuan
            vote_us += 0.5 
    
    # Tentukan strategy cleaning
    use_indo_format = vote_id > vote_us
    
    for c in cols_price + [col_vol]:
        if use_indo_format:
            # Format Indo: Hapus titik, ganti koma dengan titik
            df[c] = df[c].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        else:
            # Format US: Hapus koma
            df[c] = df[c].str.replace(',', '', regex=False)
            
        # Konversi ke angka
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 4. HANDLING KHUSUS VOLUME
    # Jangan hapus baris jika Volume NaN, tapi isi dengan 0
    df[col_vol] = df[col_vol].fillna(0)
    
    # 5. HANDLING HARGA
    # Hapus baris HANYA jika 'Close' kosong (Data harga wajib ada)
    before_len = len(df)
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    
    # Fitur Fill Forward: Jika Open/High/Low kosong, isi dengan Close
    for c in ['Open', 'High', 'Low']:
        df[c] = df[c].fillna(df['Close'])
        
    after_len = len(df)
    if before_len != after_len:
        st.warning(f"⚠️ {before_len - after_len} baris data harga kosong dibersihkan.")

    if len(df) < 100:
        st.error(f"Data valid terlalu sedikit ({len(df)} baris). Cek format file Anda.")
        return None

    # 6. AUTO-GENERATE RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Isi NaN awal RSI dengan 50 (Netral) atau drop
    df = df.iloc[14:].reset_index(drop=True) # Drop 14 hari pertama

    # 7. Scaling
    features_final = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_X.fit_transform(df[features_final].values)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(df[['Close']].values)
    
    # 8. Create Dataset
    time_step = 60
    if len(scaled_data) < time_step + 10:
        st.error("Data tidak cukup untuk windowing.")
        return None

    X, y = [], []
    target_idx = 3 # Close
    
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), :])
        y.append(scaled_data[i + time_step, target_idx])
        
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 9. TRAINING
    with st.spinner("⏳ Melatih Model Hybrid (OHLCV + RSI)..."):
        model = build_general_hybrid_model((time_step, 6))
        
        progress_bar = st.progress(0)
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / 20
                progress_bar.progress(min(progress, 1.0))
                
        model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=20, batch_size=32, verbose=0, callbacks=[StreamlitCallback()]
        )
        st.success("Training Selesai!")
        
    p_train = model.predict(X_train)
    p_test = model.predict(X_test)
    
    p_train_inv = scaler_y.inverse_transform(p_train)
    p_test_inv = scaler_y.inverse_transform(p_test)
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = evaluate_metrics(y_test_inv, p_test_inv)
    metrics['Model'] = "Hybrid (6-Input)"
    
    preds = {"NewModel": {"train": p_train_inv, "test": p_test_inv}}
    
    return {
        "processed_data": {
            "dates_full": df['Date'], 
            "dates_test": df['Date'].iloc[time_step+train_size:],
            "y_full_true": df['Close'].values,
            "y_test_true": y_test_inv,
            "rsi_values": df['RSI'].values,
            "split_date": df['Date'].iloc[int(len(df)*0.8)],
            "scaler_y": scaler_y, 
            "last_sequence": scaled_data[-time_step:].reshape(1, time_step, 6),
            "last_date": df['Date'].iloc[-1]
        },
        "metrics": metrics, "preds": preds, "model": {"NewModel": model}
    }

def load_thesis_resources(ticker, scenario):
    scen_tag = "10y" if "10" in scenario else "5y"
    tick_tag = ticker.lower()
    
    base_model = "models/"
    path_scaler_X = f"scalers/scaler_6fitur_{scen_tag}_{tick_tag}.pkl"
    path_scaler_y = f"scalers/scaler_target_{scen_tag}_{tick_tag}.pkl"
    csv_path = f"data/{tick_tag}_cleaned_data_master.csv"

    # 1. LOAD DATA
    try:
        df = pd.read_csv(csv_path)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
        
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom hilang di CSV: {missing_cols}. Pastikan CSV hasil preprocessing Colab sudah diupload.")
            return None
            
        for col in features:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna().reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        
        X_values = df[features].values 
        y_values = df[['Close']].values 
        
    except Exception as e:
        st.error(f"Gagal memproses data CSV: {e}")
        return None

    # 2. LOAD SCALERS
    try:
        if not os.path.exists(path_scaler_X) or not os.path.exists(path_scaler_y):
            st.error("File Scaler (X atau y) belum lengkap di folder 'scalers/'.")
            return None
            
        scaler_X = joblib.load(path_scaler_X)
        scaler_y = joblib.load(path_scaler_y)
        
    except Exception as e:
        st.error(f"Gagal load Scaler. Error: {e}")
        return None

    # 3. LOAD MODELS
    try:
        models = {}
        models['CNN'] = load_model(f"{base_model}model_cnn_{scen_tag}_{tick_tag}.keras", compile=False)
        models['LSTM'] = load_model(f"{base_model}model_lstm_{scen_tag}_{tick_tag}.keras", compile=False)
        models['Hybrid'] = load_model(f"{base_model}model_hybrid_{scen_tag}_{tick_tag}.keras", compile=False)
    except Exception as e:
        st.error(f"Gagal load Model Keras. Error: {e}")
        return None

    # 4. PRE-PROCESS (TRANSFORM)
    try:
        X_scaled = scaler_X.transform(X_values)
        y_scaled = scaler_y.transform(y_values)

        time_step = 60
        split_idx = int(len(X_scaled) * 0.8) 

        X_train, y_train = create_dataset(X_scaled[:split_idx], y_scaled[:split_idx], time_step)
        X_test, y_test = create_dataset(X_scaled[split_idx:], y_scaled[split_idx:], time_step)

        dates_full = df['Date']
        dates_train = dates_full[time_step:split_idx]
        dates_test = dates_full[split_idx+time_step:]
        
        y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        processed_data = {
            "dates_full": dates_full, "dates_test": dates_test, "dates_train": dates_train,
            "y_full_true": df['Close'].values,
            "y_test_true": y_test_true,
            "X_train": X_train, "X_test": X_test,
            "split_date": dates_full.iloc[split_idx],
            "scaler_y": scaler_y, 
            "last_sequence": X_scaled[-time_step:].reshape(1, time_step, 6),
            "last_date": dates_full.iloc[-1]
        }
        return processed_data, models

    except Exception as e:
        st.error(f"Error proses dataset: {e}")
        return None

# ====================================================
# 4. PLOTTING FUNCTIONS
# ====================================================
def render_comparison_plots(processed_data, predictions_inv, title_suffix=""):
    # Fungsi Helper untuk membuat Figure
    def create_fig(plot_type, preds_dict):
        fig = go.Figure()
        colors = {"Actual": "#2C3E50", "CNN": "#FF8C00", "LSTM": "#00C853", "Hybrid": "#6200EA", "NewModel": "#D32F2F"}
        
        # 1. Plot Data Asli (Actual)
        if plot_type == "Combined":
            # Plot seluruh data history
            fig.add_trace(go.Scatter(
                x=processed_data["dates_full"], 
                y=processed_data["y_full_true"], 
                name="Actual Price", 
                line=dict(color=colors["Actual"], width=2), 
                opacity=0.6
            ))
            
            # Garis Vertikal (Split)
            split_date = processed_data["split_date"]
            try:
                if hasattr(split_date, 'timestamp'):
                    split_date_numeric = split_date.timestamp() * 1000 
                else:
                    split_date_numeric = pd.to_datetime(split_date).timestamp() * 1000
            except:
                split_date_numeric = split_date 

            fig.add_vline(x=split_date_numeric, line_dash="dash", line_color="red", annotation_text="Split Data")

        elif plot_type == "Test":
            # Plot hanya data testing
            fig.add_trace(go.Scatter(
                x=processed_data["dates_test"], 
                y=processed_data["y_test_true"].flatten(), 
                name="Actual (Test)", 
                line=dict(color=colors["Actual"], width=2.5)
            ))

        # 2. Persiapan Sumbu X (Tanggal)
        # Ambil dates_train dengan aman (Support untuk Menu 1 dan Menu 2)
        if "dates_train" in processed_data:
            dates_train = np.array(processed_data["dates_train"])
        else:
            time_step = 60
            sample_train_len = len(list(preds_dict.values())[0]["train"])
            dates_train = np.array(processed_data["dates_full"])[time_step : time_step + sample_train_len]
            
        dates_test = np.array(processed_data["dates_test"])

        # 3. Plot Garis Prediksi Model
        for model_name, data_pred in preds_dict.items():
            color = colors.get(model_name, "#333333") 
            y_train = data_pred["train"].flatten()
            y_test = data_pred["test"].flatten()
            
            if plot_type == "Combined":
                # --- FIX UTAMA: Pisahkan Trace Train & Test agar tidak ditarik garis lurus penyambung ---
                
                # Trace 1: Prediksi Training
                fig.add_trace(go.Scatter(
                    x=dates_train[:len(y_train)], 
                    y=y_train, 
                    name=f"{model_name} Prediction", 
                    line=dict(color=color, width=2)
                ))
                
                # Trace 2: Prediksi Testing (Warna sama, sembunyikan nama di legend agar rapi)
                fig.add_trace(go.Scatter(
                    x=dates_test[:len(y_test)], 
                    y=y_test, 
                    name=f"{model_name} Test", 
                    line=dict(color=color, width=2),
                    showlegend=False 
                ))
                
            elif plot_type == "Test":
                fig.add_trace(go.Scatter(
                    x=dates_test[:len(y_test)], 
                    y=y_test, 
                    name=f"{model_name}", 
                    line=dict(color=color, width=2)
                ))

        fig.update_layout(
            title=f"{plot_type} Chart {title_suffix}", 
            xaxis_title="Date", 
            yaxis_title="Price (IDR)", 
            hovermode="x unified", 
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # --- TABS ---
    tab_comb, tab_test = st.tabs(["📈 Full History Overview", "🔍 Test Phase Comparison"])
    
    with tab_comb:
        st.plotly_chart(create_fig("Combined", predictions_inv), use_container_width=True)
        
    with tab_test: 
        st.plotly_chart(create_fig("Test", predictions_inv), use_container_width=True)

def render_eda_dashboard(df, ticker):
    st.markdown(f"### 🔎 Exploratory Data Analysis & Preprocessing: {ticker}")
    
    # ==========================================
    # 0. RINGKASAN DATA (Hanya 3 Metrik)
    # ==========================================
    st.write("#### 📌 Ringkasan Dataset")
    
    total_rows = len(df)
    start_date = df['Date'].min().strftime('%d %b %Y')
    end_date = df['Date'].max().strftime('%d %b %Y')
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Data Valid", f"{total_rows:,} Baris", help="Jumlah hari perdagangan yang digunakan.")
    m2.metric("Awal Periode", f"{start_date}")
    m3.metric("Akhir Periode", f"{end_date}")
    
    st.markdown("---")

    # ==========================================
    # 1. VISUALISASI HARGA (CANDLESTICK) & VOLUME
    # ==========================================
    st.write("#### 1. Pergerakan Harga & Volume")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume', marker_color='rgba(100, 150, 250, 0.6)'
    ), row=2, col=1)

    fig.update_layout(
        title=f"Price & Volume History - {ticker}",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 2. INDIKATOR RSI
    # ==========================================
    if 'RSI' in df.columns:
        st.write("#### 2. Indikator RSI (Relative Strength Index)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple', width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig_rsi.update_layout(height=300, margin=dict(t=30, b=20), yaxis_title="RSI Value")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ==========================================
    # 3. KORELASI ANTAR FITUR
    # ==========================================
    st.write("#### 3. Korelasi Antar Fitur")
    c1, c2 = st.columns([1, 1]) 
    with c1:
        cols_corr = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'RSI' in df.columns:
            cols_corr.append('RSI')
            
        corr = df[cols_corr].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='Viridis', text=np.round(corr.values, 2), texttemplate="%{text}"
        ))
        fig_corr.update_layout(height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================
    # 4. STATISTIK DESKRIPTIF
    # ==========================================
    st.write("#### 4. Statistik Deskriptif")
    cols_stats = ['Close', 'Volume']
    if 'RSI' in df.columns:
        cols_stats.append('RSI')
        
    stats = df[cols_stats].describe().T
    st.dataframe(stats.style.format("{:,.2f}"), use_container_width=True)

    # ==========================================
    # 5. VISUALISASI PIPELINE PREPROCESSING
    # ==========================================
    st.markdown("---")
    st.write("#### ⚙️ 5. Pipeline Preprocessing Data")
    st.caption("Alur transformasi data sebelum dimasukkan ke dalam arsitektur Deep Learning. Klik tab di bawah untuk melihat hasil masing-masing proses.")
    
    # A. Visualisasi Flowchart dengan HTML/CSS (5 Tahapan)
    st.markdown("""
    <div style='display: flex; justify-content: space-between; text-align: center; margin-bottom: 20px; font-family: sans-serif;'>
        <div style='flex: 1; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 2px solid #e9ecef;'>
            <h3 style='margin:0; font-size: 18px;'>🧹</h3>
            <b style='color:#2c3e50; font-size: 13px;'>1. Cleaning</b>
        </div>
        <div style='font-size: 20px; color: #adb5bd; padding: 10px 2px;'>➔</div>
        <div style='flex: 1; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 2px solid #e9ecef;'>
            <h3 style='margin:0; font-size: 18px;'>📈</h3>
            <b style='color:#2c3e50; font-size: 13px;'>2. Feature Eng.</b>
        </div>
        <div style='font-size: 20px; color: #adb5bd; padding: 10px 2px;'>➔</div>
        <div style='flex: 1; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 2px solid #e9ecef;'>
            <h3 style='margin:0; font-size: 18px;'>✂️</h3>
            <b style='color:#2c3e50; font-size: 13px;'>3. Splitting</b>
        </div>
        <div style='font-size: 20px; color: #adb5bd; padding: 10px 2px;'>➔</div>
        <div style='flex: 1; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 2px solid #e9ecef;'>
            <h3 style='margin:0; font-size: 18px;'>📏</h3>
            <b style='color:#2c3e50; font-size: 13px;'>4. Normalisasi</b>
        </div>
        <div style='font-size: 20px; color: #adb5bd; padding: 10px 2px;'>➔</div>
        <div style='flex: 1; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 2px solid #e9ecef;'>
            <h3 style='margin:0; font-size: 18px;'>🪟</h3>
            <b style='color:#2c3e50; font-size: 13px;'>5. Windowing</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # B. Interaktif Tabs untuk Visualisasi per Proses
    t1, t2, t3, t4, t5 = st.tabs([
        "🧹 1. Cleaning", 
        "📈 2. Feature Eng.", 
        "✂️ 3. Data Splitting", 
        "📏 4. Normalisasi", 
        "🪟 5. Sliding Window"
    ])
    
    # --- TAB 1: DATA CLEANING ---
    with t1:
        st.write("**🔍 Hasil Data Cleaning**")
        st.write("Menghapus baris kosong (NaN) dan menyeragamkan format angka menjadi desimal standar.")
        cols_clean = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        st.dataframe(df[cols_clean].head(5).style.format({
            "Open": "{:,.2f}", "High": "{:,.2f}", "Low": "{:,.2f}", "Close": "{:,.2f}", "Volume": "{:,.0f}"
        }), use_container_width=True)

    # --- TAB 2: FEATURE ENGINEERING ---
    with t2:
        st.write("**🔍 Hasil Feature Engineering (Penambahan RSI)**")
        st.write("Sistem otomatis menghitung nilai **RSI 14-Hari** dari kolom harga Close untuk membantu model menangkap momentum overbought/oversold.")
        if 'RSI' in df.columns:
            st.dataframe(df[['Date', 'Close', 'RSI']].tail(5).style.format({
                "Close": "{:,.2f}", "RSI": "{:.2f}"
            }), use_container_width=True)
            
    # --- TAB 3: DATA SPLITTING ---
    with t3:
        st.write("**🔍 Visualisasi Pembagian Data (80% Train, 20% Test)**")
        split_idx = int(len(df) * 0.8)
        train_len = split_idx
        test_len = len(df) - split_idx
        
        # Plotly Donut Chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Data Latih (Training)', 'Data Uji (Testing)'], 
            values=[train_len, test_len], 
            hole=.4,
            marker_colors=['#2980b9', '#e74c3c']
        )])
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
        
        c_pie, c_desc = st.columns([1, 1])
        with c_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        with c_desc:
            st.info(f"**Total Data:** {len(df)} Baris")
            st.success(f"**Training:** {train_len} Baris (Masa Lalu -> Digunakan untuk belajar)")
            st.error(f"**Testing:** {test_len} Baris (Masa Depan -> Disembunyikan untuk ujian)")

    # --- TAB 4: NORMALISASI ---
    with t4:
        st.write("**🔍 Bukti Transformasi Normalisasi (MinMaxScaler)**")
        cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'RSI' in df.columns: cols_to_scale.append('RSI')
            
        scaler_viz = MinMaxScaler()
        scaled_array = scaler_viz.fit_transform(df[cols_to_scale])
        df_scaled = pd.DataFrame(scaled_array, columns=cols_to_scale, index=df.index)
        
        c_bef, c_aft = st.columns(2)
        with c_bef:
            st.caption("🔴 **Sebelum:** Nilai asli bervariasi luas")
            st.dataframe(df[cols_to_scale].tail(4).style.format("{:,.1f}"), use_container_width=True)
        with c_aft:
            st.caption("🟢 **Sesudah:** Seluruh nilai direntang 0.0 - 1.0")
            st.dataframe(df_scaled.tail(4).style.format("{:.4f}"), use_container_width=True)

    # --- TAB 5: SLIDING WINDOW ---
    with t5:
        st.write("**🔍 Konsep Sliding Window (Time-Steps: 60)**")
        st.info("Karena model membaca pola deret waktu, data diubah menjadi format urutan (Sequence). Model membaca data historis **60 hari ke belakang** untuk memprediksi **1 hari ke depan**.")
        
        # Ilustrasi Sederhana menggunakan Tabel
        st.write("Ilustrasi Matriks X (Input) dan Y (Target):")
        ill_data = {
            "Input X (Model Belajar)": ["Hari 1 s/d Hari 60", "Hari 2 s/d Hari 61", "Hari 3 s/d Hari 62"],
            "Target Y (Prediksi)": ["Harga Close Hari 61", "Harga Close Hari 62", "Harga Close Hari 63"]
        }
        st.table(pd.DataFrame(ill_data))
        
        st.caption(f"Jika total data ada {len(df)}, maka akan dihasilkan {len(df)-60} baris urutan sequence untuk dilatih.")
    
def render_forecast_ui(processed_data, metrics_summary, models):
    st.markdown("---")
    st.header("🔮 Forecasting")
    
    # 1. Pilih Model Terbaik (Smart Selection Logic)
    if isinstance(metrics_summary, list): 
        best_model_info = min(metrics_summary, key=lambda x: x['RMSE'])
        best_name = best_model_info['Model']
        best_model = models[best_name]
        rmse_val = best_model_info['RMSE']
    else: 
        best_name = "NewModel"
        best_model = models["NewModel"]
        rmse_val = metrics_summary['RMSE']

    # --- TAMPILAN BARU: INFORMASI MODEL CERDAS ---
    st.success(f"""
    🤖 **Smart Model Selection Active** Sistem menggunakan **{best_name}** untuk prediksi masa depan karena memiliki error terendah (RMSE: **{rmse_val:.2f}**) pada tahap pengujian.
    """)

    col_input, col_chart = st.columns([1, 3])
    with col_input:
        # Slider input
        n_days = st.slider("📅 Tentukan Rentang Waktu:", 1, 30, 7)
        st.caption(f"Memprediksi harga penutupan (Close) untuk **{n_days} Hari Kerja** ke depan.")
    
    last_seq = processed_data["last_sequence"]
    scaler_y = processed_data["scaler_y"]
    
    # Generate Prediksi Murni
    future_prices = generate_future_forecast(best_model, last_seq, n_days, scaler_y)
    
    last_date = processed_data["last_date"]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='B')

    # Logika Penyambung Garis (Agar tidak putus)
    last_hist_price = processed_data["y_full_true"][-1]
    plot_y_forecast = np.concatenate(([last_hist_price], future_prices))
    plot_x_forecast = [last_date] + list(future_dates)

    with col_chart:
        fig = go.Figure()
        
        # Plot History (60 Hari Terakhir saja biar fokus)
        fig.add_trace(go.Scatter(
            x=processed_data["dates_full"][-60:], 
            y=processed_data["y_full_true"][-60:], 
            name="History (Last 60 Days)", 
            line=dict(color="#2C3E50", width=3)
        ))
        
        # Plot Forecast
        fig.add_trace(go.Scatter(
            x=plot_x_forecast,
            y=plot_y_forecast,
            name=f"Forecast ({best_name})", 
            mode='lines+markers', 
            line=dict(color="#D32F2F", dash='dot', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"Projection Chart: Next {n_days} Days",
            xaxis_title="Date",
            yaxis_title="Predicted Price (IDR)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan Tabel Angka Prediksi (Opsional, agar lebih detail)
        with st.expander("📋 Lihat Detail Angka Prediksi"):
            df_future = pd.DataFrame({
                "Tanggal": future_dates,
                "Prediksi Harga": future_prices
            })
            st.dataframe(df_future.style.format({"Prediksi Harga": "{:.2f}"}), use_container_width=True)
            
# ====================================================
# 5. MAIN APP LAYOUT (SIDEBAR & NAVIGATION)
# ====================================================
if os.path.exists("logo_sidebar.png"):
    st.sidebar.image("logo_sidebar.png", width="stretch")

st.sidebar.markdown("""
    <div style="text-align: center; margin-top: -15px; margin-bottom: 20px;">
        <h2 style="font-family: 'Helvetica Neue', sans-serif; font-weight: 300; color: #2c3e50; letter-spacing: 4px; font-size: 22px; margin: 0;">NEUROSTOCK</h2>
        <p style="font-size: 11px; color: #7f8c8d; letter-spacing: 1.5px; margin-top: 5px;">INTELLIGENT FORECASTING</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Custom Prediction", "Model Showcase", "About & Info"], 
        icons=["cloud-upload", "graph-up-arrow", "info-circle"], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#2c3e50", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#ecf0f1"
            },
            "nav-link-selected": {
                "background-color": "#2980b9", "font-weight": "600",
            },
        }
    )

if selected == "Custom Prediction":
    menu = "1. General Tool" 
    st.title("📂 Custom Prediction Mode") 
elif selected == "Model Showcase":
    menu = "2. Thesis Dashboard" 
    st.title("📈 Pre-Trained Model Showcase")
elif selected == "About & Info":
    menu = "3. About"

# ====================================================
# 6. PAGE EXECUTION LOGIC
# ====================================================

# --- MENU 1: GENERAL TOOL (CUSTOM PREDICTION) ---
if menu == "1. General Tool":
    st.subheader("🛠️ General")
    
    # Bungkus info dalam kolom agar kotaknya tidak full width (memanjang ke kanan)
    c_info, c_space = st.columns([2, 1]) 
    with c_info:
        st.info("💡 **Panduan:** Unduh data saham otomatis dari Yahoo Finance atau unggah dataset harga historis (CSV/Excel) secara manual.")

    # 1. INISIALISASI SESSION STATE
    if 'custom_data' not in st.session_state:
        st.session_state.custom_data = None
    if 'train_results' not in st.session_state:
        st.session_state.train_results = None

    # 2. INPUT USER (Dibuat Horizontal agar rapi)
    input_method = st.radio(
        "Pilih Metode Input Data:",
        ("📡 Ambil dari Yahoo Finance (Otomatis)", "📂 Upload File Dataset (Manual)"),
        horizontal=True
    )
    
    # --- LOGIKA TOMBOL "TARIK DATA" / "UPLOAD" ---
    
    # OPSI A: YAHOO FINANCE
    if input_method == "📡 Ambil dari Yahoo Finance (Otomatis)":
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            # Daftar Saham Bluechip & Konglomerasi Indonesia Terupdate
            common_tickers = [
                # --- Big Banks & Finance ---
                "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK",
                # --- Konglomerasi & Holding (Astra, Saratoga, Barito, dll) ---
                "ASII.JK", "SRTG.JK", "BREN.JK", "CUAN.JK", "BRPT.JK", "AMMN.JK",
                # --- Telco & Tech ---
                "TLKM.JK", "ISAT.JK", "GOTO.JK", 
                # --- Energi & Tambang ---
                "ADRO.JK", "PTBA.JK", "PGAS.JK", "ANTM.JK", "MEDC.JK",
                # --- Consumer Goods & Retail ---
                "ICBP.JK", "INDF.JK", "AMRT.JK", "MYOR.JK",
                # --- Opsi Manual ---
                "📝 Ketik Kode Lain..."
            ]
            
            selected_ticker = st.selectbox("Pilih Emiten:", options=common_tickers, index=0)
            
            # Logika: Jika user pilih 'Ketik Kode Lain', munculkan input box
            if selected_ticker == "📝 Ketik Kode Lain...":
                ticker_symbol = st.text_input("Masukkan Kode Saham:", value="SIDO.JK", help="Pastikan akhiran .JK untuk saham Indonesia")
            else:
                ticker_symbol = selected_ticker
        
        with c2:
            start_date = st.date_input("Mulai Tanggal:", value=pd.to_datetime("2020-01-01"))
        
        with c3:
            end_date = st.date_input("Sampai Tanggal:", value=pd.to_datetime("today"))
            
        if st.button("📥 Tarik Data"):
            with st.spinner(f"Mengunduh data {ticker_symbol}..."):
                try:
                    df_yf = yf.download(ticker_symbol, start=start_date, end=end_date)
                    if len(df_yf) > 0:
                        df_yf.reset_index(inplace=True)
                        if isinstance(df_yf.columns, pd.MultiIndex):
                            df_yf.columns = df_yf.columns.get_level_values(0)
                        
                        st.session_state.custom_data = df_yf
                        st.session_state.train_results = None 
                        
                        # UPDATE: Gunakan kolom agar alert tidak lebar full layar
                        c_succ, c_empty = st.columns([1, 2]) 
                        c_succ.success(f"Berhasil menarik {len(df_yf)} baris data!")
                        
                    else:
                        st.error("Data tidak ditemukan.")
                except Exception as e:
                    st.error(f"Gagal mengambil data: {e}")

    # OPSI B: UPLOAD MANUAL
    elif input_method == "📂 Upload File Dataset (Manual)":
        
        # Bungkus kotak upload dalam kolom rasio 1:1 agar tidak lebar memanjang ke kanan
        c_upload, c_empty_upload = st.columns([1, 1])
        
        with c_upload:
            # Teks diperpendek agar lebih rapi
            file = st.file_uploader("Upload Data Historis (CSV/XLSX):", type=["csv", "xlsx"])
            
            if file:
                with st.spinner("Membaca file..."):
                    try:
                        if file.name.endswith('.csv'): 
                            new_data = pd.read_csv(file)
                        else: 
                            new_data = pd.read_excel(file)
                        
                        # Cek apakah data berubah, jika ya reset training
                        st.session_state.custom_data = new_data
                        st.session_state.train_results = None 
                        
                        st.success(f"File '{file.name}' berhasil dimuat!")
                    except Exception as e:
                        st.error(f"Gagal membaca file: {e}")

    # 3. TAMPILKAN PREVIEW & KONFIGURASI SIMPLE
    if st.session_state.custom_data is not None:
        df_display = st.session_state.custom_data
        cols = df_display.columns.tolist()
        
        st.markdown("---")
        
        # Layout: Kiri (Preview Data), Kanan (Tombol Aksi)
        c_left, c_right = st.columns([3, 1])
        
        with c_left:
            st.write("### 📋 Preview Dataset")
            st.dataframe(df_display.tail(3), use_container_width=True)
            
        with c_right:
            st.write("### ⚙️ Aksi")
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.custom_data = None
                st.session_state.train_results = None
                st.rerun()

        # --- LOGIKA AUTO-MAPPING PINTAR ---
        def find_idx(keywords):
            for i, c in enumerate(cols):
                if any(k in c.lower() for k in keywords): return i
            return 0

        # --- UI MAPPING (TERSEMBUNYI / EXPANDER) ---
        with st.expander("🛠️ Cek & Edit Kolom (Deteksi Otomatis)", expanded=False):
            st.caption("Sistem telah memilih kolom secara otomatis. Ubah hanya jika salah.")
            
            c1, c2, c3 = st.columns(3)
            # Baris 1 (Emoticon dihapus)
            col_date = c1.selectbox("Tanggal", cols, index=find_idx(['date', 'tgl', 'waktu']))
            col_open = c2.selectbox("Open", cols, index=find_idx(['open', 'buka']))
            col_high = c3.selectbox("High", cols, index=find_idx(['high', 'tinggi']))
            
            # Baris 2 (Emoticon & text Target dihapus)
            c4, c5, c6 = st.columns(3)
            col_low  = c4.selectbox("Low", cols, index=find_idx(['low', 'rendah']))
            col_close= c5.selectbox("Close", cols, index=find_idx(['close', 'tutup', 'adj', 'price']))
            col_vol  = c6.selectbox("Volume", cols, index=find_idx(['vol', 'jum']))

        # Bungkus mapping untuk dikirim ke fungsi training
        col_map = {
            'Date': col_date, 'Open': col_open, 'High': col_high,
            'Low': col_low, 'Close': col_close, 'Volume': col_vol
        }

        # --- TOMBOL EKSEKUSI ---
        st.write("") # Spasi
        
        if st.button("🚀 Mulai Training & Prediksi", type="primary"):
            
            # Cek duplikasi kolom
            selected_features = [col_open, col_high, col_low, col_close, col_vol]
            if len(set(selected_features)) < 5:
                st.warning("⚠️ Perhatian: Ada kolom yang dipilih ganda. Pastikan 5 indikator berbeda.")
            
            # Jalankan Training
            res = process_upload_and_train(df_display, col_map)
            
            if res:
                st.session_state.train_results = res

        # 5. TAMPILKAN HASIL (KOMPETISI MODEL)
        if st.session_state.train_results is not None:
            res = st.session_state.train_results 
            
            st.markdown("---")
            st.write("### 🏆 Hasil Kompetisi Model")
            
            # A. Tabel Perbandingan (Dengan Layout Kolom agar tidak melebar)
            metrics_list = res['metrics']
            df_metrics = pd.DataFrame(metrics_list).set_index('Model')
            
            # Hapus kolom Accuracy
            if 'Accuracy' in df_metrics.columns:
                df_metrics = df_metrics.drop(columns=['Accuracy'])
            
            # Cari Juara
            best_model_name = df_metrics['RMSE'].idxmin()
            best_rmse = df_metrics.loc[best_model_name, 'RMSE']
            
            # --- LAYOUT TABEL (Gunakan kolom rasio 2:1 agar tidak full width) ---
            c_table, c_empty = st.columns([2, 1]) 
            with c_table:
                # 1. Tampilkan Tabel
                st.dataframe(
                    df_metrics.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='#d1e7dd').format("{:.2f}"),
                    use_container_width=True
                )
                
                # 2. Tampilkan Notifikasi Juara DI DALAM kolom yang sama agar lebarnya pas
                st.success(f"🥇 **Model Terbaik:** {best_model_name} (RMSE: {best_rmse:.2f})")
            
            # B. Plot Perbandingan
            st.write("#### 📈 Grafik Perbandingan")
            
            # --- LAYOUT DROPDOWN (Gunakan kolom rasio 1:2 agar kecil) ---
            available_models = list(res['preds'].keys())
            plot_options = [f"Best Model ({best_model_name})", "All Models"] + available_models
            
            c_select, c_space = st.columns([1, 2]) # 1 Bagian isi, 2 Bagian kosong
            with c_select:
                model_selection = st.selectbox("Pilih Model untuk Ditampilkan:", plot_options)
            
            # LOGIKA FILTER GRAFIK
            if "All Models" in model_selection:
                preds_to_plot = res['preds']
                title_sf = "(All Models)"
            elif "Best Model" in model_selection:
                preds_to_plot = {best_model_name: res['preds'][best_model_name]}
                title_sf = f"({best_model_name})"
            else:
                preds_to_plot = {model_selection: res['preds'][model_selection]}
                title_sf = f"({model_selection})"

            # Render Plot
            render_comparison_plots(res['processed_data'], preds_to_plot, title_suffix=title_sf)
            
            # C. RSI
            if "rsi_values" in res['processed_data']:
                st.write("#### 📉 Indikator RSI")
                rsi_data = res['processed_data']['rsi_values']
                dates_rsi = res['processed_data']['dates_full']
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=dates_rsi, y=rsi_data, name="RSI", line=dict(color='purple', width=1.5)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(height=300, margin=dict(t=10, b=10, l=20, r=20), yaxis_title="RSI", xaxis_title="Date", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig_rsi, use_container_width=True)

            # D. FORECASTING
            render_forecast_ui(res['processed_data'], res['metrics'], res['model'])
            
# ====================================================
# MENU 2: THESIS DASHBOARD (MODEL SHOWCASE)
# ====================================================
elif menu == "2. Thesis Dashboard":
    st.subheader("📌 Case Study: ASII.JK & SRTG.JK")
    
    # --- UPDATE STYLE DROPDOWN (Agar tidak terlalu panjang) ---
    # Gunakan 3 kolom: 1 bagian Ticker, 1 bagian Skenario, 2 bagian kosong
    c1, c2, c_empty = st.columns([1, 1, 2]) 
    
    with c1: 
        ticker = st.selectbox("Pilih Emiten:", ["ASII", "SRTG"])
    with c2: 
        scenario = st.selectbox("Pilih Skenario:", ["Skenario 1 (10 Tahun)", "Skenario 2 (5 Tahun)"])
    
    # Tombol Load (juga ditaruh di kolom kiri agar sejajar)
    with c1:
        load_btn = st.button("🔍 Load Model & Analisis")

    if load_btn:
        with st.spinner(f"Memuat Aset {ticker} - {scenario}..."):
            # ... (kode ke bawah tetap sama seperti sebelumnya) ...
            res = load_thesis_resources(ticker, scenario)
            
            if res:
                data_proc, models_loaded = res
                
                # Persiapan Data untuk EDA
                tick_tag = ticker.lower()
                try:
                    df_eda_full = pd.read_csv(f"data/{tick_tag}_cleaned_data_master.csv")
                    df_eda_full['Date'] = pd.to_datetime(df_eda_full['Date'])
                    
                    # Filter Tanggal
                    start_year = 2016 if "10" in scenario else 2021
                    df_eda_filtered = df_eda_full[df_eda_full['Date'].dt.year >= start_year].reset_index(drop=True)

                    st.success("Data Berhasil Dimuat!")
                    
                    # --- TABS ---
                    tab_eda, tab_pred = st.tabs(["📊 EDA & Preprocessing", "🤖 Prediksi & Evaluasi"])
                    
                    # TAB 1: EDA
                    with tab_eda:
                        render_eda_dashboard(df_eda_filtered, ticker)
                    
                    # TAB 2: PREDIKSI
                    with tab_pred:
                        preds, metrics = run_predictions_menu1(models_loaded, data_proc)
                        
                        st.write("### 📊 Evaluasi Model (Test Set)")
                        
                        # Olah Metrics
                        df_m = pd.DataFrame(metrics).set_index('Model').sort_values('RMSE')
                        if 'Accuracy' in df_m.columns:
                            df_m = df_m.drop(columns=['Accuracy'])
                        
                        # Layout Tabel (Kecil di kiri)
                        c_tbl, c_space = st.columns([2, 1])
                        with c_tbl:
                            st.dataframe(
                                df_m.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='#d1e7dd').format("{:.2f}"),
                                use_container_width=True
                            )
                        
                        # Plot Perbandingan
                        render_comparison_plots(data_proc, preds, f"- {ticker} ({scenario})")
                        
                        # Plot Forecasting
                        render_forecast_ui(data_proc, metrics, models_loaded)

                except Exception as e:
                    st.error(f"Gagal memuat data atau model: {e}")

# --- MENU 3: ABOUT ---
elif menu == "3. About":
    st.title("ℹ️ Tentang NeuroStock")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("### 🎓 Project Skripsi")
        st.info("""
        **NeuroStock** adalah sistem prediksi harga saham berbasis *Deep Learning* yang dikembangkan sebagai bagian dari Tugas Akhir/Skripsi.
        
        Sistem ini menerapkan pendekatan komparatif dengan mengevaluasi kinerja tiga arsitektur Deep Learning—yaitu CNN, LSTM, dan Hybrid CNN-LSTM—untuk menangkap pola spasial dan temporal dari data historis pasar saham, guna menentukan model terbaik dengan akurasi prediksi paling optimal.
        """)
        st.write("### 🛠️ Teknologi")
        st.markdown("""
        * **Framework:** Python 3.10+
        * **Engine:** TensorFlow & Keras
        * **Interface:** Streamlit
        * **Data Processing:** Pandas, NumPy, Scikit-learn
        * **Visualization:** Plotly Interactive
        """)
    with c2:
        st.write("### ⚠️ Disclaimer")
        st.warning("""
        **Harap Diperhatikan:**
        1. **Tujuan Akademis:** Aplikasi ini dibuat murni untuk tujuan penelitian akademis.
        2. **Bukan Saran Finansial:** Hasil prediksi **TIDAK BOLEH** dijadikan acuan mutlak untuk investasi.
        3. **Risiko:** Pasar saham memiliki risiko tinggi.
        
        *do your own research (DYOR).*
        """)


st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 13px; color: #2c3e50; margin-top: 30px; line-height: 1.6;">
        &copy; 2026 <b style="color: #2980b9; font-size: 14px;">NeuroStock</b><br>
        All Rights Reserved.<br>
        <span style="font-size: 11px; color: #7f8c8d; font-weight: 500;">Ver 1.0.0 (Professional Edition)</span>
    </div>
    """,
    unsafe_allow_html=True
)