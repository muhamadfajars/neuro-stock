import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import io
import os 

# ====================================================
# 1. CONFIG & STYLE
# ====================================================
st.set_page_config(
    page_title="Stock Model Comparison",
    layout="wide",
    page_icon="favicon.png"  # <--- Ganti dengan nama file gambar Anda
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
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #1565c0;
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
# 2. CONSTANTS (FILE PATHS)
# ====================================================
DATA_FILE_PATH_ASII = "DATASET_ASII_10_Tahun.xlsx"
CNN_FILE_PATH = "best_model_cnn.h5"
LSTM_FILE_PATH = "best_model_lstm.h5"
HYBRID_FILE_PATH = "best_model_hybrid.h5"

# ====================================================
# 3. HELPER FUNCTIONS (MATH & LOGIC)
# ====================================================
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def evaluate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat) * 100
    accuracy = 100.0 - mape
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Accuracy': accuracy}

@st.cache_resource
def load_all_models():
    """Load model sekali saja untuk efisiensi."""
    try:
        model_cnn = load_model(CNN_FILE_PATH, compile=False)
        model_lstm = load_model(LSTM_FILE_PATH, compile=False)
        model_hybrid = load_model(HYBRID_FILE_PATH, compile=False)
        return {"CNN": model_cnn, "LSTM": model_lstm, "Hybrid": model_hybrid}
    except Exception as e:
        return None

# ====================================================
# 4. FORECASTING LOGIC (FUTURE PREDICTION)
# ====================================================
def generate_future_forecast(model, last_sequence, n_days, scaler):
    """
    Melakukan prediksi rekursif ke masa depan.
    Input: last_sequence (Scaled, Shape: 1, 60, 1)
    Output: List harga (Inverse Scaled)
    """
    forecast_scaled = []
    current_step = last_sequence.copy() # Shape: (1, 60, 1)
    
    for _ in range(n_days):
        # 1. Prediksi langkah berikutnya
        pred = model.predict(current_step, verbose=0) # Shape (1, 1)
        forecast_scaled.append(pred[0, 0])
        
        # 2. Update sequence: Hapus data terlama (index 0), Tambah data baru (pred) di akhir
        # Reshape pred agar bisa digabung: (1, 1, 1)
        pred_reshaped = pred.reshape(1, 1, 1)
        current_step = np.append(current_step[:, 1:, :], pred_reshaped, axis=1)

    # 3. Kembalikan ke harga Rupiah
    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast_final = scaler.inverse_transform(forecast_scaled)
    
    return forecast_final.flatten()

# ====================================================
# 5. DATA PROCESSING FUNCTIONS
# ====================================================
@st.cache_data
def process_data_asii(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df[['Date', 'Adj Close']]
    except Exception as e:
        st.error(f"Error loading ASII file: {e}")
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= '2015-01-01') & (df['Date'] <= '2025-09-30')]
    df = df.sort_values('Date').reset_index(drop=True)
    df['Adj Close'] = df['Adj Close'].ffill().bfill() 

    return run_preprocessing_pipeline(df)

def process_data_upload(df_raw, col_date, col_price):
    try:
        df = df_raw.rename(columns={col_date: 'Date', col_price: 'Adj Close'})
        df = df[['Date', 'Adj Close']]
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
        df = df.dropna(subset=['Adj Close'])
        
        if len(df) < 100:
            st.warning("Jumlah data kurang dari 100 baris. Hasil mungkin tidak optimal.")
            
        return run_preprocessing_pipeline(df)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data upload: {e}")
        return None

def run_preprocessing_pipeline(df):
    data_series = df['Adj Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series)
    
    split_idx = int(len(scaled_data) * 0.8)
    time_step = 60
    
    if len(scaled_data) < time_step + 10:
        st.error(f"Data terlalu sedikit! Butuh minimal {time_step+10} baris.")
        return None

    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx:]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    dates_full = df['Date']
    dates_train = dates_full[time_step:split_idx]
    dates_test = dates_full[split_idx+time_step:]
    split_date = dates_full.iloc[split_idx]
    
    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_full_true = df['Adj Close'].values
    
    # Simpan data mentah terakhir untuk forecasting masa depan
    last_60_days_scaled = scaled_data[-time_step:].reshape(1, time_step, 1)
    last_date = dates_full.iloc[-1]
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "split_idx": split_idx,
        "dates_train": dates_train, "dates_test": dates_test, 
        "dates_full": dates_full, "split_date": split_date,
        "y_train_true": y_train_true, "y_test_true": y_test_true, 
        "y_full_true": y_full_true,
        "last_sequence": last_60_days_scaled, # Untuk forecasting
        "last_date": last_date # Untuk tanggal forecasting
    }

# ====================================================
# 6. PLOTTING & UI RENDERERS
# ====================================================
def run_predictions(models_dict, processed_data):
    scaler = processed_data["scaler"]
    predictions_inv = {}
    metrics_summary = []
    
    for name, model in models_dict.items():
        pred_train_scaled = model.predict(processed_data["X_train"], verbose=0)
        pred_test_scaled = model.predict(processed_data["X_test"], verbose=0)
        
        pred_train_inv = scaler.inverse_transform(pred_train_scaled)
        pred_test_inv = scaler.inverse_transform(pred_test_scaled)
        
        predictions_inv[name] = {"train": pred_train_inv, "test": pred_test_inv}
        
        metrics = evaluate_metrics(processed_data["y_test_true"], pred_test_inv)
        metrics['Model'] = name
        metrics_summary.append(metrics)
        
    return predictions_inv, metrics_summary

def render_comparison_plots(processed_data, predictions_inv, title_suffix=""):
    def create_fig(plot_type, selected_models):
        fig = go.Figure()
        colors = {"Actual": "#2C3E50", "CNN": "#FF8C00", "LSTM": "#00C853", "Hybrid": "#6200EA"}
        
        model_list = ["CNN", "LSTM", "Hybrid"] if selected_models == "All Models" else [selected_models.replace(" Only", "").replace("Hybrid CNN-LSTM", "Hybrid")]
        
        if plot_type == "Combined":
            fig.add_trace(go.Scatter(x=processed_data["dates_full"], y=processed_data["y_full_true"], name="Actual Price", line=dict(color=colors["Actual"], width=2.5), opacity=0.5))
            for m in model_list:
                fig.add_trace(go.Scatter(x=processed_data["dates_test"], y=predictions_inv[m]["test"].flatten(), name=f"{m} (Test)", line=dict(color=colors[m], width=2)))
            split_date = processed_data["split_date"].isoformat() if hasattr(processed_data["split_date"], 'isoformat') else processed_data["split_date"]
            fig.add_vline(x=split_date, line_dash="dash", line_color="red", name="Split Point")
        
        elif plot_type == "Train":
             fig.add_trace(go.Scatter(x=processed_data["dates_train"], y=processed_data["y_train_true"].flatten(), name="Actual", line=dict(color=colors["Actual"], width=2)))
             for m in model_list:
                fig.add_trace(go.Scatter(x=processed_data["dates_train"], y=predictions_inv[m]["train"].flatten(), name=f"{m}", line=dict(color=colors[m], width=1.5)))

        elif plot_type == "Test":
             fig.add_trace(go.Scatter(x=processed_data["dates_test"], y=processed_data["y_test_true"].flatten(), name="Actual", line=dict(color=colors["Actual"], width=2)))
             for m in model_list:
                fig.add_trace(go.Scatter(x=processed_data["dates_test"], y=predictions_inv[m]["test"].flatten(), name=f"{m}", line=dict(color=colors[m], width=2)))

        fig.update_layout(title=f"{plot_type} Comparison {title_suffix}", xaxis_title="Date", yaxis_title="Price (IDR)", hovermode="x unified", xaxis_hoverformat="%d %b %Y", margin=dict(l=20, r=20, t=50, b=20))
        return fig

    tab_train, tab_test, tab_comb = st.tabs(["Train Data", "Test Data", "Combined View"])
    with tab_train: st.plotly_chart(create_fig("Train", st.selectbox("Model (Train):", ["All Models", "CNN Only", "LSTM Only", "Hybrid CNN-LSTM Only"], key="s1")), use_container_width=True)
    with tab_test: st.plotly_chart(create_fig("Test", st.selectbox("Model (Test):", ["All Models", "CNN Only", "LSTM Only", "Hybrid CNN-LSTM Only"], key="s2")), use_container_width=True)
    with tab_comb: st.plotly_chart(create_fig("Combined", st.selectbox("Model (Combined):", ["All Models", "CNN Only", "LSTM Only", "Hybrid CNN-LSTM Only"], key="s3")), use_container_width=True)

# --- FUNGSI UI FORECASTING BARU ---
def render_forecast_ui(processed_data, metrics_summary, models):
    st.markdown("---")
    st.header("🔮 Forecasting")
    st.info("Fitur ini memprediksi harga saham untuk N-hari ke depan setelah tanggal terakhir data.")

    # 1. Cari Model Terbaik (Lowest RMSE)
    best_model_info = min(metrics_summary, key=lambda x: x['RMSE'])
    best_name = best_model_info['Model']
    st.markdown(f"""
    > **🤖 Smart Selection:** > Berdasarkan evaluasi Test Set, model terbaik adalah **{best_name}** dengan RMSE **{best_model_info['RMSE']:.2f}**.
    > Sistem akan menggunakan model ini untuk prediksi masa depan.
    """)

    # 2. Slider Controls
    col_input, col_chart = st.columns([1, 3])
    
    with col_input:
        n_days = st.slider("Jumlah Hari Prediksi:", min_value=1, max_value=30, value=7)
        st.write(f"Memprediksi **{n_days} hari kerja** ke depan.")
    
    # 3. Generate Forecast
    best_model = models[best_name]
    last_seq = processed_data["last_sequence"]
    scaler = processed_data["scaler"]
    
    future_prices = generate_future_forecast(best_model, last_seq, n_days, scaler)
    
    # 4. Generate Dates (Business Days - Skip Weekend)
    last_date = processed_data["last_date"]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='B')

    # 5. Visualisasi Forecasting
    with col_chart:
        fig_fore = go.Figure()
        
        # Data Historis (Zoom in 60 hari terakhir agar terlihat jelas)
        hist_days = 60
        fig_fore.add_trace(go.Scatter(
            x=processed_data["dates_full"][-hist_days:], 
            y=processed_data["y_full_true"][-hist_days:],
            name="Data Aktual (Terakhir)",
            line=dict(color="#2C3E50", width=2)
        ))
        
        # Garis Sambung (Dari titik terakhir data ke titik pertama prediksi)
        connect_x = [processed_data["dates_full"].iloc[-1], future_dates[0]]
        connect_y = [processed_data["y_full_true"][-1], future_prices[0]]
        fig_fore.add_trace(go.Scatter(
            x=connect_x, y=connect_y, showlegend=False,
            line=dict(color="#D32F2F", width=2, dash="dot")
        ))
        
        # Data Prediksi
        fig_fore.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            name=f"Prediksi {best_name}",
            mode='lines+markers',
            line=dict(color="#D32F2F", width=2, dash="dot"),
            marker=dict(size=6)
        ))
        
        fig_fore.update_layout(
            title=f"Forecast {n_days} Hari ke Depan ({best_name})",
            xaxis_title="Date", yaxis_title="Price (IDR)",
            hovermode="x unified", xaxis_hoverformat="%d %b %Y",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_fore, use_container_width=True)

    # 6. Tabel Data Prediksi
    with st.expander("📄 Lihat Tabel Detail Harga Prediksi"):
        df_future = pd.DataFrame({
            "Tanggal": future_dates,
            "Prediksi Harga (IDR)": future_prices
        })
        # Format tanggal agar rapi
        df_future['Tanggal'] = df_future['Tanggal'].dt.strftime('%d %B %Y')
        st.dataframe(df_future.style.format({"Prediksi Harga (IDR)": "{:,.2f}"}))


# ====================================================
# 7. MAIN APPLICATION LAYOUT
# ====================================================

st.title("📈 Deep Learning Stock Prediction")
# Pastikan file 'logo_sidebar.png' ada di folder yang sama
if os.path.exists("logo_sidebar.png"):
    # use_container_width=True membuat logo menyesuaikan lebar sidebar otomatis
    st.sidebar.image("logo_sidebar.png", use_container_width=True) 
    
    # OPSI: Jika ingin logo lebih kecil, gunakan parameter width (hapus use_container_width)
    # st.sidebar.image("logo_sidebar.png", width=150) 
else:
    # Fallback jika gambar belum diupload/salah nama (opsional)
    st.sidebar.write(" ")

st.sidebar.header("Navigasi Utama")

menu = st.sidebar.radio(
    "Pilih Dashboard:",
    ["1. Dashboard Skripsi (ASII)", "2. Uji Model (Upload Data Lain)"]
)

models = load_all_models()
if not models:
    st.error("Gagal memuat file Model (.h5). Pastikan file ada di folder yang sama.")
    st.stop()

# --- MENU 1: ASII ---
if menu == "1. Dashboard Skripsi (ASII)":
    st.subheader("📌 Studi Kasus: Astra International (ASII)")
    
    if os.path.exists(DATA_FILE_PATH_ASII):
        processed_data = process_data_asii(DATA_FILE_PATH_ASII)
        if processed_data:
            with st.spinner("Menjalankan Prediksi pada Data ASII..."):
                preds, metrics = run_predictions(models, processed_data)
            
            st.write("### 📊 Model Performance (Test Set)")
            df_m = pd.DataFrame(metrics).sort_values(by="RMSE")
            st.dataframe(df_m.style.format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MAPE': '{:.2f}%', 'Accuracy': '{:.2f}%'}))
            
            st.write("### 📈 Prediction Visualization")
            render_comparison_plots(processed_data, preds, title_suffix="- ASII")
            
            # --- Panggil UI Forecasting ---
            render_forecast_ui(processed_data, metrics, models)
    else:
        st.error(f"File Dataset '{DATA_FILE_PATH_ASII}' tidak ditemukan.")

# ====================================================
# MENU 2: UJI COBA DATA LAIN (UPLOAD & MAPPING) - PERBAIKAN SESSION STATE
# ====================================================
elif menu == "2. Uji Model (Upload Data Lain)":
    st.subheader("📂 Uji Generalisasi Model (Emiten Lain)")
    st.markdown("Instruksi: Upload CSV/Excel -> Mapping Kolom -> Proses.")
    
    # Reset state jika file berubah (opsional, agar bersih)
    if "last_uploaded_file" not in st.session_state:
        st.session_state["last_uploaded_file"] = None

    uploaded_file = st.file_uploader("Upload File (.xlsx / .csv)", type=['xlsx', 'csv'])
    
    # Cek jika user ganti file, kita reset hasil proses sebelumnya
    if uploaded_file and uploaded_file != st.session_state["last_uploaded_file"]:
        st.session_state["last_uploaded_file"] = uploaded_file
        if "upload_results" in st.session_state:
            del st.session_state["upload_results"]

    if uploaded_file:
        try:
            # 1. BACA FILE
            if uploaded_file.name.endswith('.csv'): 
                # Reset pointer file ke awal agar bisa dibaca ulang jika perlu
                uploaded_file.seek(0) 
                df_raw = pd.read_csv(uploaded_file)
            else: 
                df_raw = pd.read_excel(uploaded_file)
            
            st.dataframe(df_raw.head())
            st.markdown("---")
            
            # 2. MAPPING KOLOM
            c1, c2 = st.columns(2)
            with c1: 
                col_date = st.selectbox("Pilih Kolom TANGGAL:", df_raw.columns)
            with c2: 
                # Cari default index cerdas
                def_idx = next((i for i, c in enumerate(df_raw.columns) if 'close' in c.lower() or 'harga' in c.lower()), 0)
                col_price = st.selectbox("Pilih Kolom HARGA:", df_raw.columns, index=def_idx)
            
            # 3. TOMBOL PROSES (Hanya trigger simpan data ke Session State)
            if st.button("🚀 Proses & Prediksi"):
                with st.spinner("Memproses data & Menjalankan Model..."):
                    processed_data_custom = process_data_upload(df_raw, col_date, col_price)
                    
                    if processed_data_custom:
                        preds_custom, metrics_custom = run_predictions(models, processed_data_custom)
                        
                        # --- INI KUNCINYA: SIMPAN KE SESSION STATE ---
                        st.session_state['upload_results'] = {
                            "data": processed_data_custom,
                            "preds": preds_custom,
                            "metrics": metrics_custom
                        }
                        st.success("Berhasil! Silakan scroll ke bawah.")

            # 4. TAMPILKAN HASIL (Cek apakah data ada di Session State)
            if 'upload_results' in st.session_state:
                # Ambil data dari memori
                res = st.session_state['upload_results']
                
                st.write("### 📊 Evaluation Metrics")
                df_mc = pd.DataFrame(res['metrics']).sort_values(by="RMSE")
                st.dataframe(df_mc.style.format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MAPE': '{:.2f}%', 'Accuracy': '{:.2f}%'}))
                
                st.write("### 📈 Visualisasi")
                render_comparison_plots(res['data'], res['preds'], title_suffix="- Upload")
                
                # --- Panggil UI Forecasting ---
                # Sekarang Slider aman karena blok ini berada DI LUAR if st.button()
                render_forecast_ui(res['data'], res['metrics'], models)

        except Exception as e:
            st.error(f"Error membaca file: {e}")