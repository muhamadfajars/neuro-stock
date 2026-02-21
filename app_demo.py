import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import plotly.graph_objects as go
import io
import os 

def add_custom_css():
    st.markdown(
        """
        <style>
        /* 1. Gradasi untuk Sidebar (Vertikal: Atas ke Bawah) */
        /* Warna: Dari Abu-abu sangat muda (#f8f9fa) ke Biru Langit pudar (#e3f2fd) */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #f8f9fa 0%, #e3f2fd 100%);
            border-right: 1px solid #d0d7de; /* Garis pemisah tipis */
        }

        /* 2. Gradasi untuk Navbar/Header (Horizontal: Kiri ke Kanan) */
        /* Warna: Dari Putih (#ffffff) ke Biru Tipis (#e1f5fe) */
        /* Kita buat transparan sedikit agar tidak menabrak konten */
        header[data-testid="stHeader"] {
            background-image: linear-gradient(90deg, #ffffff 0%, #e1f5fe 100%);
            border-bottom: 1px solid #d0d7de; /* Garis bawah tipis */
        }
        
        /* Opsi Tambahan: Mengubah warna teks judul di Sidebar agar lebih kontras */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #1565c0; /* Biru Tua */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ====================================================
# PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="Stock Model Comparison",
    layout="wide"
)

add_custom_css()

# ====================================================
# FILENAMES (Hardcoded)
# ====================================================
DATA_FILE_PATH = "DATASET_ASII_10_Tahun.xlsx"
CNN_FILE_PATH = "best_model_cnn.h5"
LSTM_FILE_PATH = "best_model_lstm.h5"
HYBRID_FILE_PATH = "best_model_hybrid.h5"

# ====================================================
# HELPER FUNCTIONS
# ====================================================
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def evaluate_metrics(y_true, y_pred):
    """Calculates MSE, RMSE, MAE, MAPE, and Accuracy (100-MAPE)."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat) * 100
    accuracy_from_mape = 100.0 - mape
        
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Accuracy': accuracy_from_mape}

# ====================================================
# MAIN CACHED FUNCTIONS
# ====================================================

@st.cache_data
def load_and_process_data(data_file_path):
    """Loads data FROM FILE PATH."""
    try:
        df = pd.read_excel(data_file_path) 
        df = df[['Date', 'Adj Close']]
    except FileNotFoundError:
        st.error(f"Data load failed: File '{data_file_path}' not found.")
        st.info("Please ensure the data file is in the same folder as app_demo.py")
        return None
    except Exception as e:
        st.error(f"Error reading data file: {e}")
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= '2015-01-01') & (df['Date'] <= '2025-09-30')]
    df = df.sort_values('Date').reset_index(drop=True)
    df['Adj Close'] = df['Adj Close'].ffill()
    data_series = df['Adj Close'].values.reshape(-1, 1)

    time_step = 60
    split_idx = int(len(data_series) * 0.8)
    train_data = data_series[:split_idx]
    test_data = data_series[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    X_train, y_train = create_dataset(scaled_train_data, time_step)
    X_test, y_test = create_dataset(scaled_test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    dates_train = df['Date'][time_step:split_idx]
    dates_test = df['Date'][split_idx + time_step:]
    dates_full = df['Date']
    split_date = df['Date'][split_idx]
    
    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_full_true = df['Adj Close'].values

    processed_data = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "time_step": time_step, "split_idx": split_idx,
        "dates_train": dates_train, "dates_test": dates_test, 
        "dates_full": dates_full, "split_date": split_date,
        "y_train_true": y_train_true, "y_test_true": y_test_true, 
        "y_full_true": y_full_true
    }
    return processed_data

@st.cache_data
def load_models_and_predict(_processed_data, cnn_path, lstm_path, hybrid_path):
    """Loads 3 models FROM FILE PATHS and runs predictions."""
    try:
        model_cnn = load_model(cnn_path, compile=False)
        model_lstm = load_model(lstm_path, compile=False)
        model_hybrid = load_model(hybrid_path, compile=False)
    except FileNotFoundError:
        st.error("Failed to load .h5 model files. Ensure all model files are in the same folder.")
        return None, None
    except Exception as e:
        st.error(f"Failed to load .h5 models: {e}")
        return None, None

    models = {"CNN": model_cnn, "LSTM": model_lstm, "Hybrid": model_hybrid}
    scaler = _processed_data["scaler"]
    X_train = _processed_data["X_train"]
    X_test = _processed_data["X_test"]
    y_test_true = _processed_data["y_test_true"]
    predictions_inv = {}
    metrics_summary = []

    for name, model in models.items():
        pred_train_scaled = model.predict(X_train)
        pred_test_scaled = model.predict(X_test) 

        pred_train_inv = scaler.inverse_transform(pred_train_scaled)
        pred_test_inv = scaler.inverse_transform(pred_test_scaled)
        predictions_inv[name] = {"train": pred_train_inv, "test": pred_test_inv}
        
        metrics = evaluate_metrics(y_test_true, pred_test_inv)
        metrics['Model'] = name
        metrics_summary.append(metrics)

    return predictions_inv, metrics_summary

def create_comparison_plot(plot_type, display_option, data, preds, title):
    """Creates Plotly comparison plot with CUSTOM DATE FORMAT."""
    fig = go.Figure()

    # --- KONFIGURASI WARNA ---
    colors = {
        "Actual": "#2C3E50",  
        "CNN": "#FF8C00",     
        "LSTM": "#00C853",    
        "Hybrid": "#6200EA"   
    }

    # --- LOGIKA MEMILIH MODEL ---
    models_to_plot = []
    if display_option == "All Models": models_to_plot = ["CNN", "LSTM", "Hybrid"]
    elif display_option == "CNN Only": models_to_plot = ["CNN"]
    elif display_option == "LSTM Only": models_to_plot = ["LSTM"]
    elif display_option == "Hybrid CNN-LSTM Only": models_to_plot = ["Hybrid"]

    # -------------------------------------------------------
    # 1. PLOT TRAIN COMPARISON
    # -------------------------------------------------------
    if plot_type == "Train Comparison":
        dates_actual = data["dates_train"]
        y_actual = data["y_train_true"].flatten()
        
        # Garis Asli
        fig.add_trace(go.Scatter(
            x=dates_actual, y=y_actual, name="Actual (Train)", 
            line=dict(color=colors["Actual"], width=2.5), opacity=0.6
        ))
        
        for model_name in models_to_plot:
            fig.add_trace(go.Scatter(
                x=dates_actual, y=preds[model_name]["train"].flatten(), 
                name=f"{model_name} (Train)", mode='lines', 
                line=dict(color=colors[model_name], width=2) 
            ))

    # -------------------------------------------------------
    # 2. PLOT TEST COMPARISON
    # -------------------------------------------------------
    elif plot_type == "Test Comparison":
        dates_actual = data["dates_test"]
        y_actual = data["y_test_true"].flatten()
        
        # Garis Asli
        fig.add_trace(go.Scatter(
            x=dates_actual, y=y_actual, name="Actual (Test)", 
            line=dict(color=colors["Actual"], width=2.5), opacity=0.6
        ))
        
        for model_name in models_to_plot:
            fig.add_trace(go.Scatter(
                x=dates_actual, y=preds[model_name]["test"].flatten(), 
                name=f"{model_name} (Test)", mode='lines', 
                line=dict(color=colors[model_name], width=2)
            ))
            
    # -------------------------------------------------------
    # 3. PLOT COMBINED
    # -------------------------------------------------------
    elif plot_type == "Combined (Train+Test)":
        # Garis Asli Full
        fig.add_trace(go.Scatter(
            x=data["dates_full"], y=data["y_full_true"], name="Actual Price", 
            line=dict(color=colors["Actual"], width=2.5), opacity=0.5
        ))
        
        for model_name in models_to_plot:
            # Segmen Train
            fig.add_trace(go.Scatter(
                x=data["dates_train"], y=preds[model_name]["train"].flatten(), 
                name=f"{model_name} (Train)", mode='lines', showlegend=False,
                line=dict(color=colors[model_name], width=2)
            ))
            # Segmen Test
            fig.add_trace(go.Scatter(
                x=data["dates_test"], y=preds[model_name]["test"].flatten(), 
                name=f"{model_name}", mode='lines', 
                line=dict(color=colors[model_name], width=2)
            ))
            
        split_date_str = data["split_date"].isoformat()
        
        fig.add_vline(
            x=split_date_str, line_width=2, line_dash="dash", line_color="red", name="Split Point"
        )
        fig.add_annotation(
            x=split_date_str, y=data["y_full_true"].max(), text="Split 80:20",
            showarrow=False, yshift=10, bgcolor="rgba(255, 255, 255, 0.8)", font=dict(color="red")
        )

    # ====================================================
    # UPDATE LAYOUT (DENGAN FORMAT TANGGAL BARU)
    # ====================================================
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)), 
        xaxis_title="Date", 
        yaxis_title="Price (IDR)",
        hovermode="x unified",
        
        # --- PERUBAHAN DI SINI ---
        # %d = Tanggal (02)
        # %b = Nama Bulan Singkatan (Jan, Feb, Mar) -> Ini kuncinya
        # %Y = Tahun (2015)
        xaxis_hoverformat="%d %b %Y", 
        # -------------------------
        
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

    return fig

# ====================================================
# STREAMLIT APP LAYOUT (ENGLISH)
# ====================================================

st.title("🚀 Stock Price Model Comparison")
st.markdown("Comparing the performance of CNN, LSTM, and Hybrid CNN-LSTM models.")

# --- SIDEBAR (NO UPLOADER) ---
st.sidebar.header("Files in Use")
st.sidebar.info(
    f"""
    This demo script automatically loads files from its folder:
    - **Data:** `{DATA_FILE_PATH}`
    - **CNN:** `{CNN_FILE_PATH}`
    - **LSTM:** `{LSTM_FILE_PATH}`
    - **Hybrid:** `{HYBRID_FILE_PATH}`
    """
)

st.sidebar.header("About this App")
st.sidebar.success(
    """
    This application was built to visualize and compare the results of
    three pre-trained deep learning models on time-series data.
    """
)

# --- MAIN PAGE ---

all_files_exist = all([
    os.path.exists(DATA_FILE_PATH),
    os.path.exists(CNN_FILE_PATH),
    os.path.exists(LSTM_FILE_PATH),
    os.path.exists(HYBRID_FILE_PATH)
])

if all_files_exist:
    
    with st.spinner("Preparing data (scaling, windowing)..."):
        processed_data = load_and_process_data(DATA_FILE_PATH)
    
    if processed_data:
        with st.spinner("Loading models and running predictions (this may take a moment)..."):
            predictions_inv, metrics_summary = load_models_and_predict(
                processed_data, CNN_FILE_PATH, LSTM_FILE_PATH, HYBRID_FILE_PATH
            )
        
        if predictions_inv:
            st.success("Data and models processed successfully!")
            
            st.header("Model Evaluation Summary (Test Set)")
            st.info("Comparison of metrics on the 20% test set. Lower error (RMSE, MAE, MAPE) is better. Higher Accuracy (100-MAPE) is better.")
            
            df_metrics = pd.DataFrame(metrics_summary)
            
            desired_columns = ['Model', 'RMSE', 'MAE', 'MAPE', 'Accuracy', 'MSE']
            existing_cols = [col for col in desired_columns if col in df_metrics.columns]
            df_metrics = df_metrics[existing_cols] 
            df_metrics = df_metrics.sort_values(by="RMSE").reset_index(drop=True)
            
            st.dataframe(df_metrics.style.format({
                'MSE': '{:.2f}', 
                'RMSE': '{:.2f}', 
                'MAE': '{:.2f}', 
                'MAPE': '{:.2f}%',
                'Accuracy': '{:.2f}%'
            }))
            
            st.header("Prediction Visualization")
            
            tab_train, tab_test, tab_gabungan = st.tabs([
                "📊 Train Comparison", 
                "📈 Test Comparison", 
                "📉 Combined (Train+Test)"
            ])

            with tab_train:
                st.info("Comparison graph on the 80% training data (in-sample).")
                display_option_train = st.selectbox(
                    "Select Models to Display (Train):",
                    ["All Models", "CNN Only", "LSTM Only", "Hybrid CNN-LSTM Only"],
                    key='select_train' 
                )
                title_train = f"Train Comparison - {display_option_train}"
                fig_train = create_comparison_plot(
                    "Train Comparison", display_option_train, processed_data, 
                    predictions_inv, title_train
                )
                st.plotly_chart(fig_train, use_container_width=True)

            with tab_test:
                st.info("Comparison graph on the 20% test data (out-of-sample).")
                display_option_test = st.selectbox(
                    "Select Models to Display (Test):",
                    ["All Models", "CNN Only", "LSTM Only", "Hybrid CNN-LSTM Only"],
                    key='select_test' 
                )
                title_test = f"Test Comparison - {display_option_test}"
                fig_test = create_comparison_plot(
                    "Test Comparison", display_option_test, processed_data, 
                    predictions_inv, title_test
                )
                st.plotly_chart(fig_test, use_container_width=True)

            with tab_gabungan:
                st.info("Combined (Train + Test) comparison graph with split line.")
                display_option_gabungan = st.selectbox(
                    "Select Models to Display (Combined):",
                    ["All Models", "CNN Only", "LSTM Only", "Hybrid CNN-LSTM Only"],
                    key='select_gabungan' 
                )
                title_gabungan = f"Combined (Train+Test) - {display_option_gabungan}"
                fig_gabungan = create_comparison_plot(
                    "Combined (Train+Test)", display_option_gabungan, processed_data, 
                    predictions_inv, title_gabungan
                )
                st.plotly_chart(fig_gabungan, use_container_width=True)

else:
    st.error("Required Files Not Found!")
    st.info(
        f"""
        Please ensure the following files are in the same folder as `app_demo.py`:
        - `{'DATASET_ASII_10_Tahun.xlsx'}` ({'FOUND' if os.path.exists(DATA_FILE_PATH) else 'NOT FOUND'})
        - `{'best_model_cnn.h5'}` ({'FOUND' if os.path.exists(CNN_FILE_PATH) else 'NOT FOUND'})
        - `{'best_model_lstm.h5'}` ({'FOUND' if os.path.exists(LSTM_FILE_PATH) else 'NOT FOUND'})
        - `{'best_model_hybrid.h5'}` ({'FOUND' if os.path.exists(HYBRID_FILE_PATH) else 'NOT FOUND'})
        """
    )