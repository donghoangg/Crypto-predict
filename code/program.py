# program.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, date

# --- Cấu hình trang Streamlit ---
st.set_page_config(layout="wide", page_title="Dự Đoán Giá Bitcoin", page_icon="₿")

# --- Các hàm xử lý dữ liệu và mô hình (Giữ nguyên như cũ) ---
@st.cache_data
def load_and_preprocess_data(file_path='Bitcoin.csv'):
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file {file_path}. Vui lòng đảm bảo file nằm trong cùng thư mục.")
        return None, None, None, None, None

    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    except Exception as e:
        st.error(f"Lỗi khi chuyển đổi cột 'Date': {e}. Kiểm tra định dạng ngày trong file CSV.")
        return None, None, None, None, None
        
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    if 'Close' not in df.columns or 'Open' not in df.columns:
        st.error("File CSV thiếu cột 'Close' hoặc 'Open'.")
        return None, None, None, None, None

    df_processed = df[['Close', 'Open']].copy()
    for col in ['Close', 'Open']:
        if df_processed[col].dtype == 'object':
            try:
                df_processed[col] = df_processed[col].str.replace(',', '.', regex=False).astype(float)
            except Exception as e:
                st.error(f"Lỗi khi chuyển đổi cột '{col}' sang số: {e}. Kiểm tra dữ liệu trong cột.")
                return None, None, None, None, None

    df_processed.fillna(method='ffill', inplace=True)
    df_processed.fillna(method='bfill', inplace=True)

    if df_processed.isnull().values.any():
        st.error("Dữ liệu vẫn còn giá trị NaN sau khi fill. Kiểm tra lại file CSV.")
        return None, None, None, None, None

    target_col = 'Close'
    lags_list = [1, 3, 7, 14, 30]
    rolling_windows_list = [7, 30]

    for lag_val in lags_list:
        df_processed[f'Close_Lag_{lag_val}'] = df_processed[target_col].shift(lag_val)
    for window_val in rolling_windows_list:
        df_processed[f'Close_Rolling_Mean_{window_val}'] = df_processed[target_col].rolling(window=window_val, min_periods=1).mean().shift(1)
    for lag_val in lags_list:
        df_processed[f'Open_Lag_{lag_val}'] = df_processed['Open'].shift(lag_val)
    for window_val in rolling_windows_list:
        df_processed[f'Open_Rolling_Mean_{window_val}'] = df_processed['Open'].rolling(window=window_val, min_periods=1).mean().shift(1)

    df_processed.dropna(inplace=True)

    if df_processed.empty:
        st.error("Không còn dữ liệu sau khi tạo đặc trưng và loại bỏ NaN. Có thể do file CSV quá ít dòng.")
        return None, None, None, None, None

    feature_names_list = [f'Close_Lag_{lag_val}' for lag_val in lags_list] + \
                         [f'Close_Rolling_Mean_{window_val}' for window_val in rolling_windows_list] + \
                         [f'Open_Lag_{lag_val}' for lag_val in lags_list] + \
                         [f'Open_Rolling_Mean_{window_val}' for window_val in rolling_windows_list]
    
    return df_processed, feature_names_list, lags_list, rolling_windows_list, target_col

@st.cache_resource
def train_model(_df_processed, _feature_names, _target_col):
    if _df_processed is None or not _feature_names or _target_col not in _df_processed.columns:
        return None, None, None, None, None, None, None

    X = _df_processed[_feature_names]
    y = _df_processed[_target_col]

    if X.empty or y.empty:
        return None, None, None, None, None, None, None

    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train_orig, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test_orig, y_test = X.iloc[split_index:], y.iloc[split_index:]

    if len(X_train_orig) == 0:
        return None, None, None, None, None, None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    
    alpha_value = 0.1
    model = Lasso(alpha=alpha_value, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    
    y_pred_test_plot = None
    if not X_test_orig.empty:
        X_test_scaled_for_plot = scaler.transform(X_test_orig)
        y_pred_test_plot = model.predict(X_test_scaled_for_plot)

    return model, scaler, X_train_orig, y_train, X_test_orig, y_test, y_pred_test_plot

def predict_for_future_date(target_date_dt, model, scaler, historical_data_full, 
                            feature_names_list, lags_list, rolling_windows_list, target_col_name='Close'):
    last_known_date_dt = historical_data_full.index[-1]
    current_data_df = historical_data_full[[target_col_name, 'Open']].copy()
    features_for_target_prediction_unscaled_df = None 

    num_days_to_predict = (target_date_dt - last_known_date_dt).days

    if num_days_to_predict <= 0: 
        if target_date_dt in historical_data_full.index:
            actual_val = historical_data_full.loc[target_date_dt, target_col_name]
            if all(f in historical_data_full.columns for f in feature_names_list):
                 features_for_target_prediction_unscaled_df = historical_data_full.loc[[target_date_dt], feature_names_list]
            return actual_val, current_data_df, features_for_target_prediction_unscaled_df
        else:
            return None, current_data_df, None 

    for i in range(num_days_to_predict):
        current_prediction_date_dt = last_known_date_dt + timedelta(days=i + 1)
        next_day_feature_values = {}

        for lag_val in lags_list:
            next_day_feature_values[f'Close_Lag_{lag_val}'] = current_data_df[target_col_name].iloc[-lag_val]
        for window_val in rolling_windows_list:
            next_day_feature_values[f'Close_Rolling_Mean_{window_val}'] = current_data_df[target_col_name].iloc[-window_val:].mean()
        for lag_val in lags_list:
            next_day_feature_values[f'Open_Lag_{lag_val}'] = current_data_df['Open'].iloc[-lag_val]
        for window_val in rolling_windows_list:
            next_day_feature_values[f'Open_Rolling_Mean_{window_val}'] = current_data_df['Open'].iloc[-window_val:].mean()

        next_day_features_df_orig = pd.DataFrame([next_day_feature_values], columns=feature_names_list, index=[current_prediction_date_dt])
        next_day_features_scaled = scaler.transform(next_day_features_df_orig)
        prediction = model.predict(next_day_features_scaled)[0]

        new_row = pd.DataFrame({target_col_name: [prediction], 'Open': [prediction]}, index=[current_prediction_date_dt])
        current_data_df = pd.concat([current_data_df, new_row])
        
        if current_prediction_date_dt == target_date_dt:
            features_for_target_prediction_unscaled_df = next_day_features_df_orig

    if target_date_dt in current_data_df.index:
        return current_data_df.loc[target_date_dt, target_col_name], current_data_df, features_for_target_prediction_unscaled_df
    return None, current_data_df, None

# --- Bắt đầu Giao diện Streamlit ---
st.title("₿ Bảng Điều Khiển Dự Đoán Giá Bitcoin")
st.markdown("Một ứng dụng web sử dụng mô hình **Lasso Regression** để dự báo giá đóng cửa của Bitcoin trong tương lai.")

# --- Tải và xử lý dữ liệu ---
df_processed, feature_names, lags, rolling_windows, target_col = load_and_preprocess_data()

if df_processed is None:
    st.error("Không thể tải hoặc xử lý dữ liệu. Vui lòng kiểm tra file 'Bitcoin.csv' và định dạng của nó.")
    st.stop()

# --- Huấn luyện mô hình ---
model, scaler, X_train_orig, y_train, X_test_orig, y_test, y_pred_test_plot = train_model(df_processed, feature_names, target_col)

if model is None:
    st.error("Không thể huấn luyện mô hình. Kiểm tra lại dữ liệu và các bước tiền xử lý.")
    st.stop()

# --- Bố cục giao diện với Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Bảng điều khiển & Dự đoán", "🔬 Phân tích Mô hình", "🗃️ Xem Dữ liệu"])

# ==============================================================================
# --- TAB 1: BẢNG ĐIỀU KHIỂN & DỰ ĐOÁN ---
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1, 3]) # Cột input nhỏ hơn, cột output lớn hơn

    # --- Cột 1: Input của người dùng ---
    with col1:
        st.subheader("⚙️ Tùy chọn Dự đoán")
        
        last_date_in_data_dt = df_processed.index[-1].date()
        default_prediction_date = last_date_in_data_dt + timedelta(days=7)
        
        selected_date = st.date_input(
            "Chọn ngày muốn dự đoán:",
            value=default_prediction_date,
            min_value=df_processed.index.min().date(),
            help="Chọn một ngày trong tương lai để mô hình dự đoán giá."
        )
        selected_date_dt = pd.to_datetime(selected_date)

        days_to_show = st.slider(
            "Số ngày lịch sử hiển thị trên biểu đồ:",
            min_value=90,
            max_value=len(df_processed),
            value=365,
            step=30,
            help="Kéo để thay đổi khoảng thời gian lịch sử được vẽ trên biểu đồ."
        )

        if st.button("🚀 Chạy Dự Đoán", type="primary", use_container_width=True):
            st.session_state.run_prediction = True
            st.session_state.selected_date_dt = selected_date_dt
            st.session_state.days_to_show = days_to_show
        
        st.markdown("---")
        st.info(f"Dữ liệu được cập nhật lần cuối vào: **{last_date_in_data_dt.strftime('%d-%m-%Y')}**")

    # --- Cột 2: Hiển thị kết quả ---
    with col2:
        st.subheader("📊 Kết quả & Biểu đồ")
        
        # Placeholder cho kết quả
        result_placeholder = st.empty()

        # Logic hiển thị kết quả
        if 'run_prediction' in st.session_state and st.session_state.run_prediction:
            with st.spinner("Đang tính toán dự đoán..."):
                predicted_price, extended_data, features_df = predict_for_future_date(
                    st.session_state.selected_date_dt, model, scaler, df_processed,
                    feature_names, lags, rolling_windows, target_col
                )

            if predicted_price is not None:
                # Hiển thị các chỉ số
                metric_cols = st.columns(3)
                metric_cols[0].metric(
                    label=f"Giá dự đoán ngày {st.session_state.selected_date_dt.strftime('%d-%m-%Y')}",
                    value=f"${predicted_price:,.2f}"
                )
                
                # So sánh với ngày trước đó
                previous_day_price = df_processed[target_col].iloc[-1]
                delta = predicted_price - previous_day_price
                metric_cols[1].metric(
                    label=f"So với ngày cuối cùng ({last_date_in_data_dt.strftime('%d-%m-%Y')})",
                    value=f"${previous_day_price:,.2f}",
                    delta=f"${delta:,.2f}"
                )

                # Biểu đồ
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Lấy dữ liệu lịch sử để vẽ
                history_to_plot = df_processed.tail(st.session_state.days_to_show)
                ax.plot(history_to_plot.index, history_to_plot[target_col], label='Giá Lịch Sử', color='dodgerblue', lw=2)

                # Vẽ phần dự đoán trong tương lai
                # Sửa dòng 256
                if st.session_state.selected_date_dt.date() > last_date_in_data_dt:
    # Lấy ra phần dữ liệu dự đoán trong tương lai từ `extended_data`
    # Điều kiện so sánh ở đây cũng cần nhất quán
                    forecast_period = extended_data[extended_data.index.date > last_date_in_data_dt]
                    if not forecast_period.empty:
                        ax.plot(forecast_period.index, forecast_period[target_col], label='Đường Dự Đoán', color='darkorange', linestyle='--', marker='o', markersize=4)
                # Đánh dấu điểm dự đoán
                ax.scatter([st.session_state.selected_date_dt], [predicted_price], color='red', s=100, zorder=5, label=f'Điểm Dự Đoán')
                
                ax.set_title(f"Lịch sử giá và Dự đoán cho ngày {st.session_state.selected_date_dt.strftime('%d-%m-%Y')}", fontsize=14)
                ax.set_ylabel("Giá Đóng Cửa (USD)", fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Hiển thị các đặc trưng trong expander
                with st.expander("Xem các đặc trưng được sử dụng cho dự đoán"):
                    if features_df is not None and not features_df.empty:
                        st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Giá trị"}).style.format("{:,.2f}"))
                    else:
                        st.warning("Không có thông tin đặc trưng cho ngày này.")

            else:
                st.error("Không thể thực hiện dự đoán. Vui lòng thử lại.")
        else:
            result_placeholder.info("Hãy chọn ngày và nhấn nút 'Chạy Dự Đoán' để xem kết quả.")


# ==============================================================================
# --- TAB 2: PHÂN TÍCH MÔ HÌNH ---
# ==============================================================================
with tab2:
    st.header("🔬 Phân tích Mô hình Lasso Regression")
    
    st.subheader("1. Hiệu suất Mô hình trên Tập Dữ liệu Kiểm tra (Test Set)")
    st.markdown("Biểu đồ dưới đây so sánh giá thực tế (màu xanh) và giá mô hình dự đoán (màu cam) trên 20% dữ liệu cuối cùng (tập test) để đánh giá độ chính xác của mô hình.")
    
    fig_test, ax_test = plt.subplots(figsize=(12, 6))
    ax_test.plot(y_test.index, y_test, label='Giá Thực Tế (Actual)', color='dodgerblue')
    ax_test.plot(y_test.index, y_pred_test_plot, label='Giá Dự Đoán (Predicted)', color='darkorange', linestyle='--')
    ax_test.set_title("So sánh Giá Thực Tế và Dự Đoán trên Tập Test", fontsize=14)
    ax_test.set_ylabel("Giá Đóng Cửa (USD)", fontsize=10)
    ax_test.legend()
    ax_test.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig_test)

    st.subheader("2. Mức độ Quan trọng của các Đặc trưng (Feature Importances)")
    st.markdown("Mô hình Lasso gán một 'hệ số' (coefficient) cho mỗi đặc trưng. Hệ số càng lớn (cả âm và dương) cho thấy đặc trưng đó càng có ảnh hưởng lớn đến kết quả dự đoán.")

    # Lấy hệ số và tạo DataFrame
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    # Vẽ biểu đồ cột
    fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
    ax_coef.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
    ax_coef.invert_yaxis() # Hiển thị đặc trưng quan trọng nhất ở trên cùng
    ax_coef.set_title("Hệ số của các Đặc trưng trong Mô hình Lasso", fontsize=14)
    ax_coef.set_xlabel("Giá trị Hệ số (Coefficient)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_coef)


# ==============================================================================
# --- TAB 3: XEM DỮ LIỆU ---
# ==============================================================================
with tab3:
    st.header("🗃️ Dữ liệu đã được Xử lý")
    st.markdown("Đây là bảng dữ liệu đã được làm sạch, tạo đặc trưng và sẵn sàng để đưa vào mô hình.")
    
    # Chuyển đổi dữ liệu sang CSV để tải xuống
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')

    csv_data = convert_df_to_csv(df_processed)

    st.download_button(
        label="📥 Tải Dữ liệu (CSV)",
        data=csv_data,
        file_name='processed_bitcoin_data.csv',
        mime='text/csv',
    )
    
    st.dataframe(df_processed)