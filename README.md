# Dự đoán giá tiền Bitcoin bằng các mô hình học máy và học sâu 

## 1. Mô tả dự án
Xây dựng các mô hình học máy và học sâu để dự đoán giá tiền ảo, sau đó xây dựng web để trực quan hóa kết quả

## 2. Cấu trúc thư mục

```
.
├── code/                                  # Thư mục chứa các Script của các mô hình bài toán                  
│   └──CK_GRU.ipynb                        # Xây dựng mô hình Gated Recurrent Units
│   └──CK_ARIMA.ipynb                      # Xây dựng mô hình Autoregressive Intergrated Moving Average
│   └──CK_LSTM tối ưu .iypnb               # Xây dựng mô hình Long Shot Term Memory với tối ưu tham số
│   └──CK_LSTM.ipynb                       # Xây dựng mô hình Long Shot Term Memory
│   └──CK_SVR.ipynb                        # Xây dựng mô hình Support Vector Regression
│   └──CK_lasso.iypnb                      # Xây dựng mô hình Support Lasso Regression
│   └──CK_linear.ipynb                     # Xây dựng mô hình Support Linear Regression
│   └──CK_polynomial.ipynb                 # Xây dựng mô hình Support Polynomial Regression
│   └──CK_convLSTM.iypnb                   # Xây dựng mô hình Long Shot Term Memory với hằng số cho trước
│   └──program.py                          # Tổng hợp chương trình và xây dựng web với Stream Lit
│   └──xu_ly_du_lieu.py                    # Xử lý và chuẩn hóa dữ liệu đầu vào
├── Bitcoin 2.csv                          # Dữ liệu của dự án
└── report.pdf                             # Báo cáo của dự án
```

## 3. Kết quả các mô hình
| STT | Tên mô hình | Điều kiện dừng | Phương pháp tối ưu siêu tham số | Siêu tham số mô hình | Kết quả đánh giá (Test Set) |
|:---:|:---|:---|:---|:---|:---|
| 1 | Linear Regression | Mặc định | - | - (Baseline) | RMSE: 1974.59<br>MAE: 1398.98<br>R2: 0.9891<br>MAPE: 1.98% |
| 2 | Lasso Regression | Sai số $\le$ 0.0001,<br>Số lần lặp = 10000 | LassoCV | Alpha = 0.1 | RMSE: 1970.59<br>MAE: 1388.98<br>R2: 0.9891<br>MAPE: 1.98% |
| 3 | Polynomial Regression | Mặc định | GridSearchCV | Degree (bậc đa thức) = 2 | RMSE: 1982.89<br>MAE: 1407.12<br>R2: 0.9892<br>MAPE: 2.01% |
| 4 | ARIMA | Mặc định | AutoArima | p (AR) = 5<br>d (Sai phân) = 1<br>q (MA) = 0 | RMSE: 1971.3<br>MAE: 1382.27<br>R2: 0.989<br>MAPE: 1.98% |
| 5 | SVR | Mặc định | GridSearchCV | C (điều chuẩn): 100<br>Epsilon: 0.01<br>Gamma: 0.01<br>Kernel: linear | RMSE: 4050.4<br>MAE: 2755.55<br>R2: 0.9549<br>MAPE: 3.40% |
| 6 | Random Forest (RF) | Mặc định | RandomizedSearchCV | Số lượng cây: 100<br>Max features: 7<br>Depth: 10 | RMSE: 3107.3<br>MAE: 2356.34<br>R2: 0.9567<br>MAPE: 3.03% |
| 7 | LSTM | Max 100 epochs;<br>Early stopping<br>(val_loss, patience=10) | Keras Tuner | Số lớp LSTM: 1<br>Units: 96<br>Dropout: 0.5<br>Tốc độ học: 1.42e-4<br>Dense units: 32<br>Loss rate: 0.2 | RMSE: 4015.62<br>MAE: 3045.34<br>R2: 0.9537<br>MAPE: 4.38% |
| 8 | GRU | Max 100 epochs;<br>Early stopping<br>(val_loss, patience=10) | Keras Tuner | Time steps: 60<br>Số lớp GRU: 3<br>Units/lớp: 64/64/32<br>Dropout: 0.2<br>Dense units: 16<br>Loss rate: 0.2 | RMSE: 4704.74<br>MAE: 3667.24<br>R2: 0.9365<br>MAPE: 4.88% |

## 4. Đưa kết quả lên web bằng thư viện Stream lit
<img width="1067" height="783" alt="Crypto" src="https://github.com/user-attachments/assets/deb23268-d72c-4a6d-81d6-476c2c89ff77" />

## Lisence 
Dự án được phát triển phục vụ học tập và nghiên cứu của cá nhân
