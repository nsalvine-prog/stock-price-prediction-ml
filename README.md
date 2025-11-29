# Advanced Stock Market Forecasting with Deep Learning & Statistical Models

## Overview
Stock market prediction is one of the most challenging tasks in financial analytics. Stock prices are influenced by market sentiment, global events, economic indicators, and company performance.  
This project analyzes stock data from four major technology companies:

- **Apple (AAPL)**
- **Microsoft (MSFT)**
- **Amazon (AMZN)**
- **Tesla (TSLA)**

The main objective is to **forecast Apple’s future stock prices** using multiple machine-learning and statistical models.  
Data is collected from **Yahoo Finance**, analyzed, engineered, preprocessed, and modeled using:

- **LSTM Deep Learning**
- **Prophet**
- **ARIMA / Auto-ARIMA**
- **XGBOOST**
- **Ridge Regression**

Performance is compared using RMSE, MAE, and R².

---

## Workflow of the Notebook

### **1. Data Collection**
Historical stock data (Open, High, Low, Close, Adj Close, Volume) is retrieved using:

- `yfinance.download()`

---

### **2. Exploratory Data Analysis (EDA)**
Includes:

- Price movement visualization  
- Moving Averages (MA10, MA20, MA50)  
- Correlation heatmaps  
- Volume trends  

---

### **3. Feature Engineering**
Technical indicators generated using **TA** library:

- Relative Strength Index (**RSI**)  
- MACD (Moving Average Convergence Divergence)  
- Bollinger Bands  
- EMAs & SMAs  
- Volatility features  

---

### **4. Preprocessing**
- Normalization / Scaling (MinMaxScaler)  
- Reshaping sequences for LSTM  
- Train-test split  
- Handling missing values  

---

### **5. Model Implementation**
Several forecasting models are trained:

#### **LSTM (Deep Learning)**
Captures long-term dependencies in sequential stock data.

#### **Prophet (Facebook)**
Good for seasonal + trend forecasting.

#### **ARIMA / Auto-ARIMA**
Classical time-series model for non-stationary data.

#### **XGBOOST Regressor**
Powerful gradient boosting model for tabular time-series features.

#### **Ridge Regression**
Simple baseline linear model.

---

### **6. Model Evaluation**
Models are evaluated using:

- **RMSE** (Root Mean Square Error)  
- **MAE** (Mean Absolute Error)  
- **R² Score**  
- Predicted vs. Actual Plots  

---

### **7. Results & Conclusion**
Summary includes:

- Performance comparison across all models  
- Best model for forecasting Apple stock  
- Visual forecast curves  
- Insights for improvements  

---

# Essential Libraries

## yfinance
Fetches historical market data from Yahoo Finance.

```python
%%capture
!pip install --upgrade yfinance
