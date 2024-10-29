import numpy as np 
import pandas as pd
import yfinance as yf
import os
from keras.models import load_model
import streamlit as st
from datetime import date
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def load_data(company):
    start = '2010-01-01'
    end = date.today().strftime("%Y-%m-%d")
    df = yf.download(company, start=start, end=end)
    df.reset_index(inplace=True)    
    return df


def describe_data(df):
    st.markdown('1. Daily Stock Price Last 7 Days')
    st.write(df.tail(7))


def plot_raw_data(df):
    st.markdown('2. Data Visualization')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='stock_close'))
    fig.update_layout(
        title='Opening and Closing Price time series data',
        xaxis_rangeslider_visible=True,
    )
    st.plotly_chart(fig)
    

def plot_candle_chart(df):
    df_last_100 = df.tail(100)

    fig_candlestick = go.Figure(data=[go.Candlestick(x=df_last_100['Date'],
                                                    open=df_last_100['Open'],
                                                    high=df_last_100['High'],
                                                    low=df_last_100['Low'],
                                                    close=df_last_100['Close'],
                                                    increasing_line_color='green',
                                                    decreasing_line_color='red')])

    fig_candlestick.update_layout(title='Candlestick chart for last 100 days data',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                xaxis=dict(type='category', tickformat='%Y-%m-%d', tickangle=-45),
                                xaxis_rangeslider_visible=False, 
                                width=1200, height=600) 

    st.plotly_chart(fig_candlestick)


def plot_ma_data(df):
    ma20 = df.Close.rolling(20).mean()
    ma100 = df.Close.rolling(100).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='stock_close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=ma20, name='ma20'))
    fig.add_trace(go.Scatter(x=df['Date'], y=ma100, name='ma100'))
    
    fig.layout.update(title_text='Closing Price vs. time chart with Moving Average 20 & Moving Average 100', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
     

# Function to load model from Models
current_directory = os.path.dirname(__file__)
model_directory = os.path.join(current_directory, 'Models')

def load_keras_model(model_filename):
    model_path = os.path.join(model_directory, model_filename)
    return load_model(model_path)


def create_testing_set(df):
    n1 = int(len(df)*0.80)
    n2 = int(len(df)*0.10)

    data_training = pd.DataFrame(df['Close'][0:n1])
    data_validating = pd.DataFrame(df['Close'][n1:(n1+n2)])
    data_testing = pd.DataFrame(df['Close'][(n1+n2):])
    scaler = MinMaxScaler(feature_range=(0,1))

    days = data_training.tail(100)
    data_validating = pd.concat([days, data_validating], ignore_index=True)

    past_100_days = data_validating.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)   
    
    return input_data, x_test, y_test, scaler, n1, n2


def validate_model(model, df):
    test_set, x_test, y_test, scaler, n1, n2 = create_testing_set(df)
    
    # Predict Testing Set
    y_predicted = model.predict(x_test)
    y_predicted_inv = scaler.inverse_transform(y_predicted.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Visualization
    fig1 = go.Figure()
    dates_test = df['Date'][(n1+n2):].reset_index(drop=True)
    fig1.add_trace(go.Scatter(x=dates_test, y=y_test_inv.flatten(), name='actual'))
    fig1.add_trace(go.Scatter(x=dates_test, y=y_predicted_inv.flatten(), name='predicted'))
    fig1.layout.update(title_text='Actual vs. Predicted Close Prices', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)
    
    # Validate model
    mae = mean_absolute_error(y_test_inv, y_predicted_inv)
    mse = mean_squared_error(y_test_inv, y_predicted_inv)
    rmse = np.sqrt(mse)
    mape = (np.abs((y_test_inv - y_predicted_inv) / y_test_inv).mean()) * 100
    r2 = r2_score(y_test_inv, y_predicted_inv)

    st.write('Evaluate model on testing set:')
    st.write(f'MAE: {mae}')
    st.write(f'MSE: {mse}')
    st.write(f'RMSE: {rmse}')
    st.write(f'MAPE: {mape}%')
    st.write(f'R²: {r2}')
    
    
    # Predict Future
    start_date = dates_test.iloc[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=5)
    dates100 = df['Date'][-100:].reset_index(drop=True)
    y_100 = y_test_inv[-100:]
    y_pred_100 = y_predicted_inv[-100:]
    future_inputs = np.array(test_set[-100:])
    future_predictions = []

    for i in range(5): 
        x_future = np.reshape(future_inputs, (1, future_inputs.shape[0], 1))
        future_prediction = model.predict(x_future)
        future_predictions.append(future_prediction[0, 0])
        
        future_inputs = np.roll(future_inputs, -1) 
        future_inputs[-1] = future_prediction  

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates100, y=y_100.flatten(), mode='lines', name='actual'))
    fig.add_trace(go.Scatter(x=dates100, y=y_pred_100.flatten(), mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='future_predicted'))
    fig.update_layout(title='Predicted Close Prices for next 5 days',
                    xaxis_title='Date',
                    yaxis_title='Price')

    st.plotly_chart(fig)

    # Show Predict Table
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Close': future_predictions.flatten()
    })

    st.write("Prediction Detail")
    st.write(future_df)
    
    # Create present & future Close Price dataframe
    present_df = df[['Date', 'Close']].copy()
    tail_present_df = present_df.tail(11)
    perf_df = pd.concat([tail_present_df, future_df], ignore_index=True)
    
    return perf_df



###############################################################
st.title('Apple Inc. Stock Performance vs. S&P 500 Prediction')

st.subheader('Apple Inc. Stock Price')

# Load AAPL Data
data_load_state = st.text("Loading Apple Inc. stock data ...")
aapl_df = load_data('AAPL')
data_load_state.text("Data loaded successfully!")


# Describing AAPL Data
describe_data(aapl_df)


# AAPL Data Visualization
plot_raw_data(aapl_df)
plot_candle_chart(aapl_df)
plot_ma_data(aapl_df)


st.markdown('3. AAPL Stock Price Prediction')

# Load AAPL Model
model_names = [
    "AAPL_Model1___BasicLSTM",
    "AAPL_Model2___StackLSTM",
    "AAPL_Model3___BiLSTM",
    "AAPL_Model4___CNN-LSTM",
    "AAPL_Model5___CNN-BiLSTM",
    "AAPL_Model6___BiCuDNN-BiLSTM",
    "AAPL_Model7___RS-HCNN-BiLSTM",
    "AAPL_Model8___GS-HCNN-BiLSTM",
    "AAPL_Model9___DD-LSTM",
    "AAPL_Model10___IDD-LSTM",
    "AAPL_Model11___BasicGRU",
    "AAPL_Model12___VMD-GRU",
    "AAPL_Model13___StackGRU",
    "AAPL_Model14___Stack-VMD-GRU",
    "AAPL_Model15___BiGRU"
]

selected_model_name = st.selectbox("Model to predict Apple Inc. stock price:", model_names)
model_name = selected_model_name + ".keras"
aapl_model = load_keras_model(model_name)
st.write(f"{selected_model_name} loaded successfully!")


# Predict AAPL Stock Price
aapl_perf_df = validate_model(aapl_model, aapl_df)




st.subheader('S&P 500 Stock Price')

# Load S&P 500 Data
data_load_state = st.text("Loading S&P 500 stock data ...")
sp500_df = load_data('^GSPC')
data_load_state.text("Data loaded successfully!")


# Describing S&P 500 Data
describe_data(aapl_df)


# S&P 500 Data Visualization
plot_raw_data(sp500_df)
plot_candle_chart(sp500_df)
plot_ma_data(sp500_df)


st.markdown('3. S&P 500 Stock Price Prediction')

# Load S&P 500 Model
model_names = [
    "SP500_Model1___BasicLSTM",
    "SP500_Model2___StackLSTM",
    "SP500_Model3___BiLSTM",
    "SP500_Model4___CNN-LSTM",
    "SP500_Model5___CNN-BiLSTM",
    "SP500_Model6___BiCuDNN-BiLSTM",
    "SP500_Model7___RS-HCNN-BiLSTM",
    "SP500_Model8___GS-HCNN-BiLSTM",
    "SP500_Model9___DD-LSTM",
    "SP500_Model10___IDD-LSTM",
    "SP500_Model11___BasicGRU",
    "SP500_Model12___VMD-GRU",
    "SP500_Model13___StackGRU",
    "SP500_Model14___Stack-VMD-GRU",
    "SP500_Model15___BiGRU"
]

selected_model_name = st.selectbox("Model to predict S&P 500 stock price:", model_names)
model_name = selected_model_name + ".keras"
sp500_model = load_keras_model(model_name)
st.write(f"{selected_model_name} loaded successfully!")


# Predict S&P 500 Stock Price
sp500_perf_df = validate_model(sp500_model, sp500_df)

st.subheader('Apple Inc. Stock Price Performance vs. S&P 500 Prediction')

# Calculate table
data = pd.merge(aapl_perf_df, sp500_perf_df, on='Date', suffixes=('_AAPL', '_SP500'))

data['Price Relative'] = data['Close_AAPL'] / data['Close_SP500']
data['AAPL % Change'] = data['Close_AAPL'].pct_change() * 100
data['SP500 % Change'] = data['Close_SP500'].pct_change() * 100
data['% Change in PR'] = data['Price Relative'].pct_change() * 100
data['Difference'] = data['AAPL % Change'] - data['SP500 % Change']

data = data.dropna()
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

# Display the table
def color(val):
    if val > 0: color='blue' 
    elif val == 0: color='white'
    else: color='red'
    return 'color: %s' % color

data_styled = data.style.map(color, subset=['% Change in PR', 'AAPL % Change', 'SP500 % Change', 'Difference'])

st.write("Performance Comparison Table")
st.write(data_styled)