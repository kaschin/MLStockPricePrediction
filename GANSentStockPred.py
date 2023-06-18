import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from keras.utils.vis_utils import plot_model
from pickle import load
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import statsmodels.api as sm
from math import sqrt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import warnings
import nltk
warnings.filterwarnings("ignore")

STOCK_NAME = 'GOOG'
df = pd.read_csv('Data/stock_tweets.csv', error_bad_lines=False)

def Get_Tech_Ind(data):
    data['MA7'] = data.iloc[:,4].rolling(window=7).mean() 
    data['MA20'] = data.iloc[:,4].rolling(window=20).mean()
    data['MACD'] = data.iloc[:,4].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()
    data['20SD'] = data.iloc[:, 4].rolling(20).std()
    data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA20'] - (data['20SD'] * 2)
    data['EMA'] = data.iloc[:,4].ewm(com=0.5).mean()
    data['logmomentum'] = np.log(data.iloc[:,4] - 1)

    return data

def Tech_Ind(dataset):
    fig,ax = plt.subplots(figsize=(15, 8), dpi = 200)
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)
    ax.plot(dataset['Date'], dataset['MA7'], label='Moving Average (7 days)', color='g', linestyle='--')
    ax.plot(dataset['Date'], dataset['Close'], label='Closing Price', color='#6A5ACD')
    ax.plot(dataset['Date'], dataset['MA20'], label='Moving Average (20 days)', color='r', linestyle='-.')
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title('Technical indicators')
    plt.ylabel('Close (USD)')
    plt.xlabel("Year")
    plt.legend()
    plt.show()

def Get_Batch_Data(Data_X, Data_Y, BATCH_SZ, predict_period):
    XBATCHED, YBATCHED, YC = list(), list(), list()

    for i in range(0,len(Data_X),1):
        XVAL = Data_X[i: i + BATCH_SZ][:, :]
        YVAL = Data_Y[i + BATCH_SZ: i + BATCH_SZ + predict_period][:, 0]
        YC_value = Data_Y[i: i + BATCH_SZ][:, :]

        if len(XVAL) == BATCH_SZ and len(YVAL) == predict_period:
            XBATCHED.append(XVAL)
            YBATCHED.append(YVAL)
            YC.append(YC_value)

    return np.array(XBATCHED), np.array(YBATCHED), np.array(YC)

def Normalise(df, range, target_column):

    target_df_series = pd.DataFrame(df[target_column])
    data = pd.DataFrame(df.iloc[:, :])

    X_SCALER = MinMaxScaler(feature_range=range)
    Y_SCALER = MinMaxScaler(feature_range=range)
    X_SCALER.fit(data)
    Y_SCALER.fit(target_df_series)
    X_SCALE_DF = X_SCALER.fit_transform(data)
    Y_SCALE_DF = Y_SCALER.fit_transform(target_df_series)
    dump(X_SCALER, open('X_SCALER.pkl', 'wb'))
    dump(Y_SCALER, open('Y_SCALER.pkl', 'wb'))

    return (X_SCALE_DF, Y_SCALE_DF)

def Split_Train_Test(data):
    SIZE = len(data) - 20
    DataTrain = data[0:SIZE]
    DataTest = data[SIZE:]

    return DataTrain, DataTest

def Predict_IDX(dataset, X_Train, BATCH_SZ, PRED_PER):
    Train_Predict_IDX = dataset.iloc[BATCH_SZ: X_Train.shape[0] + BATCH_SZ + PRED_PER, :].index
    Test_Predict_IDX = dataset.iloc[X_Train.shape[0] + BATCH_SZ:, :].index

    return Train_Predict_IDX, Test_Predict_IDX

def Generator_Model(InpuDim, OutputDim, FeatureSize):
    model = tf.keras.Sequential([LSTM(units = 1024, return_sequences = True,
                                    input_shape=(InpuDim, FeatureSize),recurrent_dropout = 0.3),
                               LSTM(units = 512, return_sequences = True, recurrent_dropout = 0.3),
                               LSTM(units = 256, return_sequences = True, recurrent_dropout = 0.3),
                               LSTM(units = 128, return_sequences = True, recurrent_dropout = 0.3),
                               LSTM(units = 64, recurrent_dropout = 0.3),
                               Dense(32),
                               Dense(16),
                               Dense(8),
                               Dense(units=OutputDim)])
    return model


def Discriminator_Model(InpuDim):
    CNN_NET = tf.keras.Sequential()
    CNN_NET.add(Conv1D(8, input_shape=(InpuDim+1, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    CNN_NET.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    CNN_NET.add(Conv1D(32, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    CNN_NET.add(Conv1D(64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    CNN_NET.add(Conv1D(128, kernel_size=1, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    CNN_NET.add(LeakyReLU())
    CNN_NET.add(Dense(220, use_bias=False))
    CNN_NET.add(LeakyReLU())
    CNN_NET.add(Dense(220, use_bias=False, activation='relu'))
    CNN_NET.add(Dense(1, activation='sigmoid'))
    return CNN_NET

def Calc_Discriminator_Loss(Output_Real, Output_Fake):
    Loss_F = tf.keras.losses.BinarYCrossentropy(from_logits=True)
    REAL_LOSS = Loss_F(tf.ones_like(Output_Real), Output_Real)
    FAKE_LOSS = Loss_F(tf.zeros_like(Output_Fake), Output_Fake)
    TOTAL_LOSS = REAL_LOSS + FAKE_LOSS
    return TOTAL_LOSS

def Calc_Generator_Loss(Output_Fake):
    Loss_F = tf.keras.losses.BinarYCrossentropy(from_logits=True)
    LOSS = Loss_F(tf.ones_like(Output_Fake), Output_Fake)
    return LOSS

def Train_Step(X_Real, Y_Real, YC, Generator, Discriminator, G_Optimiser, D_Optimiser):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        Generated_Data = Generator(X_Real, Training=True)
        Generated_Data_reshape = tf.reshape(Generated_Data, [Generated_Data.shape[0], Generated_Data.shape[1], 1])
        D_Fake_Input = tf.concat([tf.cast(Generated_Data_reshape, tf.float64), YC], axis=1)
        Y_Real_Reshape = tf.reshape(Y_Real, [Y_Real.shape[0], Y_Real.shape[1], 1])
        D_Real_Input = tf.concat([Y_Real_Reshape, YC], axis=1)

        Output_Real = Discriminator(D_Real_Input, Training=True)
        Output_Fake = Discriminator(D_Fake_Input, Training=True)

        G_Loss = Calc_Generator_Loss(Output_Fake)
        Disc_Loss = Calc_Discriminator_Loss(Output_Real, Output_Fake)

    Generator_Gradients = gen_tape.gradient(G_Loss, Generator.Trainable_variables)
    Discriminator_Gradients = disc_tape.gradient(Disc_Loss, Discriminator.Trainable_variables)

    G_Optimiser.apply_gradients(zip(Generator_Gradients, Generator.Trainable_variables))
    D_Optimiser.apply_gradients(zip(gradients_of_Discriminator, Discriminator.Trainable_variables))

    return Y_Real, Generated_Data, {'d_loss': Disc_Loss, 'G_Loss': G_Loss}

def Train(X_Real, Y_Real, YC, EPOCHS, Generator, Discriminator, G_Optimiser, D_Optimiser, checkpoint = 50):
    Train_info = {}
    Train_info["Calc_Discriminator_Loss"] = []
    Train_info["Calc_Generator_Loss"] = []

    for epoch in tqdm(range(EPOCHS)):
        RealPrice, fake_price, loss = Train_Step(X_Real, Y_Real, YC, Generator, Discriminator, G_Optimiser, D_Optimiser)
        G_Losses = []
        D_losses = []
        RealPrice = []
        PredictedPrice = []
        D_losses.append(loss['d_loss'].numpy())
        G_Losses.append(loss['G_Loss'].numpy())
        PredictedPrice.append(fake_price.numpy())
        RealPrice.append(RealPrice.numpy())
        if (epoch + 1) % checkpoint == 0:
            tf.keras.models.save_model(Generator, f'./models_gan/{STOCK_NAME}/Generator_V_%d.h5' % epoch)
            tf.keras.models.save_model(Discriminator, f'./models_gan/{STOCK_NAME}/Discriminator_V_%d.h5' % epoch)
            print('epoch', epoch + 1, 'Calc_Discriminator_Loss', loss['d_loss'].numpy(), 'Calc_Generator_Loss', loss['G_Loss'].numpy())

        Train_info["Calc_Discriminator_Loss"].append(D_losses)
        Train_info["Calc_Generator_Loss"].append(G_Losses)

    PredictedPrice = np.array(PredictedPrice)
    PredictedPrice = PredictedPrice.reshape(PredictedPrice.shape[1], PredictedPrice.shape[2])
    RealPrice = np.array(RealPrice)
    RealPrice = RealPrice.reshape(RealPrice.shape[1], RealPrice.shape[2])

    plt.subplot(2,1,1)
    plt.plot(Train_info["Calc_Discriminator_Loss"], label='Disc_Loss', color='#000000')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(Train_info["Calc_Generator_Loss"], label='Gen_loss', color='#000000')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.legend()

    plt.show()

    return PredictedPrice, RealPrice, np.sqrt(mean_squared_error(RealPrice, PredictedPrice)) / np.mean(RealPrice)

def Evaluate_Op(Generator, X_Real):
    Generated_Data = Generator(X_Real, Training = False)
    return Generated_Data

def PlotTestData(RealTestPrice, Predicted_test_price, Index_Test):
    X_SCALER = load(open('X_SCALER.pkl', 'rb'))
    Y_SCALER = load(open('Y_SCALER.pkl', 'rb'))
    Test_Predict_IDX = Index_Test
    Rescaled_RealPrice = Y_SCALER.inverse_transform(RealTestPrice)
    Rescaled_PredictedPrice = Y_SCALER.inverse_transform(Predicted_test_price)
    PredictedResult = pd.DataFrame()

    for i in range(Rescaled_PredictedPrice.shape[0]):
        Y_PRED = pd.DataFrame(Rescaled_PredictedPrice[i], columns=["PredictedPrice"], index=Test_Predict_IDX[i:i+OutputDim])
        PredictedResult = pd.concat([PredictedResult, Y_PRED], axis=1, sort=False)

    RealPrice = pd.DataFrame()
    for i in range(Rescaled_RealPrice.shape[0]):
        Y_Train = pd.DataFrame(Rescaled_RealPrice[i], columns=["RealPrice"], index=Test_Predict_IDX[i:i+OutputDim])
        RealPrice = pd.concat([RealPrice, Y_Train], axis=1, sort=False)

    PredictedResult['predicted_mean'] = PredictedResult.mean(axis=1)
    RealPrice['real_mean'] = RealPrice.mean(axis=1)

    predicted = PredictedResult["predicted_mean"]
    real = RealPrice["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('Test RMSE: ', RMSE)

    plt.figure(figsize=(16, 8))
    plt.plot(RealPrice["real_mean"], color='#00008B')
    plt.plot(PredictedResult["predicted_mean"], color = '#8B0000', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title(f"Prediction on test data for {STOCK_NAME}", fontsize=20)
    plt.show()

def Plot_Results(RealPrice, PredictedPrice, IDX_Train):
    X_SCALER = load(open('/content/X_SCALER.pkl', 'rb'))
    Y_SCALER = load(open('/content/Y_SCALER.pkl', 'rb'))
    Train_Predict_IDX = IDX_Train
    Rescaled_RealPrice = Y_SCALER.inverse_transform(RealPrice)
    Rescaled_PredictedPrice = Y_SCALER.inverse_transform(PredictedPrice)
    PredictedResult = pd.DataFrame()
    for i in range(Rescaled_PredictedPrice.shape[0]):
        Y_PRED = pd.DataFrame(Rescaled_PredictedPrice[i], columns=["PredictedPrice"], index=Train_Predict_IDX[i:i+OutputDim])
        PredictedResult = pd.concat([PredictedResult, Y_PRED], axis=1, sort=False)

    RealPrice = pd.DataFrame()
    for i in range(Rescaled_RealPrice.shape[0]):
        Y_Train = pd.DataFrame(Rescaled_RealPrice[i], columns=["RealPrice"], index=Train_Predict_IDX[i:i+OutputDim])
        RealPrice = pd.concat([RealPrice, Y_Train], axis=1, sort=False)

    PredictedResult['predicted_mean'] = PredictedResult.mean(axis=1)
    RealPrice['real_mean'] = RealPrice.mean(axis=1)

    plt.figure(figsize=(16, 8))
    plt.plot(RealPrice["real_mean"])
    plt.plot(PredictedResult["predicted_mean"], color = 'r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("The result of Training", fontsize=20)
    plt.show()

    predicted = PredictedResult["predicted_mean"]
    real = RealPrice["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)

def main():

    df = df[df['Stock Name'] == STOCK_NAME]

    sent_df = df.copy()
    sent_df["sentiment_score"] = ''
    sent_df["Negative"] = ''
    sent_df["Neutral"] = ''
    sent_df["Positive"] = ''
    sent_df.head()
    SentimentAnalyser = SentimentIntensityAnalyzer()

    for IDX, ROW in sent_df.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', sent_df.loc[IDX, 'Tweet'])
            sentence_sentiment = SentimentAnalyser.polarity_scores(sentence_i)
            sent_df.at[IDX, 'sentiment_score'] = sentence_sentiment['compound']
            sent_df.at[IDX, 'Negative'] = sentence_sentiment['neg']
            sent_df.at[IDX, 'Neutral'] = sentence_sentiment['neu']
            sent_df.at[IDX, 'Positive'] = sentence_sentiment['pos']
        except TypeError:
            print (sent_df.loc[indexx, 'Tweet'])
            print (IDX)
            break

    AllStocks = pd.read_csv('Data/stock_yfinance_data.csv')
    StockDF = AllStocks[AllStocks['Stock Name'] == STOCK_NAME]
    StockDF['Date'] = pd.to_datetime(StockDF['Date'])
    StockDF['Date'] = StockDF['Date'].dt.date
    FinalDF = StockDF.join(twitter_df, how="left", on="Date")
    FinalDF = FinalDF.drop(columns=['Stock Name'])

    fig, ax = plt.subplots(figsize=(15,8))
    ax.plot(FinalDF['Date'], FinalDF['Close'], color='#008B8B')
    ax.set(xlabel="Date", ylabel="USD", title=f"{STOCK_NAME} Stock Price")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.show()

    TechDF = Get_Tech_Ind(FinalDF)
    dataset = TechDF.iloc[20:,:].reset_index(drop=True)
    Tech_Ind(TechDF)

    dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill()])
    datetime_series = pd.to_datetime(dataset['Date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    dataset = dataset.set_index(datetime_index)
    dataset = dataset.sort_values(by='Date')
    dataset = dataset.drop(columns='Date')

    X_SCALE_DF,Y_SCALE_DF = Normalise(dataset, (-1,1), "Close")
    XBATCHED, YBATCHED, YC = Get_Batch_Data(X_SCALE_DF, Y_SCALE_DF, BATCH_SZ = 5, predict_period = 1)
    X_Train, X_Test, = Split_Train_Test(XBATCHED)
    Y_Train, Y_Test, = Split_Train_Test(YBATCHED)
    YC_Train, YC_test, = Split_Train_Test(YC)
    IDX_Train, Index_Test, = Predict_IDX(dataset, X_Train, 5, 1)

    InpuDim = X_Train.shape[1]
    FeatureSize = X_Train.shape[2]
    OutputDim = Y_Train.shape[1]

    LEARNING_RATE = 5e-4
    EPOCHS = 500

    G_Optimiser = tf.keras.optimizers.Adam(lr = LEARNING_RATE)
    D_Optimiser = tf.keras.optimizers.Adam(lr = LEARNING_RATE)
    Generator = Generator_Model(X_Train.shape[1], OutputDim, X_Train.shape[2])
    Discriminator = Discriminator_Model(X_Train.shape[1])

    plot_model(Generator, to_file='Generator_keras_model.png', show_shapes=True)

    tf.keras.utils.plot_model(Discriminator, to_file='Discriminator_keras_model.png', show_shapes=True)
    PredictedPrice, RealPrice, RMSPE = Train(X_Train, Y_Train, YC_Train, EPOCHS, Generator, Discriminator, G_Optimiser, D_Optimiser)
    TestGenerator = tf.keras.models.load_model(f'./models_gan/{STOCK_NAME}/Generator_V_{EPOCHS-1}.h5')

    PredictedTestData = Evaluate_Op(TestGenerator, X_Test)
    PlotTestData(Y_Test, PredictedTestData,Index_Test)

if __name__ == "__main__":
    main()
