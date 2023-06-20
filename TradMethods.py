import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from itertools import cycl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from numpy import array
from sklearn import neighbors

def CreateDataset(dataset, TimeStep=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-TimeStep-1):
        a = dataset[i:(i+TimeStep), 0] 
        dataX.append(a)
        dataY.append(dataset[i + TimeStep, 0])

    return np.array(dataX), np.array(dataY)

def main():

    df = pd.read_csv("Data/RELIANCE.csv")
    df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
    df.dropna(inplace=True)
    df.isna().any()
    df['date'] = pd.to_datetime(df.date)
    MonthlyData = df.groupby(df['date'].dt.strftime('%B'))[['open','close']].mean().sort_values(by='close')
    MonthlyData.head()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=MonthlyData.index,
        y=MonthlyData['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=MonthlyData.index,
        y=MonthlyData['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45,
                      title='Monthwise comparision between Stock actual, open and close price')
    fig.show()

    df.groupby(df['date'].dt.strftime('%B'))['low'].min()
    MonthlyDataHigh = df.groupby(df['date'].dt.strftime('%B'))['high'].max()
    MonthlyDataLow = df.groupby(df['date'].dt.strftime('%B'))['low'].min()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MonthlyDataHigh.index,
        y=MonthlyDataHigh,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig.add_trace(go.Bar(
        x=MonthlyDataLow.index,
        y=MonthlyDataLow,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig.update_layout(barmode='group',
                      title=' Monthwise High and Low stock price')
    fig.show()

    Names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig = px.line(df, x=df.date, y=[df['open'], df['close'],
                                              df['high'], df['low']],
                 labels={'date': 'Date','value':'Stock value'})
    fig.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()

    ClosedF = df[['date','close']]

    fig = px.line(ClosedF, x=ClosedF.date, y=ClosedF.close,labels={'date':'Date','close':'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.6)
    fig.update_layout(title_text='Stock close price chart', plot_bgcolor='white', font_size=15, font_color='black')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    CloseStock = ClosedF.copy()
    del ClosedF['date']
    scaler=MinMaxScaler(feature_range=(0,1))
    ClosedF=scaler.fit_transform(np.array(ClosedF).reshape(-1,1))
    training_size=int(len(ClosedF)*0.65)
    test_size=len(ClosedF)-training_size
    TrainData,TestData=ClosedF[0:training_size,:],ClosedF[training_size:len(ClosedF),:1]

    TimeStep = 15
    XTrain, YTrain = CreateDataset(TrainData, TimeStep)
    XTest, YTest = CreateDataset(TestData, TimeStep)

    SVR_RBF = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
    SVR_RBF.fit(XTrain, YTrain)

    TrainPredict = SVR_RBF.predict(XTrain)
    TestPredict = SVR_RBF.predict(XTest)

    TrainPredict = TrainPredict.reshape(-1,1)
    TestPredict = TestPredict.reshape(-1,1)

    TrainPredict = scaler.inverse_transform(TrainPredict)
    TestPredict = scaler.inverse_transform(TestPredict)
    OrigYTrain = scaler.inverse_transform(YTrain.reshape(-1,1))
    OrigYTest = scaler.inverse_transform(y_test.reshape(-1,1))

    LookBack = TimeStep
    TrainPredictPlot = np.empty_like(ClosedF)
    TrainPredictPlot[:, :] = np.nan
    TrainPredictPlot[LookBack:len(TrainPredict)+LookBack, :] = TrainPredict
    print("Train predicted data: ", TrainPredictPlot.shape)

    TestPredictPlot = np.empty_like(ClosedF)
    TestPredictPlot[:, :] = np.nan
    TestPredictPlot[len(TrainPredict)+(LookBack*2)+1:len(ClosedF)-1, :] = TestPredict
    print("Test predicted data: ", TestPredictPlot.shape)

    Names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

    PlotDF = pd.DataFrame({'date': CloseStock['date'],
                           'original_close': CloseStock['close'],
                          'TrainPredicted_close': TrainPredictPlot.reshape(1,-1)[0].tolist(),
                          'TestPredicted_close': TestPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(PlotDF,x=PlotDF['date'], y=[PlotDF['original_close'],PlotDF['TrainPredicted_close'],
                                              PlotDF['TestPredicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XInput = TestData[len(TestData)-TimeStep:].reshape(1,-1)
    TempInput = list(XInput)
    TempInput = TempInput[0].tolist()

    LSTOutput = []
    NSteps = TimeStep
    i = 0
    PredDays = 10

    while(i < PredDays):

        if(len(TempInput)>TimeStep):

            XInput = np.array(TempInput[1:])
            XInput = XInput.reshape(1,-1)

            YHat = SVR_RBF.predict(XInput)
            TempInput.extend(YHat.tolist())
            TempInput = TempInput[1:]

            LSTOutput.extend(YHat.tolist())
            i = i + 1

        else:
            YHat = SVR_RBF.predict(XInput)

            TempInput.extend(YHat.tolist())
            LSTOutput.extend(YHat.tolist())

            i = i + 1

    print("Output of predicted next days: ", len(LSTOutput))

    LastDays = np.arange(1,TimeStep+1)
    DayPred = np.arange(TimeStep+1,TimeStep+PredDays+1)
    TempMat = np.empty((len(LastDays)+PredDays+1,1))
    TempMat[:] = np.nan
    TempMat = TempMat.reshape(1,-1).tolist()[0]
    LastOriginalDaysValue = TempMat
    NextPredictedDaysValue = TempMat
    LastOriginalDaysValue[0:TimeStep+1] = scaler.inverse_transform(ClosedF[len(ClosedF)-TimeStep:]).reshape(1,-1).tolist()[0]
    NextPredictedDaysValue[TimeStep+1:] = scaler.inverse_transform(np.array(LSTOutput).reshape(-1,1)).reshape(1,-1).tolist()[0]

    NewPredPlot = pd.DataFrame({
        'LastOriginalDaysValue':LastOriginalDaysValue,
        'NextPredictedDaysValue':NextPredictedDaysValue
    })

    Names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

    fig = px.line(NewPredPlot,x=NewPredPlot.index, y=[NewPredPlot['LastOriginalDaysValue'],
                                                          NewPredPlot['NextPredictedDaysValue']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    SVRDF = ClosedF.tolist()
    SVRDF.extend((np.array(LSTOutput).reshape(-1,1)).tolist())
    SVRDF = scaler.inverse_transform(SVRDF).reshape(1,-1).tolist()[0]
    Names = cycle(['Close Price'])
    fig = px.line(SVRDF,labels={'value': 'Stock Price','index': 'Timestamp'})
    fig.update_layout(title_text='Closing Stock Price Prediction (SVR)',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    Regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    Regressor.fit(XTrain, YTrain)
    TrainPredict=Regressor.predict(XTrain)
    TestPredict=Regressor.predict(XTest)

    TrainPredict = TrainPredict.reshape(-1,1)
    TestPredict = TestPredict.reshape(-1,1)

    TrainPredict = scaler.inverse_transform(TrainPredict)
    TestPredict = scaler.inverse_transform(TestPredict)
    OrigYTrain = scaler.inverse_transform(YTrain.reshape(-1,1))
    OrigYTest = scaler.inverse_transform(y_test.reshape(-1,1))

    LookBack=TimeStep
    TrainPredictPlot = np.empty_like(ClosedF)
    TrainPredictPlot[:, :] = np.nan
    TrainPredictPlot[LookBack:len(TrainPredict)+LookBack, :] = TrainPredict
    print("Train predicted data: ", TrainPredictPlot.shape)

    TestPredictPlot = np.empty_like(ClosedF)
    TestPredictPlot[:, :] = np.nan
    TestPredictPlot[len(TrainPredict)+(LookBack*2)+1:len(ClosedF)-1, :] = TestPredict
    print("Test predicted data: ", TestPredictPlot.shape)

    Names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

    PlotDF = pd.DataFrame({'date': CloseStock['date'],
                           'original_close': CloseStock['close'],
                          'TrainPredicted_close': TrainPredictPlot.reshape(1,-1)[0].tolist(),
                          'TestPredicted_close': TestPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(PlotDF,x=PlotDF['date'], y=[PlotDF['original_close'],PlotDF['TrainPredicted_close'],
                                              PlotDF['TestPredicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XInput = TestData[len(TestData)-TimeStep:].reshape(1,-1)
    TempInput = list(XInput)
    TempInput = TempInput[0].tolist()

    LSTOutput = []
    NSteps = TimeStep
    i = 0
    PredDays = 10
    while (i < PredDays):

        if(len(TempInput) > TimeStep):

            XInput = np.array(TempInput[1:])
            XInput=XInput.reshape(1,-1)
            YHat = Regressor.predict(XInput)
            TempInput.extend(YHat.tolist())
            TempInput=TempInput[1:]
            LSTOutput.extend(YHat.tolist())
            i = i + 1

        else:
            YHat = Regressor.predict(XInput)
            TempInput.extend(YHat.tolist())
            LSTOutput.extend(YHat.tolist())
            i = i + 1

    print("Output of predicted next days: ", len(LSTOutput))

    LastDays = np.arange(1,TimeStep + 1)
    DayPred = np.arange(TimeStep + 1, TimeStep + PredDays + 1)
    TempMat = np.empty((len(LastDays)+PredDays+1,1))
    TempMat[:] = np.nan
    TempMat = TempMat.reshape(1,-1).tolist()[0]

    LastOriginalDaysValue = TempMat
    NextPredictedDaysValue = TempMat

    LastOriginalDaysValue[0:TimeStep+1] = scaler.inverse_transform(ClosedF[len(ClosedF)-TimeStep:]).reshape(1,-1).tolist()[0]
    NextPredictedDaysValue[TimeStep+1:] = scaler.inverse_transform(np.array(LSTOutput).reshape(-1,1)).reshape(1,-1).tolist()[0]

    Names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

    NewPredPlot = pd.DataFrame({
        'LastOriginalDaysValue':LastOriginalDaysValue,
        'NextPredictedDaysValue':NextPredictedDaysValue
    })

    fig = px.line(NewPredPlot,x=NewPredPlot.index, y=[NewPredPlot['LastOriginalDaysValue'],
                                                          NewPredPlot['NextPredictedDaysValue']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    DFDF = ClosedF.tolist()
    DFDF.extend((np.array(LSTOutput).reshape(-1,1)).tolist())
    DFDF = scaler.inverse_transform(DFDF).reshape(1,-1).tolist()[0]

    Names = cycle(['Close price'])

    fig = px.line(DFDF,labels={'value': 'Stock Price','index': 'Timestamp'})
    fig.update_layout(title_text='Closing Stock Price Prediction (RF)',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()
    K = TimeStep
    neighbor = neighbors.KNeighborsRegressor(n_neighbors = K)
    neighbor.fit(XTrain, YTrain)
    TrainPredict=neighbor.predict(XTrain)
    TestPredict=neighbor.predict(XTest)

    TrainPredict = TrainPredict.reshape(-1,1)
    TestPredict = TestPredict.reshape(-1,1)

    TrainPredict = scaler.inverse_transform(TrainPredict)
    TestPredict = scaler.inverse_transform(TestPredict)
    OrigYTrain = scaler.inverse_transform(YTrain.reshape(-1,1))
    OrigYTest = scaler.inverse_transform(y_test.reshape(-1,1))

    LookBack=TimeStep
    TrainPredictPlot = np.empty_like(ClosedF)
    TrainPredictPlot[:, :] = np.nan
    TrainPredictPlot[LookBack:len(TrainPredict)+LookBack, :] = TrainPredict
    print("Train predicted data: ", TrainPredictPlot.shape)

    TestPredictPlot = np.empty_like(ClosedF)
    TestPredictPlot[:, :] = np.nan
    TestPredictPlot[len(TrainPredict)+(LookBack*2)+1:len(ClosedF)-1, :] = TestPredict
    print("Test predicted data: ", TestPredictPlot.shape)

    Names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

    PlotDF = pd.DataFrame({'date': CloseStock['date'],
                           'original_close': CloseStock['close'],
                          'TrainPredicted_close': TrainPredictPlot.reshape(1,-1)[0].tolist(),
                          'TestPredicted_close': TestPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(PlotDF,x=PlotDF['date'], y=[PlotDF['original_close'],PlotDF['TrainPredicted_close'],
                                              PlotDF['TestPredicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XInput = TestData[len(TestData)-TimeStep:].reshape(1,-1)
    TempInput = list(XInput)
    TempInput = TempInput[0].tolist()
    LSTOutput = []
    NSteps = TimeStep
    i = 0
    PredDays = 10
    while( i < PredDays):

        if(len(TempInput)>TimeStep):

            XInput = np.array(TempInput[1:])
            XInput = XInput.reshape(1,-1)

            YHat = neighbor.predict(XInput)
            TempInput.extend(YHat.tolist())
            TempInput = TempInput[1:]

            LSTOutput.extend(YHat.tolist())
            i = i+1

        else:
            YHat = neighbor.predict(XInput)

            TempInput.extend(YHat.tolist())
            LSTOutput.extend(YHat.tolist())

            i = i+1

    print("Output of predicted next days: ", len(LSTOutput))

    LastDays = np.arange(1,TimeStep+1)
    DayPred = np.arange(TimeStep+1,TimeStep+PredDays+1)
    TempMat = np.empty((len(LastDays)+PredDays+1,1))
    TempMat[:] = np.nan
    TempMat = TempMat.reshape(1,-1).tolist()[0]

    LastOriginalDaysValue = TempMat
    NextPredictedDaysValue = TempMat

    LastOriginalDaysValue[0:TimeStep+1] = scaler.inverse_transform(ClosedF[len(ClosedF)-TimeStep:]).reshape(1,-1).tolist()[0]
    NextPredictedDaysValue[TimeStep+1:] = scaler.inverse_transform(np.array(LSTOutput).reshape(-1,1)).reshape(1,-1).tolist()[0]

    NewPredPlot = pd.DataFrame({
        'LastOriginalDaysValue':LastOriginalDaysValue,
        'NextPredictedDaysValue':NextPredictedDaysValue
    })

    Names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

    fig = px.line(NewPredPlot,x=NewPredPlot.index, y=[NewPredPlot['LastOriginalDaysValue'],
                                                          NewPredPlot['NextPredictedDaysValue']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()
    knndf=ClosedF.tolist()
    knndf.extend((np.array(LSTOutput).reshape(-1,1)).tolist())
    knndf=scaler.inverse_transform(knndf).reshape(1,-1).tolist()[0]

    Names = cycle(['Close price'])

    fig = px.line(knndf,labels={'value': 'Stock Price','index': 'Timestamp'})
    fig.update_layout(title_text='Closing Stock Price Prediction (KNN)',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XTrain =XTrain.reshape(XTrain.shape[0],XTrain.shape[1] , 1)
    XTest = XTest.reshape(XTest.shape[0],XTest.shape[1] , 1)

    print("XTrain: ", XTrain.shape)
    print("XTest: ", XTest.shape)

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(TimeStep,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(XTrain,YTrain,validation_data=(XTest,y_test),epochs=200,batch_size=5,verbose=1)

    TrainPredict=model.predict(XTrain)
    TestPredict=model.predict(XTest)
    TrainPredict.shape, TestPredict.shape

    TrainPredict = scaler.inverse_transform(TrainPredict)
    TestPredict = scaler.inverse_transform(TestPredict)
    OrigYTrain = scaler.inverse_transform(YTrain.reshape(-1,1))
    OrigYTest = scaler.inverse_transform(y_test.reshape(-1,1))
    LookBack=TimeStep
    TrainPredictPlot = np.empty_like(ClosedF)
    TrainPredictPlot[:, :] = np.nan
    TrainPredictPlot[LookBack:len(TrainPredict)+LookBack, :] = TrainPredict
    print("Train predicted data: ", TrainPredictPlot.shape)

    TestPredictPlot = np.empty_like(ClosedF)
    TestPredictPlot[:, :] = np.nan
    TestPredictPlot[len(TrainPredict)+(LookBack*2)+1:len(ClosedF)-1, :] = TestPredict
    print("Test predicted data: ", TestPredictPlot.shape)

    Names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    PlotDF = pd.DataFrame({'date': CloseStock['date'],
                           'original_close': CloseStock['close'],
                          'TrainPredicted_close': TrainPredictPlot.reshape(1,-1)[0].tolist(),
                          'TestPredicted_close': TestPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(PlotDF,x=PlotDF['date'], y=[PlotDF['original_close'],PlotDF['TrainPredicted_close'],
                                              PlotDF['TestPredicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XInput = TestData[len(TestData)-TimeStep:].reshape(1,-1)
    TempInput = list(XInput)
    TempInput = TempInput[0].tolist()
    LSTOutput = []
    NSteps = TimeStep
    i = 0
    PredDays = 10
    while(i < PredDays):

        if(len(TempInput) > TimeStep):

            XInput = np.array(TempInput[1:])
            XInput = XInput.reshape(1, -1)
            XInput = XInput.reshape((1, NSteps, 1))

            YHat = model.predict(XInput, verbose=0)
            TempInput.extend(YHat[0].tolist())
            TempInput = TempInput[1:]

            LSTOutput.extend(YHat.tolist())
            i = i + 1

        else:

            XInput = XInput.reshape((1, NSteps,1))
            YHat = model.predict(XInput, verbose=0)
            TempInput.extend(YHat[0].tolist())
            LSTOutput.extend(YHat.tolist())
            i = i + 1

    print("Output of predicted next days: ", len(LSTOutput))

    LastDays=np.arange(1, TimeStep + 1)
    DayPred=np.arange(TimeStep + 1, TimeStep + PredDays + 1)
    TempMat = np.empty((len(LastDays) + PredDays + 1, 1))
    TempMat[:] = np.nan
    TempMat = TempMat.reshape(1, -1).tolist()[0]

    LastOriginalDaysValue = TempMat
    NextPredictedDaysValue = TempMat

    LastOriginalDaysValue[0:TimeStep+1] = scaler.inverse_transform(ClosedF[len(ClosedF)-TimeStep:]).reshape(1, -1).tolist()[0]
    NextPredictedDaysValue[TimeStep+1:] = scaler.inverse_transform(np.array(LSTOutput).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    NewPredPlot = pd.DataFrame({
        'LastOriginalDaysValue':LastOriginalDaysValue,
        'NextPredictedDaysValue':NextPredictedDaysValue
    })

    Names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

    fig = px.line(NewPredPlot,x=NewPredPlot.index, y=[NewPredPlot['LastOriginalDaysValue'],
                                                          NewPredPlot['NextPredictedDaysValue']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(Names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    LSTMDF = ClosedF.tolist()
    LSTMDF.extend((np.array(LSTOutput).reshape(-1,1)).tolist())
    LSTMDF = scaler.inverse_transform(LSTMDF).reshape(1,-1).tolist()[0]

    Names = cycle(['Close price'])

    fig = px.line(LSTMDF,labels={'value': 'Stock Price','index': 'Timestamp'})
    fig.update_layout(title_text='Closing Stock Price Prediction (LSTM)',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XTrain = XTrain.reshape(XTrain.shape[0],XTrain.shape[1] , 1)
    XTest = XTest.reshape(XTest.shape[0],XTest.shape[1] , 1)

    print("XTrain: ", XTrain.shape)
    print("XTest: ", XTest.shape)

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(GRU(32,return_sequences=True,input_shape=(TimeStep,1)))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(XTrain,YTrain,validation_data=(XTest,y_test),epochs=200,batch_size=5,verbose=1)

    TrainPredict = model.predict(XTrain)
    TestPredict = model.predict(XTest)
    TrainPredict.shape, TestPredict.shape
    TrainPredict = scaler.inverse_transform(TrainPredict)
    TestPredict = scaler.inverse_transform(TestPredict)
    OrigYTrain = scaler.inverse_transform(YTrain.reshape(-1, 1))
    OrigYTest = scaler.inverse_transform(y_test.reshape(-1, 1))

    LookBack = TimeStep
    TrainPredictPlot = np.empty_like(ClosedF)
    TrainPredictPlot[:, :] = np.nan
    TrainPredictPlot[LookBack:len(TrainPredict)+LookBack, :] = TrainPredict
    print("Train predicted data: ", TrainPredictPlot.shape)

    TestPredictPlot = np.empty_like(ClosedF)
    TestPredictPlot[:, :] = np.nan
    TestPredictPlot[len(TrainPredict)+(LookBack*2)+1:len(ClosedF)-1, :] = TestPredict
    print("Test predicted data: ", TestPredictPlot.shape)

    Names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

    PlotDF = pd.DataFrame({'date': CloseStock['date'],
                           'original_close': CloseStock['close'],
                          'TrainPredicted_close': TrainPredictPlot.reshape(1,-1)[0].tolist(),
                          'TestPredicted_close': TestPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(PlotDF,x=PlotDF['date'], y=[PlotDF['original_close'],PlotDF['TrainPredicted_close'],
                                              PlotDF['TestPredicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    XInput = TestData[len(TestData)-TimeStep:].reshape(1,-1)
    TempInput = list(XInput)
    TempInput = TempInput[0].tolist()

    LSTOutput = []
    NSteps = TimeStep
    i = 0
    PredDays = 10
    while(i < PredDays):

        if(len(TempInput) > TimeStep):

            XInput=np.array(TempInput[1:])
            XInput = XInput.reshape(1,-1)
            XInput = XInput.reshape((1, NSteps, 1))
            YHat = model.predict(XInput, verbose=0)
            TempInput.extend(YHat[0].tolist())
            TempInput=TempInput[1:]
            LSTOutput.extend(YHat.tolist())
            i = i + 1

        else:

            XInput = XInput.reshape((1, NSteps,1))
            YHat = model.predict(XInput, verbose=0)
            TempInput.extend(YHat[0].tolist())
            LSTOutput.extend(YHat.tolist())
            i = i + 1

    print("Output of predicted next days: ", len(LSTOutput))

    LastDays = np.arange(1, TimeStep + 1)
    DayPred = np.arange(TimeStep + 1, TimeStep + PredDays + 1)
    TempMat = np.empty((len(LastDays) + PredDays + 1, 1))
    TempMat[:] = np.nan
    TempMat = TempMat.reshape(1, -1).tolist()[0]
    LastOriginalDaysValue = TempMat
    NextPredictedDaysValue = TempMat
    LastOriginalDaysValue[0:TimeStep + 1] = scaler.inverse_transform(ClosedF[len(ClosedF)-TimeStep:]).reshape(1, -1).tolist()[0]
    NextPredictedDaysValue[TimeStep + 1:] = scaler.inverse_transform(np.array(LSTOutput).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    NewPredPlot = pd.DataFrame({
        'LastOriginalDaysValue':LastOriginalDaysValue,
        'NextPredictedDaysValue':NextPredictedDaysValue
    })
    Names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

    fig = px.line(NewPredPlot,x=NewPredPlot.index, y=[NewPredPlot['LastOriginalDaysValue'],
                                                          NewPredPlot['NextPredictedDaysValue']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    GRUDF = ClosedF.tolist()
    GRUDF.extend((np.array(LSTOutput).reshape(-1,1)).tolist())
    GRUDF=scaler.inverse_transform(GRUDF).reshape(1,-1).tolist()[0]

    Names = cycle(['Close price'])
    fig = px.line(GRUDF,labels={'value': 'Stock Price','index': 'Timestamp'})
    fig.update_layout(title_text='Closing Stock Price Prediction (GRU)',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()
    XTrain = XTrain.reshape(XTrain.shape[0],XTrain.shape[1] , 1)
    XTest = XTest.reshape(XTest.shape[0],XTest.shape[1] , 1)

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(TimeStep,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(XTrain,YTrain,validation_data=(XTest,y_test),epochs=200,batch_size=5,verbose=1)

    TrainPredict = model.predict(XTrain)
    TestPredict = model.predict(XTest)
    TrainPredict.shape, TestPredict.shape

    TrainPredict = scaler.inverse_transform(TrainPredict)
    TestPredict = scaler.inverse_transform(TestPredict)
    OrigYTrain = scaler.inverse_transform(YTrain.reshape(-1,1))
    OrigYTest = scaler.inverse_transform(y_test.reshape(-1,1))

    XInput = TestData[len(TestData)-TimeStep:].reshape(1,-1)
    TempInput = list(XInput)
    TempInput = TempInput[0].tolist()
    LSTOutput = []
    NSteps = TimeStep
    i = 0
    PredDays = 10
    while(i < PredDays):

        if(len(TempInput) > TimeStep):

            XInput=np.array(TempInput[1:])
            XInput = XInput.reshape(1,-1)
            XInput = XInput.reshape((1, NSteps, 1))
            YHat = model.predict(XInput, verbose=0)
            TempInput.extend(YHat[0].tolist())
            TempInput=TempInput[1:]
            LSTOutput.extend(YHat.tolist())
            i = i + 1

        else:
            XInput = XInput.reshape((1, NSteps,1))
            YHat = model.predict(XInput, verbose=0)
            TempInput.extend(YHat[0].tolist())
            LSTOutput.extend(YHat.tolist())
            i=i+1

    print("Output of predicted next days: ", len(LSTOutput))

    LastDays = np.arange(1, TimeStep + 1)
    DayPred = np.arange(TimeStep + 1, TimeStep + PredDays + 1)
    TempMat = np.empty((len(LastDays) + PredDays + 1, 1))
    TempMat[:] = np.nan
    TempMat = TempMat.reshape(1, -1).tolist()[0]
    LastOriginalDaysValue = TempMat
    NextPredictedDaysValue = TempMat
    LastOriginalDaysValue[0:TimeStep + 1] = scaler.inverse_transform(ClosedF[len(ClosedF)-TimeStep:]).reshape(1, -1).tolist()[0]
    NextPredictedDaysValue[TimeStep + 1:] = scaler.inverse_transform(np.array(LSTOutput).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    NewPredPlot = pd.DataFrame({
        'LastOriginalDaysValue':LastOriginalDaysValue,
        'NextPredictedDaysValue':NextPredictedDaysValue
    })
    Names = cycle(['Last 15 days close price','Predicted next 10 days close price'])

    fig = px.line(NewPredPlot,x=NewPredPlot.index, y=[NewPredPlot['LastOriginalDaysValue'],
                                                          NewPredPlot['NextPredictedDaysValue']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    lstmGRUDF=ClosedF.tolist()
    lstmGRUDF.extend((np.array(LSTOutput).reshape(-1,1)).tolist())
    lstmGRUDF=scaler.inverse_transform(lstmGRUDF).reshape(1,-1).tolist()[0]

    Names = cycle(['Close Price'])

    fig = px.line(lstmGRUDF,labels={'value': 'Stock Price','index': 'Timestamp'})
    fig.update_layout(title_text='Closing Stock Price Prediction (LSTM + GRU)',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

    FinalDataframe = pd.DataFrame({
        'svr':SVRDF,
        'rf':DFDF,
        'knn':knndf,
        'lstm':LSTMDF,
        'gru':GRUDF,
        'lstm_gru':lstmGRUDF,
    })
    FinalDataframe.head()

    Names = cycle(['SVR', 'RF','KNN','LSTM','GRU','LSTM + GRU'])

    fig = px.line(FinalDataframe[225:], x=FinalDataframe.index[225:], y=[FinalDataframe['svr'][225:],FinalDataframe['rf'][225:], FinalDataframe['knn'][225:],
                                              FinalDataframe['lstm'][225:], FinalDataframe['gru'][225:], FinalDataframe['lstm_gru'][225:]],
                 labels={'x': 'Timestamp','value':'Stock Close Price'})
    fig.update_layout(title_text='Stock Comparison: All Methods', font_size=15, font_color='black',legend_title_text='Algorithms')
    fig.for_each_trace(lambda t:  t.update(name = next(Names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()

if __name__ == "__main__":
    main()