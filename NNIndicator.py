import numpy as np
import pandas as pd
import sys
from pandas._libs.tslibs.offsets import BDay
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import plotly
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

from statsmodels.tsa.stattools import adfuller,kpss

TrainingAccuracy = []
ValidationAccuracy = []
ValidationSimulation = []

#def SimulationsTest()

def train(model, device, train_loader, optimizer, epoch, batchSize, test_loader, ValidationPrice):
    model.train() # When you call train(), tensor function
    total_loss = 0
    total_classification = 0
    total_correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Sets gradients of all model parameters to zero
        output = model(data) #insert the actual data Tensor[Batch][Channels In Each Batch][Height of Matrix][Width of Matrix]
        loss = torch.nn.CrossEntropyLoss()
        #output = output[:, -1] # THis is when we are using convolutional nets etc
        #output = output.double()
        loss = loss(output, target.long())
        loss.backward()
        optimizer.step()
        classification = output.detach().numpy()
        realTarget = target.numpy()

        for i in range (0, len(realTarget)):
            x = np.argmax(classification[i])
            if x == int(realTarget[i]):
                #print("correct")
                total_correct += 1

        total_loss += loss.item()
        total_classification += target.size(0)
        #if batch_idx % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))
    model_accuracy = total_correct / total_classification * 100
    TrainingAccuracy.append(model_accuracy)
    ValidationResults = Validation(model, device, test_loader, ValidationPrice)
    print('epoch {0} total_correct: {1} loss: {2:.2f} acc: {3:.2f} '.format(
        epoch, total_correct, total_loss, model_accuracy))
    print(f"epoch {epoch} Validation Accuracy = {ValidationResults[0]}, Validation Simulation Predicted = {ValidationResults[1]}, Validation Simulations Real = {ValidationResults[2]}")

def Validation(model, device, test_loader, ValidationPrice):
    model.eval()
    total_correct = 0
    total_classification = 0
    Signals = []
    BestSignals = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #output = output[:,-1]
            classification = output.detach().numpy()
            realTarget = target.numpy()
            Signals.append(np.argmax(classification[0]))
            BestSignals.append(int(realTarget[0]))
            if np.argmax(classification[0]) == int(realTarget[0]):
                total_correct += 1
            total_classification += target.size(0)

    ValidationReturns = TestSimulation(ValidationPrice, Signals)
    SubOptimalReturns = TestSimulation(ValidationPrice, BestSignals)
    ValidationAccuracy.append((total_correct/total_classification)*100)
    ValidationSimulation.append(ValidationReturns)
    return (total_correct / total_classification) * 100, ValidationReturns, SubOptimalReturns

class Network2(torch.nn.Module):
    def __init__(self, window, n):
        super(Network2, self).__init__()
        window *= n
        self.HiddenLayer1 = nn.Linear(window, window)
        self.HiddenLayer2 = nn.Linear(window, int(window/2))
        self.OutputLayer = nn.Linear(int(window/2), 3)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, input):
        #Should
        self.L1 = torch.relu(self.HiddenLayer1(input))
        self.L2 = torch.relu(self.HiddenLayer2(self.L1))
        self.result = self.OutputLayer(self.L2)
        self.Output = self.out(self.result)
        return self.Output

#More Hidden layers MORE complexity
class Network(torch.nn.Module):
    def __init__(self, window, n):
        super(Network, self).__init__()
        window *= n
        self.HiddenLayer1 = nn.Linear(window, window)
        self.HiddenLayer2 = nn.Linear(window, int(window/2))
        self.HiddenLayer3 = nn.Linear(int(window/2), int(window/4))
        self.OutputLayer = nn.Linear(int(window/4), 3)
        self.out = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.dropout(self.HiddenLayer1(x))
        self.L1 = torch.relu(x)
        self.L2 = torch.relu(self.HiddenLayer2(self.L1))
        x = self.dropout(self.L2)
        self.L3 = torch.relu(self.HiddenLayer3(x))
        self.result = self.OutputLayer(self.L3)
        self.Output = self.out(self.result)
        return self.Output

class ConvNetwork(torch.nn.Module):
    def __init__(self, window):
        super(ConvNetwork, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,1))
        self.HiddenLayer1 = nn.Linear(3*window, window)
        self.HiddenLayer2 = nn.Linear(window, int(window/2))
        self.HiddenLayer3 = nn.Linear(int(window/2), int(window/4))
        self.OutputLayer = nn.Linear(int(window/4), 3)
        self.out = nn.Softmax()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #print(f"0 x.shape = {x.shape}")
        x = self.C1(x)
        #print(f"1 x.shape = {x.shape}")
        x = x.reshape(x.shape[0], -1)
        #print(f"2 x.shape = {x.shape}")
        x = self.HiddenLayer1(x)
        #print(f"3 x.shape = {x.shape}")
        self.L1 = torch.relu(x)
        #print(f"4 x.shape = {self.L1.shape}")
        self.L2 = torch.relu(self.HiddenLayer2(self.L1))
        x = self.L2
        #print(f"5 x.shape = {x.shape}")
        self.L3 = torch.relu(self.HiddenLayer3(x))
        #print(f"6 x.shape = {self.L3.shape}")
        self.result = self.OutputLayer(self.L3)
        #print(f"7 x.shape = {self.result.shape}")
        self.Output = self.out(self.result)
        #print(f"8 x.shape = {self.Output.shape}")
        return self.Output

class PatternNetwork(torch.nn.Module):
    def __init__(self, window):
        super(PatternNetwork, self).__init__()
        k = window
        #Inputs will be k * 3
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3,1))
        # Now we have (k, 1, 6)
        #self.C2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,3))
        # Now we have (k-2, 1, 3)
        # Now we multiply k-2 * 1 * 3 = 3*k - 6
        self.LL1 = nn.Linear(6*k, 3*k)
        self.LL2 = nn.Linear(3*k, k)
        self.LL3 = nn.Linear(k, 3)
        self.OutputLayer = nn.Linear(3, 3)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print(f"shape is : {x.shape}")
        self.L1 = torch.relu(self.C1(x))
        self.L1 = self.L1.reshape(self.L1.shape[0], -1)
        self.L2 = torch.relu(self.LL1(self.L1))
        self.L3 = torch.relu(self.LL2(self.L2))
        self.L4 = torch.relu(self.LL3(self.L3))
        self.result = self.OutputLayer(self.L4)
        #self.output = self.out(self.result)
        return self.result

class WeirdNetwork(torch.nn.Module):
    def __init__(self, window, n):
        super(WeirdNetwork, self).__init__()
        window *= n
        self.HiddenLayer1 = nn.Linear(window, 48*window)
        self.HiddenLayer2 = nn.Linear(48*window, 48*window)
        self.HiddenLayer3 = nn.Linear(48*window, 96*window)
        self.HiddenLayer4 = nn.Linear(96*window, 48*window)
        self.HiddenLayer5 = nn.Linear(48*window, 16*window)
        self.HiddenLayer6 = nn.Linear(16*window, 8*window)
        self.HiddenLayer7 = nn.Linear(8*window, 4*window)
        self.HiddenLayer8 = nn.Linear(4*window, 2*window)
        self.HiddenLayer9 = nn.Linear(2*window, window)
        self.HiddenLayer10 = nn.Linear(window, int(window/2))
        self.OutputLayer = nn.Linear(int(window/2), 3)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, input):
        #Should
        self.L1 = torch.relu(self.HiddenLayer1(input))
        self.L2 = torch.relu(self.HiddenLayer2(self.L1))
        self.L3 = torch.relu(self.HiddenLayer3(self.L2))
        self.L4 = torch.relu(self.HiddenLayer4(self.L3))
        self.L5 = torch.relu(self.HiddenLayer5(self.L4))
        self.L6 = torch.relu(self.HiddenLayer6(self.L5))
        self.L7 = torch.relu(self.HiddenLayer7(self.L6))
        self.L8 = torch.relu(self.HiddenLayer8(self.L7))
        self.L9 = torch.relu(self.HiddenLayer9(self.L8))
        self.L10 = torch.relu(self.HiddenLayer10(self.L9))
        self.result = self.OutputLayer(self.L10)
        self.Output = self.out(self.result)
        return self.Output

class LargeNetwork(torch.nn.Module):
    def __init__(self, window, n):
        super(LargeNetwork, self).__init__()
        window *= n
        self.HiddenLayer1 = nn.Linear(window, 4*window)
        self.HiddenLayer2 = nn.Linear(4*window, 8*window)
        self.HiddenLayer3 = nn.Linear(8*window, 4*window)
        self.HiddenLayer4 = nn.Linear(4*window, int(window/2))
        self.HiddenLayer5 = nn.Linear(int(window/2), int(window/2))
        self.HiddenLayer6 = nn.Linear(int(window/2), int(window/2))
        self.HiddenLayer7 = nn.Linear(int(window/2), int(window/4))
        self.HiddenLayer8 = nn.Linear(int(window/4), int(window/4))
        self.HiddenLayer9 = nn.Linear(int(window/4), int(window/4))
        self.OutputLayer = nn.Linear(int(window/4), 3)
        self.out = nn.Softmax()

    def forward(self, input):
        self.L1 = torch.sigmoid(self.HiddenLayer1(input))
        self.L2 = torch.sigmoid(self.HiddenLayer2(self.L1))
        self.L3 = torch.sigmoid(self.HiddenLayer3(self.L2))
        self.L4 = torch.sigmoid(self.HiddenLayer4(self.L3))
        self.L5 = torch.sigmoid(self.HiddenLayer5(self.L4))
        self.L6 = torch.sigmoid(self.HiddenLayer6(self.L5))
        self.L7 = torch.sigmoid(self.HiddenLayer7(self.L6))
        self.L8 = torch.sigmoid(self.HiddenLayer8(self.L7))
        self.L9 = torch.sigmoid(self.HiddenLayer9(self.L8))
        self.result = self.OutputLayer(self.L9)
        self.Output = self.out(self.result)
        self.Output = self.Output[:,-1]

        return self.Output


def TestSimulation(Prices, Indicator):
    buySignal, sellSignal, transactionFee, initial = 1,0,0.005,10000
    amount = 10000
    shares = 0
    index = 0
    size = len(Prices)
    prevAmount = 0

    # 0 us hold
    # 1 is buy
    # 2 is sell

    while index < size:
        x = Indicator[index]
        if x == 0:
            index += 1
            continue
        elif x == 1:
            if buySignal:
                shares = (amount - amount*transactionFee)/Prices[index]
                #print(
                #    f"Shares bought = {shares}, at price = {Prices[index]}, money spent = {amount}, actual money spent = {amount - (amount * transactionFee)}")
                #print(f"Bought at {index}")
                prevAmount = amount
                amount = 0
                buySignal = 0
                sellSignal = 1
        elif x == 2:
            if sellSignal:
                amount = shares*Prices[index]
                amount = (amount - amount*transactionFee)
                #print(
                #    f"Shares sold = {shares} at price = {Prices[index]}, money recieved = {amount}, actual money recieved = {amount - amount * transactionFee}")
                #print(f"Sold at {index}")
                shares = 0
                sellSignal = 0
                buySignal = 1
        index += 1

    if sellSignal:
        return prevAmount/initial
    else:
        return amount/initial

#We build the signals on a daily basis.
def IterativeOptimalSignalling(price, partitioning):
    #partitioning is assumed to be 24.
    size = len(price) - partitioning
    Signals = []
    #for i in range (0, size):
    i = 0
    end = partitioning
    while end <= len(price):
        print(f"i = {i} -- end = {end}")
        x = OptimisedfindMaxProfitK(price[i:end], 12)
        Signals.extend(x[3]) #Lets make a maximum of 3 trades per day
        i += partitioning
        end += partitioning
    return Signals

def OptimisedfindMaxProfitK(price, k):
    # get the number of days `n`
    n = len(price)
    # base case
    if n <= 1:
        return 0
    # profit[i][j] stores the maximum profit gained by doing
    # at most `i` transactions till j'th day
    #Signals = [[[[],[]] for x in range(n)] for y in range(k + 1)]
    Signals2 = [[{'From' : [], 'New' : []} for x in range(n)] for y in range(k + 1)]
    profit = [[0 for x in range(n + 1)] for y in range(k + 1)]
    # fill profit[][] in a bottom-up fashion

    for i in range(0, k + 1):
        # initialize `prev` diff to `-INFINITY`
        prev_diff = -sys.maxsize
        prev_xindex = 0
        prev_yindex = 0

        for j in range(0, n):
            # profit is 0 when
            # i = 0, i.e., for 0th day
            # j = 0, i.e., no transaction is being performed
            if i == 0 or j == 0:
                profit[i][j] = 0
            else:
                #prev_diff = max(prev_diff, profit[i - 1][j - 1] - price[j - 1])
                #print(f"Iteration = Transaction Number = {i}, Day Number = {j}")
                if prev_diff < profit[i-1][j-1] - price[j-1]: #Finding the max of - price[x] + profit[i - 1][x]
                    prev_diff = profit[i-1][j-1] - price[j-1] #Using previous iterations
                    prev_xindex = i-1
                    prev_yindex = j-1
                    index_buy = j-1

                if profit[i][j-1] >= price[j] + prev_diff: #This is the same as price[j] - price[x] + profit[i-1][x]
                    profit[i][j] = profit[i][j-1]
                    Signals2[i][j]['From'] = [i, j-1]
                    Signals2[i][j]['New'] = []
                else:
                    profit[i][j] = price[j] + prev_diff
                    Signals2[i][j]['From'] = [prev_xindex, prev_yindex]
                    Signals2[i][j]['New'] = [index_buy, j]

    xind = k
    yind = n-1
    BuySignals = []
    SellSignals = []

    while True:
        S = Signals2[xind][yind]['New']
        if len(S) != 0:
            BuySignals.append(S[0])
            SellSignals.append(S[1])

        index = Signals2[xind][yind]['From']
        if len(index) == 0:
            break
        xind = index[0]
        yind = index[1]

    indicator = np.ones(len(price))
    indicator *= 0
    # 0 us hold
    # 1 is buy
    # 2 is sell

    for i,j in zip(BuySignals, SellSignals):
        indicator[i] = 1
        indicator[j] = 2

    return profit[k][n - 1], BuySignals, SellSignals, indicator

#Literally finds the best Buy / Sell without a restriction on k transactions
def findMaxProfit(price):
    # keep track of the maximum profit gained
    #buyprices = [element * 1 / (1 - r) for element in price]
    #sellprices = [element * (1 - r) for element in price]
    profit = 0
    # initialize the local minimum to the first element's index
    j = 0
    Signals = np.full(len(price), 0)  # 0 is hold
    # start from the second element
    for i in range(1, len(price)):
        # update the local minimum if a decreasing sequence is found
        if price[i - 1] > price[i]:
            j = i
        # sell shares if the current element is the peak, i.e.,
        # (`previous <= current > next`)
        if price[i - 1] <= price[i] and \
                (i + 1 == len(price) or price[i] > price[i + 1]):
            profit += (price[i] - price[j])
            Signals[j] = 1
            Signals[i] = 2

    return Signals

def getData():
    CryptoData = []
    Variables = ["BNBUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT",
                 "NEOUSDT", "QTUMUSDT", "XRPUSDT", "LTCUSDT"]

    for x in Variables:
        path = 'CryptoMinuteData/Binance_' + x + "_1h.csv"
        data = pd.read_csv(path)
        #data = reset_my_index(data)
        CryptoData.append(data)

    return CryptoData, Variables


def DowJones():
    path = 'Dow_Jones_1896.csv'
    df = pd.read_csv(path)
    df.Date = pd.to_datetime(df.Date)
    df = df[["Date", "Close"]]
    df.rename(columns={'Close': 'close'}, inplace=True)
    isBusinessDay = BDay().is_on_offset
    match_series = pd.to_datetime(df['Date']).map(isBusinessDay)
    df = df[match_series]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def reset_my_index(df):
    res = df[::-1].reset_index(drop=True)
    return (res)

def StationarityTestOfStandardizedData(matrix, name):
    Mshape = matrix.shape
    ADFPvalues = []
    KPSSPvalues = []
    print("Dickey-Fuller Test for stationarity")
    for i in range (Mshape[0]): #Test data column wise.
        dftest = adfuller(matrix[i], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        CriticalVals = []
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
            CriticalVals.append(value)

        print(dfoutput)

        ADFPvalues.append(dftest[1])

        if dftest[1] < 0.05 and dftest[0] < min(CriticalVals):
            print("p-val < 0.05 and Test statistic < min(Critical values) therefore it is stationary")
        print("---------------------------")

    print("Kwaitkowsku-Phillips-Schmidt-Shin(KPSS) Test")
    for i in range(Mshape[0]):
        kpsstest = kpss(matrix[i], regression='c', nlags="auto")
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
        CriticalVals = []

        for key, value in kpsstest[3].items():
            kpss_output['Critical Value (%s)' % key] = value
            CriticalVals.append(value)

        print(kpss_output)

        KPSSPvalues.append(kpsstest[1])

        if kpsstest[1] - 0.05 > 0:
            print("p-val > 0.05 therefore it is stationary")

        print("---------------------------")

        print(kpss_output)

    pandasDict = {"ADF p-values" : ADFPvalues, "KPSS p-values" : KPSSPvalues}
    data = pd.DataFrame.from_dict(pandasDict)
    data.to_csv(name+"TestValues"+".csv")




if __name__ == "__main__":
    CryptoData, Variables = getData()
    DowData = DowJones()
    x = reset_my_index(CryptoData[0])
    Price = x['close']
    Volume = x['Volume ' + Variables[0].replace('USDT', '')]
    TradeCount = x['tradecount']

    '''
    Time Series Cross Validation and Network Architecture
    
    We can either Construct Neural Networks which focus on Large Windows of Data  (But not the whole dataset, the 
    actual dataset we use to produce signals can shift depending on our window size).
    
    For Example Lets say our Dataset has 10000 price points. Index = [0, 1, 2, 3, ... , 9999]
    Now lets say our large window is 2000, and we start at index = 0
    IndexTrain = [0, 1, 2, 3, ... , LargeWindow-1] = [0, 1, 2, 3, ... , 1999]
    We use the the price points from the index range 0 to 1999, for our neural network indicator. 
    
    To test the efficacy of our neural network, we can use IndexTest = [LargeWindow, LargeWindow+1, ... , LargeWindow+499]
    We produce signals on 500 test data points.  [2000, 2001, ... , 2499]
    Now using this index range, we then produce our signals and run our "Naive Simulations" for Going Long and 
    for Going Short. We look at the profit generated. 
    
    Now, for our next iteration we STILL ONLY use 2000 points but on a different index range, and we repeat the tests
    in the same manner. 
    
    Or 
    
    We can use Neural Networks which focuses entirely on the whole dataset too, but theoretically this Neural 
    Network would need to be larger, Say we started off with 2000 data points, this would eventually keep increasing. 
    '''

    '''
    Importance of Network Architecture
    
    It is not about finding the "Optimal Parameters", we want to find a Network Architecture that can capture the
    complexities of movements in price. Hence that is also, why I think it would be better to choose an adequate
    window size, the window size may be adjusted by the system or trader depending on what is happening in the 
    outside world. 
    
    It should also be noted, the larger your dataset the larger the network architecture should be too in general.
    More Data points should generally mean that there are more complexities within the data to capture. 
    '''
    # Over here I am specifying the Range to use to compute the optimal trading actions

    logR = np.diff(np.log(Price))

    dftest = adfuller(logR, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    CriticalVals = []
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
        CriticalVals.append(value)

    print(dfoutput)

    kpsstest = kpss(logR, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
    CriticalVals = []

    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
        CriticalVals.append(value)

    print(kpss_output)

    TempPrice = np.array(Price[3500:])
    Price = np.array(Price[3500:])
    Volume = np.array(Volume[3500:])
    TradeCount = np.array(TradeCount[3500:])

    n = len(Price)
    #k is the number of inputs or features the network is going to have in the first layer.
    k = 6

    # Feature Creation
    PriceFeatureMatrix = np.zeros(shape=(n-k, k))
    VolumeFeatureMatrix = np.zeros(shape=(n-k, k))
    TradeCountFeatureMatrix = np.zeros(shape=(n-k, k))
    ConvolutionalMatrix = np.zeros(shape=(n-k, 1, 3, k))

    '''
    The layout most certainly WILL have an effect on the results of the Network. Rather, there should be 
    separate neural networks, which combine the results. 
    Right Now this is essentially the structure of the input vector:
    Where P_i = Price, and V_i = Volume, T_i = Trade Count
    [P_0, P_1, ..., P_23, V_0, V_1, ... , V_23, T_0, T_1, ... , T_23]^Transpose
    '''
    GiganticMatrix = np.zeros(shape=(n-k, 3*k))
    # Extracting the signals which create the max profit with an unrestricted amount of transactions

    #Signals = findMaxProfit(list(Price))
    #Implement Iterative Optimal Signalling
    #Signals = OptimisedfindMaxProfitK(list(TempPrice), 2000)
    #Signals = Signals[3] #Use this line when specifying the number of transactions using OptimalFindMaxProfitK(price, k)
    #Signals = IterativeOptimalSignalling(list(Price), 24)
    Signals = findMaxProfit(list(TempPrice))

    '''
    [P_0, P_1, ..., P_23, V_0, V_1, ... , V_23, T_0, T_1, ... , T_23] predict S_23. 
    [P_1, P_2, ..., P_24, V_1, V_2, ... , V_24, T_1, T_2, ... , T_24] predict S_24. 
    .
    .
    .
    [P_((n-1)-(k-1)), P_((n-1)-(k-1)+1), ..., P_(n-1), V_((n-1)-(k-1)), V_((n-1)-(k-1)+1), ... , V_(n-1), T_((n-1)-(k-1)), T_((n-1)-(k-1)+1) ... , T_(n-1)] predict S_(n-1). 
    
    In general, i = [k-1, k, k+1, k+2, ... ,n-1]
    So we have:
    [P_(i-(k-1)), P_(i-(k-1)+1), ..., P_(i), V_(i-(k-1)), V_(i-(k-1)+1), ... , V_(i), T_(i-(k-1)), T_(i-(k-1)+1), ... , T_(i)] predict S_(i). 
    
    So if currently now we have signals at index = [0,1,2,3, ... ,3999]
    So since we aren't predicting anything below S_23, we remove the first 23 signals. 
    
    Signals = Signals[k-1 : -1]
    means 
    k = 24
    k-1 = 23
    Index = [23, 24, ... , 3999]
    '''

    print(f"Length of Prices = {len(TempPrice)}")
    print(f"Signals before removing non-assesable signals (all signals < k-1) = {len(Signals)}")
    Signals = np.array(Signals[k-1:-1])
    TempPrice = np.array(TempPrice[k-1:-1])
    print(f"Signals after removing non-assesable signals (all signals < k-1) = {len(Signals)}")
    print(f"Length of Temporary Prices = {len(TempPrice)}")

    '''
    Lets just let our Train Signals = Signals[k-1 : -1]. 
    '''
    
    TrainSignals, ValidationSignals = Signals[:6000], Signals[6000:]
    ValidationPrice = TempPrice[6000:]

    TrainSignals = torch.from_numpy(TrainSignals)
    TrainSignals = TrainSignals.float()

    ValidationSignals = torch.from_numpy(ValidationSignals)
    ValidationSignals = ValidationSignals.float()

    print(f"Training Signals Size = {len(TrainSignals)}")
    print(f"Validation Signals Size = {len(ValidationSignals)}")
    print(f"Length of Validation Prices = {len(ValidationPrice)}")

    i = 0
    j = 0
    OG = k
    scaler = StandardScaler()

    '''
    Note how we are iterating from 0 to n-k, we are constructing a matrix which is k by n-k.
    n= 4000, k = 24. So n-k = 3976 which is the len(TrainSignals)
    
    We are now filling our Feature Matrix. 
    '''

    tl = k
    sizes = n-k
    strDict = {0:"Hold", 1:"Buy", 2:"Sell"}

    for i in range(0, sizes):
        temp = np.zeros(shape=(3,tl))
        #PriceFeatureMatrix[i] = (Price[i:k] - min(Price[i:k]))/(max(Price[i:k]) - min(Price[i:k]))
        x = Price[i:k]; x = scaler.fit_transform(x.reshape(-1,1)).ravel(); temp[0] = x
        #print(f"x = {x}")

        y = Volume[i:k]; y = scaler.fit_transform(y.reshape(-1,1)).ravel(); temp[1] = y
        #print(f"y = {y}")
        z = TradeCount[i:k]; z = scaler.fit_transform(z.reshape(-1,1)).ravel(); temp[2] = z
        #print(f"z = {z}")

        #temp = np.transpose(temp)
        ConvolutionalMatrix[i,0] = temp
        PriceFeatureMatrix[i] = x; VolumeFeatureMatrix[i] = y; TradeCountFeatureMatrix[i] = z
        GiganticMatrix[i] = np.concatenate((PriceFeatureMatrix[i],VolumeFeatureMatrix[i], TradeCountFeatureMatrix[i]), axis=None)
        k += 1



    print(f"Print Price Feature Matrix = {PriceFeatureMatrix.shape}")

    print(f"Print Volume Feature Matrix = {VolumeFeatureMatrix.shape}")

    print(f"Print TradeCount Feature Matrix = {TradeCountFeatureMatrix.shape}")

    print(f"Print Gigantic Feature Matrix = {GiganticMatrix.shape}")
    print(f"Print Convolutional Feature Matrix = {ConvolutionalMatrix.shape}")


    k = 6
    # Train, Test= GiganticMatrix[:6000, :], GiganticMatrix[6000:, :]

    # print(ConvolutionalMatrix.shape)
    # Train, Test = ConvolutionalMatrix[:6000, :, :, :], ConvolutionalMatrix[6000:, :, :, :]
    Train, Test = ConvolutionalMatrix[:6000, :, :], ConvolutionalMatrix[6000:, :, :]

    print(f"Print Train Feature Matrix = {Train.shape}")
    print(f"Print Test Feature Matrix = {Test.shape}")

    '''
    Note how there is no DEFINED Train and TEST set YET. 
    I will do this later, I am only focusing on getting the training correct. 
    We must ensure that we remove our non-assessable signals first. 
    Then we split our TrainSignals and TestSignals. 

    For our features, we just need to split based on our split for our Signals to produce TrainSignals and TestSignals
    '''

    no_cuda = True
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    '''
    FeatureTypes is just referring to how we concatenated the 3 matrices. 
    Price, Volume and TradeCount.
    '''

    FeatureTypes = 3

    net = PatternNetwork(k)
    Train = torch.from_numpy(Train)
    Train = Train.float()

    Test = torch.from_numpy(Test)
    Test = Test.float()

    batchSize = 10
    train_dataset = torch.utils.data.TensorDataset(Train, TrainSignals)
    Train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(Test, ValidationSignals)
    Test_Loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    totalEpoch = 10000
    if list(net.parameters()):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
        # optimizer = torch.optim.Adam(net.parameters())
        for epoch in range(1, totalEpoch):
            train(net, device, Train_Loader, optimizer, epoch, batchSize, Test_Loader, ValidationPrice)

    pandadict = {"Validation Accuracy": ValidationAccuracy, "Validation Simulations": ValidationSimulation,
                 "Training Accuracy": TrainingAccuracy}
    data = pd.DataFrame.from_dict(pandadict)
    data.to_csv("PatternNetworkResults.csv")
    '''
    A couple of things:

    Instead of Concatenating Matrices we should do something different.
    Recall in Linear/Logistic Regression we have something called the Synergy Effect

    So since we have say 3 variables x_1, x_2, x_3
    x_1 is our price
    x_2 is our volume
    x_3 is our trade count

    Say this is our regression equation -> relavent for both logistic and linear regression

    y = w_0 + w_1*x_1 + w_2*x_2 + w_3*x_3 + w_4*(f(x_1,x_2)) + w_5*(f(x_1,x_3)) + w_6*(f(x_2,x_3)) + w_7*(g(x_1,x_2,x_3))

    Now let 
    x_4 = f(x_1,x_2)
    x_5 = f(x_1,x_2)
    x_6 = f(x_1,x_2)
    x_7 = g(x_1,x_2,x_3)

    Now we have:
    y = w_0 + w_1*x_1 + w_2*x_2 + w_3*x_3 + w_4*x_4 + w_5*x_5  + w_6*x_6  + w_7*x_7 

    The above represents synergy amount the variables. We apply the same concept but with neural networks.

    Now instead of x_1, x_2, ... , x_7 representing a variable.
    We let our variable instead be a neural network for a specific variable
    So now:

    x_1 = NN(X_1)
    x_2 = NN(X_2)
    x_3 = NN(X_3)
    x_4 = NN(X_4)
    x_5 = NN(X_5)
    x_6 = NN(X_6)
    x_7 = NN(X_7)

    We now have a neural network for each variable. 
    The output of those neural networks, should be a predicted classification, so essentially there will be
    7 nodes. These 7 nodes should be the max value. 

    Each of these 7 nodes are then linked to 3 output nodes. There are 21 weights.

    Backpropagation will be different. Here in the last later we want to optimise the 21 weights, 
    but we also want to optimise the parameters of the individual and independent neural networks.

    So hence the weights connecting 7 nodes to the 3 output nodes are adjusted first. The adjustment of these
    nodes should not impact the way individual neural networks undergo backpropagation. Possibly, there could 
    be more hidden layers in between the 7 classification value nodes and the 3 output nodes too. 
    '''

