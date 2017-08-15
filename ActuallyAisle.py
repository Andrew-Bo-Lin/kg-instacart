import numpy as np
import pandas
import csv as snr
import tensorflow as tf
import scipy
import h5py
import math
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K
print("Meow, this is a cat's debugger 11")

np.random.seed(1234)
csv = np.genfromtxt("20prior.csv", delimiter=",")
print("Initial Array Generated")
csv = csv.tolist()
print("Converted to List")
csvt = np.genfromtxt("20train.csv", delimiter=",")
print("Test Array Generated")
csvt = csvt.tolist()
print("Converted to List")
num_orders = np.genfromtxt('20numorders.csv', delimiter = ",").tolist()
a = np.zeros((10311*4,99,134 ))
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))
b = np.zeros((10311*4,99,36 ))

major_aisles = [3,16,17,21,31,37,38,53,67,69,77,78,81,91,93,96,98,106,107,108,112,115,116,117,120,121,123]
mid_aisles = [49,43,114,110,51,13,99,14,64,105,74,122,57,42,89,75,2,1,34,29,5,85,95,100,30,20,25,48,58,127,
              6,111,65,47,41,12,7,39,60,124,125,71,22,8,70,56,27,28]
lower_aisles = [36,45,59,19,9,54,32,94,128,61,52,26,92,129,72,4,23,104,79,50,130,35,66,63]
lowest_aisles = [11,62,90,119,40,15,87,97,133,46,126,101,80,68,18,134,55,73,76,103,82,109,102,118,44,33,10,132,113]

counter = 1
def fileloader(file,a,b,major_aisles,mid_aisles,lower_aisles,boolean):
    counter = 0
    while (counter < len(file)):
        if (math.isnan(file[counter][0]) == False):
            user_id = file[counter][0]
            order_number = int(file[counter][3])
            offset = int(user_id) / 5
            Aisle_offset = int(file[counter][2] - 1)
            Aisle_ID = file[counter][2]
            if (math.isnan(file[counter][6])):
                file[counter][6] = 0
            if (user_id % 20 == 1 and boolean == True):
                a[offset][order_number - 1][Aisle_offset] = file[counter][7]
            if (order_number > 1 and file[counter][8] == 1):
                marker = 0
                for i in range(len(major_aisles)):
                    if (Aisle_ID == major_aisles[i]):
                        b[offset][order_number - 2][i] = 1
                        marker = 1
                for i in range(len(mid_aisles)):
                    if (Aisle_ID == mid_aisles[i]):
                        b[offset][order_number - 2][33] = 1
                        marker = 1
                for i in range(len(lower_aisles)):
                    if (Aisle_ID == lower_aisles[i]):
                        b[offset][order_number - 2][34] = 1
                        marker = 1
                if (marker != 1):
                    b[offset][order_number - 2][35] = 1

        counter += 1
        if (counter == len(file) / 4 * 3):
            print("Load: 75% done")
        elif (counter == len(file) / 4 * 2):
            print("Load: 50% done")
        elif (counter == len(file) / 4):
            print("Load: 25% done")

fileloader(csv, a, b, major_aisles, mid_aisles, lower_aisles,True)
fileloader(csvt, a, b, major_aisles, mid_aisles, lower_aisles,False)

##for x in range(len(b)):
##    for y in range(len(b[0])):
##        for z in range(len(b[0][0])):
##            if(b[x][y][z] != 1):
##                b[x][y][z] = -1.0



a = a.reshape((10311*99*4,134))
b = b.reshape((10311*99*4,36))

train_size = int(len(a) * 2.0/3.0)
test_size = len(a) - train_size
trainy_size = int(len(b)* 2.0/3.0)
testy_size = len(b) - trainy_size
scaler = MinMaxScaler(feature_range=(0, 1))
##dataset = scaler.fit_transform(dataset)
trainx, testx = a[0:train_size,:], a[train_size:len(a),:]
trainx = scaler.fit_transform(trainx)
testx = scaler.fit_transform(testx)
trainy, testy = b[0:trainy_size,:], b[trainy_size:len(b),:]
##trainy = scaler.fit_transform(trainy)
##testy = scaler.fit_transform(testy)
trainx = trainx.reshape((10311/3*2*4, 99, 134))
testx = testx.reshape((10311/3*4,99,134))
trainy = trainy.reshape((10311/3*2*4, 99, 36))
testy = testy.reshape((10311/3*4,99,36))

Eve = keras.optimizers.Adam(lr = 0.000314159, beta_1 = 0.9, beta_2=0.999, epsilon =1e-08)
Steve = keras.optimizers.SGD(lr=0.01, momentum=0.5, decay=0.00001, nesterov=True)
Meow = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.01)
Neve = keras.optimizers.Nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

batch_size = 1

model = Sequential()
model.add(Masking(mask_value = 0.0 ,input_shape=(99,134)))
##model.add(LSTM(268,return_sequences = True,kernel_initializer='random_uniform',
##                bias_initializer='zeros'))
model.add(LSTM(25,return_sequences = True,activation = "relu", kernel_initializer='random_uniform',
                bias_initializer='zeros',recurrent_activation = 'sigmoid', use_bias = True,input_shape=(99,134), implementation = 0))
model.add(LSTM(25,return_sequences = True,activation = "relu", kernel_initializer='random_uniform',
                bias_initializer='zeros', use_bias = True,implementation = 0,recurrent_activation = 'sigmoid'))
model.add(LSTM(25,return_sequences = True,activation = "relu", kernel_initializer='random_uniform',
                bias_initializer='zeros', use_bias = True,implementation = 0,recurrent_activation = 'sigmoid'))
model.add(keras.layers.wrappers.TimeDistributed(Dense(25, activation = "sigmoid", kernel_initializer='random_uniform',
                bias_initializer='zeros', use_bias = True)))
##model.add(keras.layers.wrappers.TimeDistributed(Dense(134,kernel_initializer='random_uniform',
##                bias_initializer='zeros')))
##model.add(keras.layers.wrappers.TimeDistributed(Dense(268, activation = "softmax",kernel_initializer='random_uniform',
##                bias_initializer='zeros')))
model.add(keras.layers.wrappers.TimeDistributed(Dense(36, activation = "sigmoid", kernel_initializer='random_uniform',
                bias_initializer='zeros', use_bias = True)))
model.compile(loss = 'binary_crossentropy', optimizer = Eve, metrics = [f1])

model.fit(trainx,trainy,validation_data=(testx,testy), epochs = 20, batch_size = 200,  verbose = 2, shuffle = True)
prediction = model.predict(testx)
counter = 0.0;
counter2 = 0.0;
sum1 = 0.0;
sum2 = 0.0;
counter3 = 0
counter4 = 0

for x in range(len(testy)):
    for y in range(len(testy[0])):
        for z in range(len(testy[0][0])):
            if(testy[x][y][z] == 1):
                counter += 1
                sum1 += prediction[x][y][z]
                if (prediction[x][y][z] >= 0.6) :
                    counter3 += 1
            elif(testy[x][y][z] == 0):
                counter2 += 1
                sum2 += prediction[x][y][z]
                if (prediction[x][y][z] <= 0.4) :
                    counter4 += 1
average1 = sum1/counter;
average2 = sum2/counter2;


print("We predicted " + str(counter3/counter *100) + "% of 1's with relative certainty and " + str(counter4/counter2 * 100) + " -1's with relative certainty")
print("The average prediction for a 1 was :" + str(average1) + " and the average prediction for a -1 was " + str(average2) )
##print("LSTM Weights")
##print(model.layers[1].get_weights()[0])
##print("LSTM Biases")
##print(model.layers[1].get_weights()[1])
##print("Dense Weights")
##print(model.layers[2].get_weights()[0])
##print("Dense Biases")
##print(model.layers[2].get_weights()[1])
with open('finaloutput.csv', 'wb') as myfile:
    wr = snr.writer(myfile)
    for i in range(len(num_orders)):
        arrayindex = num_orders[i][0]/5
        last_order = num_orders[i][1]-1
        for j in range(len(prediction[0][0])):
            if(prediction[arrayindex][last_order][j] >0.5):
                if(j<len(major_aisles)):
                    wr.writerow([num_orders[i][0], major_aisles[j]])
                elif(j== 33):
                    for x in range(len(mid_aisles)):
                        wr.writerow([num_orders[i][0], mid_aisles[x]])
                elif(j == 34):
                    for x in range(len(lower_aisles)):
                        wr.writerow([num_orders[i][0], lower_aisles[x]])
                else:
                    for x in range(len(lowest_aisles)):
                        wr.writerow([num_orders[i][0], lowest_aisles[x]])


with open('output.csv', 'wb') as myfile:
    wr = snr.writer(myfile)
    for i in range(len(prediction)):
        for j in range((len(prediction[0]))):
            sublist = [prediction[i][j][0],trainy[i][j][0]]
            wr.writerow(sublist)
np.savetxt('e.csv', prediction[0], delimiter = ',')
np.savetxt('f.csv', testy[0], delimiter = ',')
model.save("mymodel.h5")
model.reset_states()