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
from operator import itemgetter, attrgetter, methodcaller
print("Meow, this is a cat's debugger 12")

np.random.seed(1234)
train = sorted(np.genfromtxt("train2.csv", delimiter=",").tolist(), key = itemgetter(0))
print("Initial Array Generated")
test = sorted(np.genfromtxt("test2.csv", delimiter=",").tolist(), key = itemgetter(0))
print("Initial Array Generated")
print("Converted to List")
csvt = np.genfromtxt("valid2.csv", delimiter=",")
print("Test Array Generated")
csvt = sorted(csvt.tolist(), key = itemgetter(0))
print("Converted to List")
num_orders = np.genfromtxt('numberorders.csv', delimiter = ",").tolist()
a = np.zeros((30000,99,134 ))
c = np.zeros((20000,99,134))


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
b = np.zeros((30000,99,36 ))
userid = []
uselesslist = []
major_aisles = [3,16,17,21,31,37,38,53,67,69,77,78,81,91,93,96,98,106,107,108,112,115,116,117,120,121,123]
mid_aisles = [49,43,114,110,51,13,99,14,64,105,74,122,57,42,89,75,2,1,34,29,5,85,95,100,30,20,25,48,58,127,
              6,111,65,47,41,12,7,39,60,124,125,71,22,8,70,56,27,28]
lower_aisles = [36,45,59,19,9,54,32,94,128,61,52,26,92,129,72,4,23,104,79,50,130,35,66,63]
lowest_aisles = [11,62,90,119,40,15,87,97,133,46,126,101,80,68,18,134,55,73,76,103,82,109,102,118,44,33,10,132,113]
def fileloader(list,numpyarray,user_id,boolean, numpyarray2):
    user_index = -1;
    for i in range(len(list)):
        marker = 0
        if len(user_id)>0:
            if(user_id[len(user_id)-1] == list[i][0]):
                marker = 1
        if(marker == 0):
            if(math.isnan(list[i][0])== False):
                user_index += 1
                user_id = user_id + [int(list[i][0])]

        if (math.isnan(list[i][0]) == False):
            order_num = int(list[i][3]) - 1
            aisle_index = int(list[i][2] - 1)

            numpyarray[user_index][order_num][aisle_index] = 1
            order_num -= 1
            if (order_num >= 0 and boolean):
                Aisle_ID = int(list[i][2])
                if (Aisle_ID in major_aisles):
                    numpyarray2[user_index][order_num][major_aisles.index(Aisle_ID)] = 1
                elif (Aisle_ID in mid_aisles):
                    numpyarray2[user_index][order_num][33] = 1
                elif(Aisle_ID in lower_aisles):
                    numpyarray2[user_index][order_num][34] = 1
                else:
                    numpyarray2[user_index][order_num][35] = 1
    return [numpyarray,user_id,numpyarray2]
listicle = fileloader(train, a,userid,True, b)
a = listicle[0]
user_id = listicle[1]
b = listicle[2]

print("List 1 done")
listicle=fileloader(test,c,uselesslist,False, b)
c= listicle[0]
store = listicle[1]
print("List 2 done")
def validloader(list,list2,numpyarray,user_id):
    user_index = 0;
    for i in range(len(list2)):
        if(list2[i][0] in user_id):
            user_index = user_id.index(list[i][0])
            order_num = int(list[i][3]) - 2
            if (order_num >= 0):
                marker = 0
                Aisle_ID = int(list[i][2])
                if (Aisle_ID in major_aisles):
                    numpyarray[user_index][order_num][major_aisles.index(Aisle_ID)] = 1


                elif (Aisle_ID in mid_aisles):
                    numpyarray[user_index][order_num][33] = 1

                elif(Aisle_ID in lower_aisles):
                    numpyarray[user_index][order_num][34] = 1
                else:
                    numpyarray[user_index][order_num][35] = 1
    return numpyarray

b=validloader(train,csvt,b,user_id)
print("List 3 done")
##for x in range(len(b)):
##    for y in range(len(b[0])):
##        for z in range(len(b[0][0])):
##            if(b[x][y][z] != 1):
##                b[x][y][z] = -1.0



model = keras.models.load_model("mymodel.h5")

model.fit(a,b,epochs = 5, batch_size = 1000,  verbose = 2, shuffle = True)
prediction = model.predict(c)



##print("LSTM Weights")
##print(model.layers[1].get_weights()[0])
##print("LSTM Biases")
##print(model.layers[1].get_weights()[1])
##print("Dense Weights")
##print(model.layers[2].get_weights()[0])
##print("Dense Biases")
##print(model.layers[2].get_weights()[1])
with open('finaloutput2.csv', 'wb') as myfile:
    wr = snr.writer(myfile)
    for i in range(len(num_orders)):
        if(num_orders[i][0] in store):
            arrayindex = store.index(num_orders[i][0])
            last_order = int(num_orders[i][1]-1)
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


model.save("mymodel.h5", custom_objects = {"f1":f1})
model.reset_states()