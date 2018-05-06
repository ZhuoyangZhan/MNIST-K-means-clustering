
# coding: utf-8


'''
This code is a part of Group Big 4's APAN 4335 Machine Learning Final Project. It is attempting to
use DNN and K-Means Clustering Algorithm to recognize MNIST handwriting numbers. The main ML
packages used in this code is Keras and SKLearn
'''
# Training 2-hidden layer DNN with 60000 samples as Benchmark
# Activation:relu, Optimizer:rmsprop, nodes: 512/layer, drop out rate: 0.2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, RepeatVector
from keras.utils import np_utils
from sklearn.model_selection  import train_test_split
import keras.backend as K
K.set_learning_phase(1)

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train_orgin, Y_train_orgin), (X_test_orgin, Y_test_orgin) = mnist.load_data()

X_train = X_train_orgin.reshape(60000, 784)
X_test = X_test_orgin.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(Y_train_orgin, nb_classes)
Y_test = np_utils.to_categorical(Y_test_orgin, nb_classes)

model = Sequential()

#Hidden layer 1
model.add(Dense(512, input_shape=(784,)))
hidden_layer = Activation('relu')
model.add(hidden_layer) 
model.add(Dropout(0.2))  # Dropout helps prevent "overfitting" the training data

#Hidden layer 2
model.add(Dense(512))
hidden_layer_2 = (Activation('relu'))
model.add(hidden_layer_2)
model.add(Dropout(0.2))

#Output Layer
model.add(Dense(10))
model.add(Activation('softmax')) 

#Compile model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=256, epochs =15,verbose=1)

#Evaluate the model on testing set
score = model.evaluate(X_test, Y_test,)
print('Test score:', score)



# In[64]:

# Build DNN model using only 6000 samples FOM:0.0465
# Activation:relu, Optimizer:rmsprop, nodes: 512/layer, drop out rate: 0.1

# Build a subset with only 6000 samples
X_train_subset, X_train_rest, Y_train_subset, Y_train_rest = train_test_split(X_train, Y_train, test_size=0.90, random_state=1)

model2 = Sequential()

#Hidden layer 1
model2.add(Dense(512, input_shape=(784,)))
model2.add(Activation('tanh')) 
model2.add(Dropout(0.1))

#Hidden layer 2
model2.add(Dense(512))
model2.add(Activation('tanh'))
model2.add(Dropout(0.1))

#Output layer
model2.add(Dense(10))
model2.add(Activation('softmax')) 

#Compile model
model2.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model2.fit(X_train_subset, Y_train_subset, batch_size=256, epochs=15,verbose=1)

#Calculate the FOM
score = model2.evaluate(X_test, Y_test)
fom = len(X_train_subset)/(len(X_train)*2) + (1 - (score[1]))
print 'Test score:', score[1]
print 'FOM: ', fom 


# In[2]:

# K-means Clustering on trainingset with 60000 samples
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection  import train_test_split

#Setting the trainingset
X_km_train = X_train
Y_km_train = Y_train_orgin

#Using PCA to speed up and improve accuracy
pca = PCA(n_components=10).fit(X_km_train)

#Contruct the Kmeans model
km = KMeans(init=pca.components_, n_clusters=10, n_init=1)
km.fit(X_km_train)

#Determining each cluster represents what digit
validation = pd.DataFrame(X_km_train)
validation['actual'] = Y_km_train
validation['predict'] = km.labels_

from collections import Counter
i = 0
p = 0
q = 0
matrix = pd.DataFrame()
while i < 10:
    count = 0
    num_actual = 0
    numlist = []
    for index, row in validation.iterrows():
        if row['actual'] == i:
            num_actual = num_actual + 1 
            numlist.append(row['predict'])
    label = int(Counter(numlist).most_common()[0][0])
    line = pd.Series([i, label])
    matrix = matrix.append(line,ignore_index=True)
    i = i + 1

#Validating the model
for index, row in matrix.iterrows():
    digit = index
    label = int(row[1])
    n = 0
    m = 0
    for index,row in validation.iterrows():
        if row['actual'] == digit:    
            m += 1
            if row['predict'] == label:
                n += 1
    p = p + n
    q = q + m
    accuracy = float(n)/m
    print "digit: ", digit, " accuracy: ", accuracy
accuracy_all = float(p)/q
print 'overall accuracy: ', accuracy_all


# In[71]:

# Using the first hidden layer's output as a part of features to do the K-means clustering

# Importing hidden layer
import keras.backend as K
_hiddenlayer = K.function(model.inputs,[hidden_layer.output])
hiddenlayer_ = _hiddenlayer([X_km_train])[0]
hiddenlayer = pd.DataFrame(hiddenlayer_)

# Build the new training set
X_km_train2 = pd.DataFrame(X_km_train)
X_km_train2_EX = pd.concat([X_km_train2, hiddenlayer], axis=1)

# Construct new Kmeans model
pca = PCA(n_components=10).fit(X_km_train2_EX)
km2 = KMeans(init=pca.components_, n_clusters=10, n_init=1)
km2.fit(X_km_train2_EX)

#Determining each cluster represents what digit
validation = pd.DataFrame(X_km_train2_EX)
validation['actual'] = Y_km_train
validation['predict'] = km2.labels_

from collections import Counter
i = 0
p = 0
q = 0
matrix = pd.DataFrame()
while i < 10:
    count = 0
    num_actual = 0
    numlist = []
    for index, row in validation.iterrows():
        if row['actual'] == i:
            num_actual = num_actual + 1 
            numlist.append(row['predict'])
    label = int(Counter(numlist).most_common()[0][0])
    line = pd.Series([i, label])
    matrix = matrix.append(line,ignore_index=True)
    i = i + 1
    
#Validating the model
for index, row in matrix.iterrows():
    digit = index
    label = int(row[1])
    n = 0
    m = 0
    for index,row in validation.iterrows():
        if row['actual'] == digit:    
            m += 1
            if row['predict'] == label:
                n += 1
    p = p + n
    q = q + m
    accuracy = float(n)/m
    print "digit: ", digit, " accuracy: ", accuracy
accuracy_all = float(p)/q
print 'overall accuracy: ', accuracy_all    


# In[68]:

# Using the first and second hidden layer's output as a part of features to do the K-means clustering

# Importing hidden layer

_hiddenlayer1 = K.function(model.inputs,[hidden_layer.output])
_hiddenlayer2 = K.function(model.inputs,[hidden_layer_2.output])
hiddenlayer_1 = _hiddenlayer1([X_train])[0]
hiddenlayer_2 = _hiddenlayer2([X_train])[0]
hiddenlayer1 = pd.DataFrame(hiddenlayer_1)
hiddenlayer2 = pd.DataFrame(hiddenlayer_2)

# Build the new training set
X_km_train3 = pd.DataFrame(X_km_train)
X_km_train3_EX = pd.concat([X_km_train3, hiddenlayer1, hiddenlayer2], axis=1)

# Construct new Kmeans model
pca = PCA(n_components=10).fit(X_km_train3_EX)
km3 = KMeans(init=pca.components_, n_clusters=10, n_init=1)
km3.fit(X_km_train3_EX)

#Determining each cluster represents what digit
validation = pd.DataFrame(X_km_train3_EX)
validation['actual'] = Y_km_train
validation['predict'] = km3.labels_

from collections import Counter
i = 0
p = 0
q = 0
matrix = pd.DataFrame()
while i < 10:
    count = 0
    num_actual = 0
    numlist = []
    for index, row in validation.iterrows():
        if row['actual'] == i:
            num_actual = num_actual + 1 
            numlist.append(row['predict'])
    label = int(Counter(numlist).most_common()[0][0])
    line = pd.Series([i, label])
    matrix = matrix.append(line,ignore_index=True)
    i = i + 1

#Validating the model
for index, row in matrix.iterrows():
    digit = index
    label = int(row[1])
    n = 0
    m = 0
    for index,row in validation.iterrows():
        if row['actual'] == digit:    
            m += 1
            if row['predict'] == label:
                n += 1
    p = p + n
    q = q + m
    accuracy = float(n)/m
    print "digit: ", digit, " accuracy: ", accuracy
accuracy_all = float(p)/q
print 'overall accuracy: ', accuracy_all    


# In[79]:

#Visualization for clusters (Copied from Github)
reduced_data = PCA(n_components=2).fit_transform(X_km_train2_EX)
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

