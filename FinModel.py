from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import r2_score
import sys, os
from keras.models import load_model
from datetime import  datetime
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Conv2D, MaxPooling2D, TimeDistributed
from keras.layers import ConvLSTM2D


DEBUG=1

class Model:
    # create ANN model
    def __init__(self, data, labels, epochs, test_data=None, test_labels=None, is_binary=False, pretrained_model=None):
        self.is_binary=is_binary
        self.DEBUG=1
        self.test_data, self.test_labels = None, None
        if test_data is not None:
            self.test_data =  np.asarray(test_data)
        self.data, self.labels = np.asarray(data), np.asarray(labels)
        
        
        # split to train and test
        self.setTrainAndTestData
            
        self.defineModelParams(epochs)
        
        if self.DEBUG>0:
            if pretrained_model is not None:
                print("====================> have a pre-trained model to load")
        
        if pretrained_model is not None:
            # set it up
            self.model=pretrained_model
        else:
            self.__build_LSTM
        
    ###################################################
    # train & test set settings
    ###################################################
    @property
    def setTrainAndTestData(self):
        if self.test_data is None:
            # split to train and test
            self.X=self.data
            self.Y=self.labels
            self.X_train,  self.X_test,  self.Y_train,  self.Y_test = \
                    train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
            if self.DEBUG>0:
                print("Data was split 0.8/0.2")
        else:
            self.X_train=self.data
            self.Y_train=self.labels
            self.X_test =self.test_data
            #self.Y_test=self.test_labels
            if self.DEBUG>0:
                print("Data was set to predetermined train/test")
                
        self.shape = self.X_train.shape
        if self.DEBUG>0:
            print("====> X_train.shape ",self.X_train.shape," Y_train.shape= ",self.Y_train.shape)
            print("====> X_test.shape ",self.X_test.shape)
        
    ###################################################
    # compile & run
    ###################################################
    @property
    def predict(self):
        self.compile
        self.fit
        self.evaluate
        
    def get_predictions(self):
        self.predictions = self.model.predict(self.X_test)
        self.predictions =[p[0] for p in self.predictions]
        if self.DEBUG>2:
            print("predictions[0:10]=",self.predictions[0:10])
            print("test labels[0:10]=",(self.Y_test.tolist())[0:10])
        if self.DEBUG>0:
            #print("Y_test.shape=",self.Y_test.shape)
            print("self.predictions.shape=",np.asarray(self.predictions).shape)
            if self.is_binary:
                self.predictions=[0 if p<0.5 else 1 for p in self.predictions]
        if self.DEBUG>2:    
            print("self.predictions      =",self.predictions)
            '''
            test_labels=[]
            for yt in self.Y_test:
                test_labels.extend(yt)
            test_labels=np.asarray(test_labels)
            print("test_labels.shape=",test_labels.shape)
            '''
            if not self.is_binary:
                r2=r2_score(self.predictions, self.Y_test)
                print("===========> R2=",r2)
        return self.predictions
        
    @property
    def evaluate(self):
        if self.DEBUG>0:
            print("evaluate X_test.shape=",self.X_test.shape)
            print("evaluate Y_test.shape=",self.Y_test.shape)
        eval_results = self.model.evaluate(self.X_test, self.Y_test, batch_size=self.__batch_size)
        if self.DEBUG>0:
            print("==========> Evaluation results are ",self.model.metrics_names,"=",eval_results)
        
    @property
    def fit(self):
        if self.DEBUG>0:
            print("fit X_train.shape=",self.X_train.shape)
            print("fit Y_train.shape=",self.Y_train.shape)
            print("fit X_train type=",type(self.X_train))
            print("fit Y_train type=",type(self.Y_train))
        if self.DEBUG>2:
            for seq in self.X_train:
                print(np.asarray(seq).shape)
        self.model.fit(self.X_train, self.Y_train, epochs=self.__epochs, shuffle=False)
        
    @property
    def compile(self):    
        # compile
        try:
            self.model.compile(loss=self.__loss, optimizer='adam', metrics=self.__metrics)
            print(str(self.model.summary(print_fn=print)))
        except:pass
        
        
    ###################################################
    # default model params
    ###################################################
    def defineModelParams(self, epochs):
        self.__epochs = epochs
        self.__batch_size = 1
        self.__dropout = 0.2
        self.model = Sequential()
        self.__lstm_units = 50
        self.__kernel_size = 3
        self.__kernel_size2d = (2,2)
        self.__filters = 100
        self.__pool_size = 4
        self.__pool_size2d = (2,2)
        self.__strides1d =  2
        self.__strides2d = (1,1)
        self.__conv1d_padding = 'same'
        self.__conv1d_activation = 'relu'
        self.__conv2d_padding = 'same' #'same'
        self.__conv2d_activation = 'relu'
        self.__dense_activation = 'relu'
        self.__lstm_units = 20
        self.__final_activation='sigmoid' if self.is_binary else None
        self.__loss='binary_crossentropy' if self.is_binary else 'mse'
        self.__metrics=['accuracy'] if self.is_binary else ['mse']
        
          
    @property        
    def __build_LSTM(self):
        # reshape the data
        #print("--------------------------LSTM---------------------------------")
        if self.DEBUG>0:
            print(" LSTM X_train.shape=",self.X_train.shape)
            print(" LSTM Y_train.shape=",self.Y_train.shape)
            print(" LSTM X_test.shape=",self.X_test.shape)
            #print(" LSTM Y_test.shape=",self.Y_test.shape)
        self.shape=(self.X_train.shape[1], self.X_train.shape[2])
        # build the model
        if self.DEBUG>0:
            print("-----------------------------> LSTM starting")
            print("-----------------------------> LSTM input_shape=", self.shape)
        self.model.add(Bidirectional(LSTM(self.__lstm_units, input_shape=self.shape)))
        self.model.add(Dense(1, activation=self.__final_activation))
        
        if self.DEBUG>0:
            print("-----------------------------> LSTM definition done")
        
        
