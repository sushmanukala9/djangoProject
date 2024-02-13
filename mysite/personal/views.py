from django.shortcuts import render

import codecs
import pandas as pd
from personal.forms import HomeForm,UserlistForm
import pickle
import math
import numpy as np
      
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression, SGDRegressor
import sklearn.model_selection as model_selection
from sklearn.metrics import mean_squared_error
import datetime
        
import matplotlib.pyplot as plt
import csv
from matplotlib import style
import pickle

import numpy as np
  


    

def columns(reader):
     

        style.use('ggplot')
       
        
        df = reader
        list1 = list(df.columns.values)
        return list1
def testandtrain(reader, factorArr):
       

        style.use('ggplot')
       
        
        df = reader
        list1 = list(df.columns.values)
        df = df[factorArr]
        #df.is_copy = False
        #df = df.convert_objects(convert_numeric=True)
        #df['label'] = df['Sales']
        
        for i in range(len(factorArr)):
               if ((df[factorArr[i]].dtype.name == 'object') or (df[factorArr[i]].dtype.name == 'int64')or (df[factorArr[i]].dtype.name == 'int16')or (df[factorArr[i]].dtype.name == 'category') or (df[factorArr[i]].dtype.name == 'bool') or (df[factorArr[i]].dtype.name == 'datetime64') or (df[factorArr[i]].dtype.name == 'timedelta[ns]') ):
                      print(df[factorArr[i]].dtype.name )
                      df[factorArr[i]]= pd.Categorical(df[factorArr[i]]).codes
                      print(df[factorArr[i]].dtype.name)


        

        print(df.head())
        
        print(df.shape)
        df.fillna(df.mean(), inplace=True)       
        
        X = np.array(df.drop(['Sales'], 1))
        print(df.drop(['Sales'], 1))
        print(X)                                        
        X = preprocessing.scale(X)
        print("###")
        print(X)
        

        #X_lately = X[-100:]
        

        y = np.array(df['Sales'])
        print("*********")
        print(X,y)
        print("*********")
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
        
        clf = SGDRegressor()

        def train(X_train,y_train,clf):
                clf.partial_fit(X_train, y_train)
       
        def acc(X_test, y_test, clf):
            print(X_test, y_test)    
            accuracy = clf.score(X_test, y_test)
            print(accuracy)
            return accuracy

        train(X_train,y_train,clf)
        model_accuracy = acc(X_test,y_test,clf)
          


        with open('linearregression.pickle', 'wb') as f:
            pickle.dump(clf, f)
       
        
        return [model_accuracy];

def prediction(reader1,factorArr):
          pickle_in = open('linearregression.pickle', 'rb')
          clf = pickle.load(pickle_in)
          Z= reader1
          df = Z[factorArr]
          for i in range(len(factorArr)):
               if ((df[factorArr[i]].dtype.name == 'object') or (df[factorArr[i]].dtype.name == 'int64')or (df[factorArr[i]].dtype.name == 'int16')or (df[factorArr[i]].dtype.name == 'category') or (df[factorArr[i]].dtype.name == 'bool') or (df[factorArr[i]].dtype.name == 'datetime64') or (df[factorArr[i]].dtype.name == 'timedelta[ns]') ):
                      print(df[factorArr[i]].dtype.name )
                      df[factorArr[i]]= pd.Categorical(df[factorArr[i]]).codes
                      print(df[factorArr[i]].dtype.name)
          def pred(df,clf) :

              forecast_se = clf.predict(df)
              return forecast_se
          prediction_set = pred(df, clf)
          return (prediction_set/100)


def training(reader2,factorArr):
        pickle_in = open('linearregression.pickle', 'rb')
        clf = pickle.load(pickle_in)
        df= reader2
        df = df[factorArr]
        for i in range(len(factorArr)):
              #if ((df[factorArr[i]].dtype.name == 'object') or (df[factorArr[i]].dtype.name == 'category') or (df[factorArr[i]].dtype.name == 'bool') or (df[factorArr[i]].dtype.name == 'datetime64') or (df[factorArr[i]].dtype.name == 'timedelta[ns]') ):
               #if ((df[factorArr[i]].dtype.name == 'object') or (df[factorArr[i]].dtype.name == 'int64')or (df[factorArr[i]].dtype.name == 'int16')or (df[factorArr[i]].dtype.name == 'category') or (df[factorArr[i]].dtype.name == 'bool') or (df[factorArr[i]].dtype.name == 'datetime64') or (df[factorArr[i]].dtype.name == 'timedelta[ns]')  ):
                      df[factorArr[i]]= pd.Categorical(df[factorArr[i]]).codes
        
        #df.dropna(inplace=True)
        X = np.array(df.drop(['Sales'], 1))	
                                                 
        X = preprocessing.scale(X)
       
        y = np.array(df['Sales'])
       
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
        

        def train(X_train,y_train,clf):
                clf.partial_fit(X_train, y_train)
    


        def acc(X_test, y_test, clf):
            accuracy = clf.score(X_test, y_test)
            return accuracy

        
        train(X_train,y_train,clf)
        model_accuracy = acc(X_test,y_test,clf)


        with open('linearregression.pickle', 'wb') as f:
            pickle.dump(clf, f)
        pickle_in = open('linearregression.pickle', 'rb')
        clf = pickle.load(pickle_in)

        #prediction_set = pred(Z, clf)

        #mse = mean_squared_error(Y_lately,y_test)

        #sq=np.sqrt(mse)
        
        return [model_accuracy];


def index(request):
    if request.POST and request.FILES:
            
        csvfile = request.FILES['csv_file']
       
        reader = pd.read_csv(csvfile)
      
    return render(request,'personal/home.html',globals())



def contact(request):
        
 
        if request.FILES:
            csvfile = request.FILES['csv_file']
            global reader
            reader = pd.read_csv(csvfile)
            global list1
            list1 = columns(reader)
            list1.remove('Sales')
       
            
        return render(request, 'personal/basic.html' , {'content':list1})
def string(request):
        global arr
        
        
        if request.method == 'POST':
           
        # You have access to data inside request.POST
            
            arr=[]
           
            i=0
            print('test')
            print(list1)
            for c in list1:
                    
                check = request.POST.get(c)

                
                if check is not None:

                    arr.append(c)
                    i=i+1
            arr.append('Sales')
            print(arr)
            list2 = testandtrain(reader,arr)


        return render(request,'personal/string.html', {'content':['model accuracy is ' , list2]})
def predict(request):
        if request.POST and request.FILES:
            
                csvfile = request.FILES['csv_file']
                global reader1
                reader1 = pd.read_csv(csvfile)
      
        return render(request,'personal/predict.html',globals())
        

def train(request):
        if request.POST and request.FILES:
            
                csvfile = request.FILES['csv_file']
                global reader2
                reader2 = pd.read_csv(csvfile)
      
        return render(request,'personal/train.html',globals())
def predict2(request):
        
        if request.FILES:
            csvfile = request.FILES['csv_file']
            global reader1
            reader1 = pd.read_csv(csvfile)
        if 'Sales' in arr:
            arr.remove('Sales')
        
        list3= prediction(reader1,arr)

        return render(request,'personal/predict2.html',{'content':['the sales prediction is',list3]})
def train2(request):
        
        if request.FILES:
            csvfile = request.FILES['csv_file']
            global reader2
            reader2 = pd.read_csv(csvfile)
        if 'Sales' not in arr:
            
             arr.append('Sales')   
        list4 = training(reader2,arr)
        return render(request,'personal/train2.html',{'content':['successfully trained your model','and new model accuracy is:',list4]})
