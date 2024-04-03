import pandas as pd
import numpy as np
 
from keras.models import Sequential
from keras.layers import Dense

import PySimpleGUI as sg
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

import copy
import os
import shutil
import os.path
import glob

output = None
def Build_and_use_NN(pickle_path):
# To remove the scientific notation from numpy arrays
    np.set_printoptions(suppress=True)
 
    CarPricesDataNumeric=pd.read_pickle(pickle_path)
    CarPricesDataNumeric.head()


# Separate Target Variable and Predictor Variables
    TargetVariable=['Price']
    Predictors=['Age', 'KM', 'Weight', 'HP', 'MetColor', 'CC', 'Doors']
 
    X=CarPricesDataNumeric[Predictors].values
    y=CarPricesDataNumeric[TargetVariable].values
 
### Sandardization of data ###
    from sklearn.preprocessing import StandardScaler
    PredictorScaler=StandardScaler()
    TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)
    TargetVarScalerFit=TargetVarScaler.fit(y)
 
# Generating the standardized values of X and y
    X=PredictorScalerFit.transform(X)
    y=TargetVarScalerFit.transform(y)
 
# Split the data into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Quick sanity check with the shapes of Training and testing datasets
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


# create ANN model
    model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
    model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
    model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
    model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam')
 
# Fitting the ANN to the Training set
    model.fit(X_train, y_train ,batch_size = 20, epochs = 50, verbose=1)


    def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    
    # Defining the list of hyper parameters to try
        batch_size_list=[5, 10, 15, 20]
        epoch_list  =   [5, 10, 50, 100]
    
        import pandas as pd
        SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    # initializing the trials
        TrialNumber=0
        for batch_size_trial in batch_size_list:
            for epochs_trial in epoch_list:
                TrialNumber+=1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
 
            # Defining the Second layer of the model
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
 
            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))
 
            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')
 
            # Fitting the ANN to the Training set
            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
 
            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))
            
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
            temp=pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] )
            SearchResultsData=pd.concat([SearchResultsData,temp])
        return(SearchResultsData)
 
 
######################################################
# Calling the function
    ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)



  


# Fitting the ANN to the Training set
    model.fit(X_train, y_train ,batch_size = 15, epochs = 5, verbose=0)
 
# Generating Predictions on testing data
    Predictions=model.predict(X_test)
 
# Scaling the predicted Price data back to original price scale
    Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test Price data back to original price scale
    y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
    Test_Data=PredictorScalerFit.inverse_transform(X_test)
 
    TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
    TestingData['Price']=y_test_orig
    TestingData['PredictedPrice']=Predictions
    TestingData.head()


# Computing the absolute percent error
    APE=100*(abs(TestingData['Price']-TestingData['PredictedPrice'])/TestingData['Price'])
    TestingData['APE']=APE
 
    print('The Accuracy of ANN model is:', 100-np.mean(APE))
    TestingData.head()
    NN_results=TestingData

    return(NN_results)


sg.theme('Reddit')
layout =  [ [sg.Text( " Peform regression using a neural network "), sg.Input(),sg.FileBrowse(key="-IN-")],[sg.Submit()],[sg.Cancel()]]
        
newlayout = copy.deepcopy(layout)
window = sg.Window('Select and submit a pickle file containing data', newlayout, size=(270*4,4*100))
event, values = window.read()

while True:
    event, values = window.read()
    print(event, values)
    
    if event == 'Cancel':
        break
    elif event == 'Submit':
        #results=analyze_list_of_images("im_path_list")
        #results
         pickle_path= values["-IN-"]
         if pickle_path:
    
            output =  pd.DataFrame(Build_and_use_NN(pickle_path))
         
            break
window.close()

def save_file():
            file = filedialog.asksaveasfilename(
                
            filetypes=[("csv file", ".cvs")],
            defaultextension=".csv",
            title='Save Output')
            results_file=results.to_csv(str(file))
            if file: 
                            fob=open(str(results_file),'w')
                            fob.write("Save results")
                            fob.close()
            else: # user cancel the file browser window
                        print("No file chosen")
       
if output is not None:        
        my_w = tk.Tk()
        my_w.geometry("400x300")  # Size of the window 
        my_w.title('Save results as a CVS')
        my_font1=('times', 18, 'bold')
        l1 = tk.Label(my_w,text='Save File',width=30,font=my_font1)
        l1.grid(row=1,column=1)
        
        b1 = tk.Button(my_w, text='Save', 
        width=20,command = lambda:save_file())
        b1.grid(row=2,column=1)
        
       
        my_w.mainloop() 


#orginal code came from https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/  , changes are  replacing a deprecaited funciton , exporting the results and added a GUI.