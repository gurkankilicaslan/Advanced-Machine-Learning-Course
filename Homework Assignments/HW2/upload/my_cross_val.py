# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:22:31 2023

@author: grknk
"""



"""
Gurkan Kilicaslan 5810725 kilia064@umn.edu
"""
"""
I'm sorry it's been a very primitive code. Hopefully you'll understand what I did,
if you follow the comments. However, feel free to contact me in any case. Thanks.
"""
import numpy as np
import pandas as pd
import random as rd

def my_cross_val(model, loss_func, X, y, k=10):
    
    X_list = []
    for i in range(len(X)):
        temp_Xi = X[i].tolist() #convert array to list
        temp_Xi.append(y[i].tolist()) #convert y[i] array to list and append it
        X_list.append(temp_Xi) #convert 
    
    list_X_y = X_list #list_X_y contains information from both X and y. Our dataset
    data_append = [] #empty list to play with later, please see its usage below
    
    length_data = np.size(list_X_y,0) #length of data
        
    # print(np.size(list_X_y,0))
    #print(list_X_y[1])
    
    for i in range(0,k):
        fold_i = []
        for j in range(length_data):
            if len(fold_i) < int(length_data / k):
                # int(length_data / k) --> we've k fold and we're making room for each fold
                
                rand_item = rd.choice(list_X_y)
                list_X_y.remove(rand_item)
                fold_i.append(rand_item)
                # removed a random item from dataset and added it to our current fold
            else:
                break
                # break if not len(fold_i) < int(length_data / k)
                # not using else - break wouldn't change anything
        data_append.append(fold_i) # add all folds to create new dataset
    
    data_append = pd.DataFrame(data_append)
    # change the shape from (n,m,t) to (n,m)
    data_append = data_append.T
    # take transpose

    #return data_append
    
    
    # abc = data_append.values.tolist()
    # print(abc)
    

    mse_all = []
    err_rate_all = []   
    for i in range(k):
            
        fold_use = data_append[i] # data_append contains the folds, fold_use is the target fold, i'th fold

        # list_empt = np.empty((0, len(fold_use[0])))
        list_empt=[]
        for t in range(len(fold_use[0])):
            list_empt.append([])
        list_empt = np.transpose(list_empt) # list_empt is now a list with shape (0,len(fold_use[0])
        # print(type(list_empt))
        for j in range(len(fold_use)):
            list_empt = np.vstack((list_empt,np.asarray(fold_use[j])))
        
        # model_list_empt = list_empt
        # for m in range(k-1):
            
        #     noise = np.random.normal(0, 1, list_empt.shape)
        #     list_empt_noise = list_empt + noise
            
        #     model_list_empt = np.concatenate((model_list_empt, list_empt_noise), axis=0)
        # # print(model_list_empt.shape)

        list_empt = pd.DataFrame(list_empt) # make list_empt dataframe
        Xith = list_empt[list_empt.columns[:-1]].to_numpy()
        y_ith = list_empt[list_empt.columns[-1]]
        # print(type(y_ith))
        
        # model_list_empt = pd.DataFrame(model_list_empt)
        # X_model = model_list_empt[model_list_empt.columns[:-1]]
        # y_model = model_list_empt[model_list_empt.columns[-1]]
        
        
        fit_df = fitmodel(data_append)
        X_fit_df = fit_df[list_empt.columns[:-1]].to_numpy()
        y_fit_df = fit_df[list_empt.columns[-1]].to_numpy()
        # could also use iloc instead  of list_empt.columns
        # print(X_model)
        model.fit(X_fit_df, y_fit_df)
        
        # Mean Squared Error:
        if loss_func == 'mse':
            sum_mse = 0
            for h in range(len(y_ith)):
                sum_mse = sum_mse + (y_ith[h] - model.predict(Xith)[h])**2
            mse = sum_mse/len(y_ith)
            mse_all.append(mse)
        
        # Error Rate
        elif loss_func == 'err_rate':
            yHat = model.predict(Xith)
            sum_err = 0
            for h in range(len(y_ith)):
                if yHat[h] != y_ith[h]:
                   sum_err = sum_err + 1 
            err_rate = sum_err/len(y_ith)
            err_rate_all.append(err_rate)
            

    if loss_func == 'mse':
        mean_mse = np.mean(mse_all)
        std_mse = np.std(mse_all)
        mse_loss = [ '%.5f' % ind for ind in mse_all]
        mse_loss.append('Mean: '+ str(round(mean_mse,5)) + '  Std: ' + str(round(std_mse,5)))
        return mse_loss
    
    elif loss_func == 'err_rate':
        mean_error_rate = np.mean(err_rate_all)
        std_error_rate = np.std(err_rate_all)
        err_rate_loss = [ '%.5f' % ind for ind in err_rate_all ]  
        err_rate_loss.append('Mean: '+ str(round(mean_error_rate,5)) + '  Std: ' + str(round(std_error_rate,5)))
        return err_rate_loss    

def fitmodel (dataframe):
    
    column_num = len(dataframe.values[0, 0])
    fit_list = np.zeros((0, column_num))
    # print(type(dataframe))
    col = len(dataframe.columns)
    row = len(dataframe.index)
    for i in range(col):
        list_empt = np.empty((0, len(dataframe.values[0, 0])))
        
        row = len(dataframe.index)
        for j in range(row):
            list_empt = np.vstack((list_empt, dataframe.values[j, i]))
        
        fit_list = np.vstack((fit_list, list_empt))
        fit_list = pd.DataFrame(fit_list)
    return fit_list



    


    



