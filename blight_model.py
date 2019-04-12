#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:26:46 2019

@author: yanielc
"""

# this script reads in a csv file with information relating to fines 
# given in Detroit for blight violations. It uses a tree classifier 
# algorithm to predict the probability that any given fine will be
# paid

# The program returns a series indexed by the ticket_id coming from the csv
# and the predicted probability of payment
import matplotlib.pyplot as plt
import numpy as np, pandas as pd



## to plot importance of features (the columns of X_train)

def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)




def blight_model():
    
    
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    
    #evaluations
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import auc, roc_curve
    
    
    #read in data
    df = pd.read_csv('detroit_fine_info.csv', encoding='latin1', low_memory=False).set_index('ticket_id')
    df = df[df['compliance'].notna()] # keep entries with a valid target label

    

##############################################################################    
#    ## short code to learn which columns are all NaN in the dataframe
#    
#    for column in df.columns:
#        if len(df[column][df[column].isna()]) == len(df):
#            print(column)
#        else:
#            pass
# the code above shows that we need to exclude violation_zip_code and grafitti_status
# also exclude non_us_str_code because is almost all NaN
###############################################################################
    
    
    # select columns which don't reveal future information
    cols_use = ['agency_name','inspector_name',
                'violation_street_number','violation_street_name',
                'mailing_address_str_number',
                'mailing_address_str_name', 'city', 'zip_code',
                'ticket_issued_date','hearing_date',
                'violation_code', 'disposition', 'fine_amount',
                'admin_fee', 'state_fee','late_fee','discount_amount','clean_up_cost',
                'judgment_amount']
    
    
    X = df[cols_use].dropna(axis=0)
    y = df['compliance'].loc[X.index]
    
  
    
    # encode strings for training data and testing data
    le = LabelEncoder()
    for column in X.columns:
        if X[column].dtypes == object:
            X[column] = le.fit_transform(X[column].astype(str))  
        else:
            pass
    
    # break into training and testing data
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
    
    
    
#####################################################################
# code to tune max_depth in tree classifier
#            
#            
#    max_depths = np.linspace(1,len(X_train.columns),len(X_train.columns),endpoint=True)
#    
#    train_results = []
#    for max_depth in max_depths:
#        dt = DecisionTreeClassifier(max_depth=max_depth)
#        dt.fit(X_train,y_train)
#        
#        y_pred = dt.predict(X_train)
#        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred)
#        roc_auc = auc(false_positive_rate, true_positive_rate)
#        
#        train_results.append(roc_auc)
#      
#    from matplotlib.legend_handler import HandlerLine2D
#    line1, = plt.plot(max_depths, train_results, label="Training AUC")
#    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#    plt.ylabel("AUC score")
#    plt.xlabel("Tree depth")
#    plt.show()
#
##
# Use 5 in max depth
    
#########################################################################
# to tune min_samples_split in tree classifier
#    
#    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
#    train_results = []
#
#    for min_samples_leaf in min_samples_leafs:
#        dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
#        dt.fit(X_train, y_train)
#        
#        train_pred = dt.predict(X_train)
#        
#        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#        roc_auc = auc(false_positive_rate, true_positive_rate)
#        train_results.append(roc_auc)
#        
#        
#    from matplotlib.legend_handler import HandlerLine2D
#    line1, = plt.plot(min_samples_leafs, train_results, label='Train AUC')
#
#    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#    plt.ylabel('AUC score')
#    plt.xlabel('min samples leaf')
#    plt.show()



    
    treeClsf = DecisionTreeClassifier(max_depth=5).fit(X_train,y_train)
    
    print('score on training data', treeClsf.score(X_train,y_train))
    print('score on training data', treeClsf.score(X_test,y_test))
    
    ### evaluations
    y_test_pred = treeClsf.predict(X_test)
    y_train_pred = treeClsf.predict(X_train)
    
    confMatrix = confusion_matrix(y_test,y_test_pred)
    print('Confusion matrix on testing data:\n',confMatrix)
    
    #### accuracy score = (TP + TN)/(TP+TN+FP+FN)
    print('Accuracy score on testing data = {:.3f}'.format(accuracy_score(y_test,y_test_pred)))
    
    #### precision score = TP / (TP + FP)
    print('Precision score on testing data = {:.3f}'.format(precision_score(y_test,y_test_pred)))
    
    #### recall score = TP / (TP + FN)
    print('Recall score on testing data = {0:.3f}'.format(recall_score(y_test,y_test_pred)))
    
    #### f1 score = TP / (TP + FN)
    print('F1 score on testing data = {0:.3f}'.format(f1_score(y_test,y_test_pred)))
    
    
    
    #### information
    plot_feature_importances(treeClsf, X_train.columns.tolist())
    
    # get probabilities
    y_prob = treeClsf.predict_proba(X_test)
    result = pd.Series(y_prob[:,1], index = X_test.index)
    

    return result
    

    
    
    


    
    

if __name__ == '__main__':
    
    blight_model()