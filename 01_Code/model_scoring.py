## Description: This code is used to score the deep learning model with the data.
## Created By: Aditya Mohan Arasanipalai (aditya.am@justanalytics.com)
## Created On: 3rd December, 2018

## Importing all the required libraries
import os
import sys
import pandas as pd
import pyodbc # For importing data from the sql database
from configparser import SafeConfigParser
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
import numpy as np

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")
sys.__stdout__ = sys.stdout

# Logger
logger = logging.getLogger('claim_ml_log.mlscore')

def modelScoreBatch(feat_data):

    logger.info("In Deep Learning Model Scoring Script")

    '''
    Takes one argument: Processed DataFrame: feat_data
    Returns output dataframe

    Deep Learning Model Details:
    Inputs (Features): 18
    Outputs: 1

    '''
    try:

        feat_data = feat_data.drop(['CUSTOMER_NUMBER', 'CUSTOMER_DOB', 'CLAIM_NUMBER',                                              'POLICY_NUMBER','CURRFROM','CURRTO','STATDATE',
                                            'OCCDATE','CCDATE'], axis=1)
        inputs = 19

        # Re-arrange columns
        cols = list(feat_data.columns.values) #Make a list of all of the columns in the df
        cols.pop(cols.index('REJECT_CODE')) #Remove b from list
        feat_data = feat_data[cols+['REJECT_CODE']] #Create new dataframe with columns in the order you want

        # Machine Learning Implementation starts here
        feat_data.sort_values(by='REJECT_CODE', ascending=False, inplace=True)

        feature = np.array(feat_data.values[:,0:inputs])
        label = np.array(feat_data.values[:,-1])

        # Shuffle and split the data into train and test sets
        from sklearn.utils import shuffle
        shuffle_df = shuffle(feat_data, random_state=42)
        req_len = int(len(feat_data)*0.30)
        df_train = shuffle_df[0:req_len]
        df_test = shuffle_df[req_len:]
        logger.info("Divided the data into training and test sets.")

        train_feature = np.array(df_train.values[:,0:inputs])
        train_label = np.array(df_train.values[:,-1])
        test_feature = np.array(df_test.values[:,0:inputs])
        test_label = np.array(df_test.values[:,-1])

        # Standardize the features for speeding up deep learning (MinMaxScaler)
        scaler = MinMaxScaler()
        scaler.fit(train_feature)
        train_feature_trans = scaler.transform(train_feature)
        test_feature_trans = scaler.transform(test_feature)

        # Load JSON and Create Model
        json_file = open('claim_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # Load weights into new model
        loaded_model.load_weights("claim_model.h5")
        print("Loaded model from disk")
        
        # Evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        prediction = loaded_model.predict_classes(test_feature_trans)
        prediction_probs = loaded_model.predict_proba(test_feature_trans)

        # Display Predicted Outputs
        df_ans = pd.DataFrame({'Real Class' :test_label})
        df_ans['Prediction'] = prediction
        df_ans['PredictionProbabilities'] = prediction_probs
        df_ans[df_ans['Real Class'] != df_ans['Prediction']]
        print(df_ans['Prediction'].value_counts())
        print(df_ans['Real Class'].value_counts())

        cols = ['Real_Class_1','Real_Class_0']
        rows = ['Prediction_1','Prediction_0']

        B1P1 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
        B1P0 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
        B0P1 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class'] == 0)])
        B0P0 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 0)])

        conf = np.array([[B1P1,B0P1],[B1P0,B0P0]])
        df_cm = pd.DataFrame(conf, columns = [i for i in cols], index = [i for i in rows])

        f, ax= plt.subplots(figsize = (5, 5))
        sns.heatmap(df_cm, annot=True, ax=ax, fmt='d')
        ax.xaxis.set_ticks_position('top')

        # Joining the two dataframes and populating a final CSV
        test_cols = df_test.columns
        test_cols = list(test_cols)
        test_cols.append('Prediction')
        test_cols.append('PredictionProbabilities')
        df_test.reset_index(drop=True, inplace=True)
        df_ans.reset_index(drop=True, inplace=True)
        final_merged_df = pd.concat([df_test, df_ans['Prediction'], df_ans['PredictionProbabilities']], axis=1, sort=False,ignore_index = True)
        final_merged_df.columns = test_cols
        final_merged_df.to_csv("FinalActualPredicted.csv")
        
        return True

    except Exception as e:
        return False

def modelScoreRealtime(ip_data):

    '''
    In Progress
    '''

    return True