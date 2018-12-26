## Description: This code is the master script which runs the entire data and ml pipeline
## Created By: Aditya Mohan Arasanipalai (aditya.am@justanalytics.com)
## Created On: 23rd November, 2018

## Importing all the required libraries
import os
import sys
import pandas as pd
import pyodbc # For importing data from the sql database
from configparser import SafeConfigParser
import logging
import data_acquisition as da
import data_cleaning as dc
import feature_engineering as fe
import model_training as mt
import model_scoring as ms

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

def main(mode):

    # Creating the Logger
    logger = logging.getLogger("claim_ml_log")
    logger.setLevel(logging.DEBUG)

    # Creating the Handler for logging data to a file
    logger_handler = logging.FileHandler('claim_ml_log.log')
    logger_handler.setLevel(logging.DEBUG)

    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_handler)
    logger.info('Logger Configuration Complete!') #logger

    # Processing starts here
    # Get the data
    trans_data = da.get_data()

    # Clean the data
    trans_data = dc.data_clean(trans_data)

    # Feature Engineering
    feat_df = fe.deriveFeatures(trans_data)
    

    if(mode=='t'):
        # Model Training
        trained_model_stat = mt.modelTrain(feat_df)
        scored_model_stat = False
    else:
        # Model Scoring
        scored_model_stat = ms.modelScoreBatch(feat_df)
        trained_model_stat = False

    if(trained_model_stat == True or scored_model_stat == True):
        logger.info("Process Complete Successfully")
    else:
        logger.info("Process Failed")

if __name__ == "__main__":

    '''
    t - Training
    s - Scoring
    '''
    
    main(mode='t')