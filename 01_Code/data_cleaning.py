## Description: This code is used to fetch all the data required and perform various cleaning operations.
## Created By: Aditya Mohan Arasanipalai (aditya.am@justanalytics.com)
## Created On: 26th November, 2018

## Importing all the required libraries
import os
import sys
import pandas as pd
import pyodbc # For importing data from the sql database
from configparser import SafeConfigParser
import logging

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Logger
logger = logging.getLogger('claim_ml_log.dataclean')

logger.info("In Data Cleaning Script")
def data_clean(trans_data):

    '''
    Function takes one argument
    Return one dataframe (cleaned data)
    '''

    print(trans_data.describe())
    print(trans_data.dtypes)

    # Data type conversions
    trans_data['CLAIM_TYPE_CODE'] = pd.Categorical(trans_data.CLAIM_TYPE_CODE).codes
    trans_data['CUSTOMER_GENDER'] = pd.Categorical(trans_data.CUSTOMER_GENDER).codes
    trans_data['COVERAGE'] = pd.Categorical(trans_data.COVERAGE).codes
    trans_data['Is_PO'] = pd.Categorical(trans_data.Is_PO).codes
    trans_data['LIFE'] = pd.Categorical(trans_data.LIFE).codes
    trans_data['Marrige_Code'] = pd.Categorical(trans_data.Marrige_Code).codes
    trans_data['Occupation_Code'] = pd.Categorical(trans_data.Occupation_Code).codes
    trans_data['PRODUCT_CODE'] = pd.Categorical(trans_data.PRODUCT_CODE).codes

    # Mapping reject codes NON, NOR, EXC to FRAUD
    trans_data['REJECT_CODE'] = trans_data.REJECT_CODE.map({"NON":"FRAUD","NOR":"FRAUD","EXC":"FRAUD"})
    trans_data['REJECT_CODE'] = trans_data['REJECT_CODE'].fillna("NOFRAUD")
    trans_data['REJECT_CODE'] = pd.Categorical(trans_data.REJECT_CODE).codes

    trans_data['RIDER'] = pd.Categorical(trans_data.RIDER).codes
    trans_data['FAMOUT'] = trans_data['FAMOUT'].astype(float)
    trans_data['PREMIUM'] = trans_data['PREMIUM'].astype(float)

    logger.info("All data has been cleaned.")

    return trans_data
