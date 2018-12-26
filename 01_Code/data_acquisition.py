## Description: This code is used to fetch all the data required for the machine learning training and validation process.
## Created By: Aditya Mohan Arasanipalai (aditya.am@justanalytics.com)
## Created On: 22nd November, 2018

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
logger = logging.getLogger('claim_ml_log.dataacq')

logger.info("In Data Acquisition Script")
def get_data():

    ''' 
    Function has no input arguments
    Returns one dataframe: trans_data
    '''

    # Importing and Parsing the config file
    cfg_file = './02_config/databaseconn.cfg'
    config = SafeConfigParser()
    config.read(cfg_file)

    logger.info('Reading configuration file.')
    hostname = str(config.get('connectiondetails','hostname'))
    username = str(config.get('connectiondetails','username'))
    password = str(config.get('connectiondetails','password'))
    database = str(config.get('connectiondetails','database'))

    ## Connecting and retrieving data from the database
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+hostname+';DATABASE='+database+';UID='+username+';PWD='+password)
    cur = conn.cursor()
    logger.info("Database connection established.")

    # Importing claiming transaction data from the database
    sql_statement = 'SELECT TOP(50000) * FROM dbo.ML_RawData ORDER BY 1,2'
    cur.execute(sql_statement)
    trans_data = pd.DataFrame.from_records(cur.fetchall())
    trans_data.columns = [column[0] for column in cur.description]
    logger.info("All data has been fetched.")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(trans_data.head()) # Checking whether the data has been fetched or not

    conn.close() # Closing the connection

    trans_data.to_csv("RawData.csv")
    
    return trans_data

# def main():
#     cust_trans_data, claim_trans_data = get_data()
#     print("All data fetched")

# if __name__ == "__main__":
#     main()