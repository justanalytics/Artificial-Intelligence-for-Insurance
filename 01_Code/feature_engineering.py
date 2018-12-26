## Description: This code is used to derive all the features required by the machine learning model. These features are extracted directly from the data. Readme.md file has the list of all the features that are to be used as part of this model.
## Created By: Aditya Mohan Arasanipalai (aditya.am@justanalytics.com)
## Created On: 26th November, 2018

## Importing all the required libraries
import os
import sys
import pandas as pd
import pyodbc # For importing data from the sql database
from configparser import SafeConfigParser
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
date_format = "%Y-%m-%d"
date_format_pol = "%Y%m%d"

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Logger
logger = logging.getLogger('claim_ml_log.feateng')

def randForest(feat_df):

    # Drop columns not used for machine learning
    feat_df = feat_df.drop(['CUSTOMER_NUMBER', 'CUSTOMER_DOB', 'CLAIM_NUMBER', 'POLICY_NUMBER',
                                    'CURRFROM','CURRTO','STATDATE',
                                        'OCCDATE','CCDATE'], axis=1)
    inputs = 19

    # Re-arrange columns
    cols = list(feat_df.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('REJECT_CODE')) #Remove b from list
    feat_df = feat_df[cols+['REJECT_CODE']] #Create new dataframe with columns in the order you want

    feature = np.array(feat_df.values[:,0:inputs])
    label = np.array(feat_df.values[:,-1])

    # RandomForest to calculate the importance of the features
    # Create decision tree classifer object
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    inputs = 19

    # Train model
    model = clf.fit(feature, label)
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feat_df.columns[i] for i in indices]

    # Create plot
    plt.figure()

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(feature.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(feature.shape[1]), names, rotation=90)

    # Show plot
    plt.show()

    return True


def deriveFeatures(trans_data):

    logger.info("In Feature Engineering Script")

    '''
    Takes one argument: Raw Data Frame: trans_data
    Returns one dataframe: trans_data
    '''

    # Get all unique customers
    un_customers = trans_data.CUSTOMER_NUMBER.unique()
    feat_df = pd.DataFrame()

    if os.path.isfile('features.csv'):
        
        logger.warn('Features file already exists. Skipping recalculation of features.')
        feat_df = pd.read_csv('features.csv')
        feat_df = feat_df.dropna()
        randForest(feat_df)
        return feat_df

    else:

        for ea_customer in un_customers: # 86315 unique customers
            
            all_rec_ea_customer = trans_data[trans_data['CUSTOMER_NUMBER'] == ea_customer]

            # Before Value
            last_claim_value = 0
            time_since_last_claim = 0
            avg_claim_value = 0
            rej_ratio = 0
            total_claims = 0
            time_policy_exp = 0

            for index, ea_rec_cust in all_rec_ea_customer.iterrows():

                total_claims = total_claims + 1
                avg_claim_value = (avg_claim_value + float(ea_rec_cust['FAMOUT']))/total_claims
                
                if(time_since_last_claim == 0):
                    time_since_last_claim = 0
                else:
                    time_since_last_claim = datetime.strptime(str(ea_rec_cust['REGISTRATION_DT']), date_format) -                                    datetime.strptime(time_since_last_claim, date_format)
                    time_since_last_claim = time_since_last_claim.days
                
                if(ea_rec_cust['REJECT_CODE'] == 'NON' or ea_rec_cust['REJECT_CODE'] == 'NOR' or ea_rec_cust                        ['REJECT_CODE'] == 'EXC'):
                    rej_ratio = rej_ratio + 1

                if(ea_rec_cust['CURRTO'] != 99999999):
                    time_policy_exp = datetime.strptime(str(ea_rec_cust['CURRTO']), date_format_pol) -                                       datetime.strptime(str(ea_rec_cust['CURRFROM']), date_format_pol)
                    time_policy_exp = time_policy_exp.days
                else:
                    time_policy_exp = 0

                ea_rec_cust['TIME_POLICY_EXP'] = time_policy_exp
                ea_rec_cust['REJ_RATIO'] = rej_ratio
                ea_rec_cust['TIME_SINCE_LAST_CLAIM'] = time_since_last_claim
                ea_rec_cust['AVG_CLAIM_VALUE'] = avg_claim_value
                ea_rec_cust['TOTAL_CLAIMS'] = total_claims
                ea_rec_cust['LAST_CLAIM_VALUE'] = last_claim_value

                # Post assignment - Overwrite all values
                last_claim_value = float(ea_rec_cust['FAMOUT'])
                time_since_last_claim = ea_rec_cust['REGISTRATION_DT']

                # Append the dataframe
                feat_df = feat_df.append(ea_rec_cust, ignore_index=True) 
        
        # Converting all dates to required formats
        for each_df_val in range(0,len(feat_df)):
            
            feat_df['REGISTRATION_DT'][each_df_val] = str(datetime.strptime((feat_df['REGISTRATION_DT'][each_df_val]), date_format).month) + str(datetime.strptime((feat_df['REGISTRATION_DT'][each_df_val]), date_format).day)
            
        feat_df['REGISTRATION_DT'] = pd.Categorical(feat_df.REGISTRATION_DT).codes

        # Write features to a CSV file/table so that they are not recomputed everytime.
        feat_df.to_csv("features.csv", index=False)

        # Analyzing the importance of the variables which would be used as part of machine learning
        randForest(feat_df)

        return feat_df # Returning

    
# def main():
#     cust_trans_data, claim_trans_data = deriveFeatures(cust_trans_data, claim_trans_data)

# if __name__ == "__main__":
#     main()