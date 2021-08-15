"""
    NAME:          PEWSDataAnalysis/PEWS_scores.py
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.1
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DETAILS:       Collection of functions for calculating the PEWS score for a selection of parameters
    DEPENDENCIES:  This program requires the following modules:
                    Numpy, Pandas
"""

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas

# Import the PEWS models for calculating scores
import PEWS_models as pm


""" Load Data Files """


def load_synthetic_data():
    # function to import Synthetic Observations dataset
    raw_df = pd.read_csv('Data/synthetic_obs.csv')
    return raw_df


def load_saved_data(filename):
    # function to load previously processed and saved PEWS data (using PEWS_data_clean.py)
    df = pd.read_csv(f'data/{filename}.csv', header='infer', index_col=0)

    return df


""" Data Explore """


def explore_data(df):
    # function to explore the raw dataframe
    print('\n*** Displaying DataFrame Summary ***\n')

    # set pandas options to display all columns in a DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # explore and examine the DataFrame
    print(df.info(verbose=True, memory_usage='deep'))
    print('\n')
    print(df.describe())
    print('\n')
    print(df.head(50))
    print('\n')
    return(df)


""" Calculate PEWS scores for different models """


def par_score(PEWS_model, chart, par, value):
    # Function to calculate the scores for continuous parameters
    # chart = the PEWS chart corresponding to the age range
    # par = vital sign or observation parameter
    # value = value of the parameter

    # Select the correct PEWS model
    if PEWS_model == 'UPEWS':
        model = pm.UPEWS_model
    elif PEWS_model == 'NPEWS':
        model = pm.NPEWS_model

    # create a mini-DataFrame of parameter limits based on the age (chart) and parameter to be scored
    limits = model.loc[(model['chart'] == chart) & (model['par'] == par), ['lower', 'upper', 'score']]

    # iterate over the mini-DataFrame
    for index, row in limits.iterrows():
        lower = row.lower
        upper = row.upper + 1

        # return the score if the parameter value is within the range of the limits
        if value in range(lower, upper):
            score = row.score
            return score


def cat_score(PEWS_model, par, value):
    # Function to calculate the scores for categorical parameters
    # chart = the PEWS chart corresponding to all age ranges (chart: 4)
    # par = vital sign or observation parameter
    # value = value of the parameter

    # Select the correct PEWS model
    if PEWS_model == 'UPEWS':
        model = pm.UPEWS_model
    elif PEWS_model == 'NPEWS':
        model = pm.NPEWS_model

    # create a mini-DataFrame of parameter limits based on the age (chart) and parameter to be scored
    limits = model.loc[(model['chart'] == 4) & (model['par'] == par), ['lower', 'upper', 'score']]

    if par in ['sats', 'receiving_o2']:

        # iterate over the mini-DataFrame
        for index, row in limits.iterrows():
            lower = row.lower
            upper = row.upper + 1

            # return the score if the parameter value is within the range of the limits
            if value in range(lower, upper):
                score = row.score
                return score

    elif par in ['concern', 'WoB']:

        # iterate over the mini-DataFrame
        for index, row in limits.iterrows():
            upper = row.upper

            # return the score if the parameter value matches the score category
            if value == upper:
                score = row.score
                return score

    else:
        print(f'\n* Error: {par} is not in the PEWS model')
        score = 0
        return score


def calculate_PEWS(PEWS_df, model):
    # function to calculate PEWS scores

    #  bin ages by chart age ranges
    bins = [0, 1, 5, 12, 18]   # Age bins according to PEWS chart categories
    charts = [0, 1, 2, 3]           # age ranges are 0: 0-11m, 1: 1-4y, 2: 5-11y, 3: >12y

    # add a chart column to the Dataframe and classify age according to PEWS model age ranges
    PEWS_df['chart'] = pd.cut(PEWS_df.age, bins=bins, labels=charts)

    # define the parameter list to score
    parameter_list = ['HR', 'RR', 'sBP']
    category_list = ['concern', 'sats', 'receiving_o2']

    # iterate through the parameter list to calculate the PEWS score for that parameter
    for par in PEWS_df.columns:
        if par in parameter_list:
            print(f'\n...Calculating scores for {par}...')
            PEWS_df[par+'_'+model] = PEWS_df.apply(lambda row: par_score(model, row['chart'], par, row[par]), axis=1)

        elif par in category_list:
            print(f'\n...Calculating scores for {par}...')
            PEWS_df[par + '_' + model] = PEWS_df.apply(lambda row: cat_score(model, par, row[par]), axis=1)

        else:
            print(f'\n** Skipping past {par} **')

    return PEWS_df


""" Sequential Function Call """

PEWS_df = load_saved_data('PEWS_data_clean')
# NPEWS_limits = load_saved_data('NPEWS_limits')
# UHL_PEWS_limits = load_saved_data('UHL_PEWS_limits')

# explore_data(PEWS_df)
#
# Calculate UHL PEWS and explore data
# explore_data(calculate_PEWS(PEWS_df, 'UPEWS'))
#
# Calculate National PEWS and explore data
# explore_data(calculate_PEWS(PEWS_df, 'NPEWS'))
#
# print(PEWS_df.loc[(PEWS_df['chart'] != 0)&(PEWS_df['chart'] != 1)&(PEWS_df['chart'] != 2)&(PEWS_df['chart'] != 3)&(PEWS_df['chart'] != 4) ])


exit()

