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


def par_score(chart, par, value):
    # chart = the PEWS chart corresponding to the age range
    # par = vital sign or observation parameter
    # value = value of the parameter

    # create a mini-DataFrame of parameter limits based on the age (chart) and parameter to be scored
    model = PEWS_model.loc[(PEWS_model['chart'] == chart) & (PEWS_model['par'] == par), ['lower', 'upper', 'score']]

    # iterate over the mini-DataFrame
    for index, row in model.iterrows():
        lower = row.lower
        upper = row.upper + 1

        # return the score if the parameter value is within the range of the limits
        if value in range(lower, upper):
            score = row.score
            return score


def cat_score(par, value):
    # chart = the PEWS chart corresponding to all age ranges (chart: 4)
    # par = vital sign or observation parameter
    # value = value of the parameter

    # create a mini-DataFrame of parameter limits based on the age (chart) and parameter to be scored
    model = PEWS_model.loc[(PEWS_model['chart'] == 4) & (PEWS_model['par'] == par), ['lower', 'upper', 'score']]

    if par in ['sats', 'receiving_o2']:

        # iterate over the mini-DataFrame
        for index, row in model.iterrows():
            lower = row.lower
            upper = row.upper + 1

            # return the score if the parameter value is within the range of the limits
            if value in range(lower, upper):
                score = row.score
                return score

    elif par in ['concern', 'WoB']:

        # iterate over the mini-DataFrame
        for index, row in model.iterrows():
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
    PEWS_bins = [0, 1, 5, 12, 18]   # Age bins according to PEWS chart categories
    charts = [0, 1, 2, 3]           # age ranges are 0: 0-11m, 1: 1-4y, 2: 5-11y, 3: >12y

    # add a chart column to the Dataframe and classify age according to PEWS model age ranges
    PEWS_df['chart'] = pd.cut(PEWS_df.age, PEWS_bins, labels=charts)

    # define the parameter list to score
    parameter_list = ['HR', 'RR', 'sBP']
    category_list = ['concern', 'sats', 'receiving_o2']

    # iterate through the parameter list to calculate the PEWS score for that parameter
    for par in PEWS_df.columns:
        if par in parameter_list:
            print(f'\n...Calculating scores for {par}...')
            PEWS_df[par+'_'+model] = PEWS_df.apply(lambda row: par_score(row['chart'], par, row[par]), axis=1)

        elif par in category_list:
            print(f'\n...Calculating scores for {par}...')
            PEWS_df[par + '_' + model] = PEWS_df.apply(lambda row: cat_score(par, row[par]), axis=1)

        else:
            print(f'\n** Skipping past {par} **')

    return PEWS_df


""" UHL Nervecentre PEWS model """

class PEWS(object):

    """
    age ranges are 0-11m, 1-4y, 5-11y, >12y
    scores are calculated like this: (based on paper chart scoring convention)
        (value >= lower) & (value <= upper) = score as per score for that range
    columns are: 'chart', 'parameter', 'lower', 'upper', 'score'
    lower = lower limit, upper = upper limit
    """

    limits = []

    def __init__(self):
        self.limits = [

            [0, 'HR', 201, 310, 1],
            [0, 'HR', 161, 200, 1],
            [0, 'HR', 91, 160, 0],
            [0, 'HR', 61, 90, 1],
            [0, 'HR', -1, 60, 1],

            [1, 'HR', 181, 310, 1],
            [1, 'HR', 141, 180, 1],
            [1, 'HR', 91, 140, 0],
            [1, 'HR', 51, 90, 1],
            [1, 'HR', 20, 50, 1],

            [2, 'HR', 161, 250, 1],
            [2, 'HR', 121, 160, 1],
            [2, 'HR', 71, 120, 0],
            [2, 'HR', 45, 70, 1],
            [2, 'HR', 20, 44, 1],

            [3, 'HR', 101, 250, 1],
            [3, 'HR', 61, 100, 0],
            [3, 'HR', 20, 60, 1],

            [0, 'RR', 61, 110, 1],
            [0, 'RR', 31, 60, 0],
            [0, 'RR', 21, 30, 1],
            [0, 'RR', 11, 20, 1],
            [0, 'RR', 0, 10, 1],

            [1, 'RR', 51, 80, 1],
            [1, 'RR', 41, 50, 1],
            [1, 'RR', 21, 40, 0],
            [1, 'RR', 17, 20, 1],
            [1, 'RR', 11, 16, 1],
            [1, 'RR', 0, 10, 1],

            [2, 'RR', 41, 70, 1],
            [2, 'RR', 31, 40, 1],
            [2, 'RR', 21, 30, 0],
            [2, 'RR', 15, 20, 1],
            [2, 'RR', 9, 14, 1],
            [2, 'RR', 0, 8, 1],

            [3, 'RR', 25, 60, 1],
            [3, 'RR', 21, 25, 1],
            [3, 'RR', 11, 20, 0],
            [3, 'RR', 7, 10, 1],
            [3, 'RR', 0, 6, 1],

            [0, 'sBP', 110, 250, 0],
            [0, 'sBP', 71, 109, 0],
            [0, 'sBP', 61, 70, 0],
            [0, 'sBP', 51, 60, 0],
            [0, 'sBP', 30, 50, 0],

            [1, 'sBP', 111, 250, 0],
            [1, 'sBP', 81, 110, 0],
            [1, 'sBP', 71, 80, 0],
            [1, 'sBP', 61, 70, 0],
            [1, 'sBP', 30, 60, 0],

            [2, 'sBP', 121, 250, 0],
            [2, 'sBP', 91, 120, 0],
            [2, 'sBP', 81, 90, 0],
            [2, 'sBP', 71, 80, 0],
            [2, 'sBP', 30, 70, 0],

            [3, 'sBP', 151, 250, 0],
            [3, 'sBP', 101, 150, 0],
            [3, 'sBP', 76, 100, 0],
            [3, 'sBP', 30, 75, 0],

            [4, 'receiving_o2', 24, 100, 1],
            [4, 'receiving_o2', 20, 23, 0],
            [4, 'receiving_o2', -1, 19, 1],

            [4, 'sats', 94, 100, 0],
            [4, 'sats', 30, 93, 0],

            [4, 'WoB', '', 'stridor', 1],
            [4, 'WoB', '', 'grunting', 1],
            [4, 'WoB', '', 'severe', 1],
            [4, 'WoB', '', 'moderate', 1],
            [4, 'WoB', '', 'mild', 0],
            [4, 'WoB', '', 'none', 0],

            [4, 'concern', '', 'No', 0],
            [4, 'concern', '', 'Nurse concern', 1],
            [4, 'concern', '', 'Parent concern', 1],

            [4, 'AVPU', '', 'unresponsive', 1],
            [4, 'AVPU', '', 'pain', 1],
            [4, 'AVPU', '', 'voice', 1],
            [4, 'AVPU', '', 'alert', 0],
            [4, 'AVPU', '', 'asleep', 0],

            [4, 'temp', 28.0, 36.0, 0],
            [4, 'temp', 36.1, 38.4, 0],
            [4, 'temp', 38.5, 42.0, 0],

            [4, 'cap_refill', '', '0-2 secs', 0],
            [4, 'cap_refill', '', '3-4 secs', 0],
            [4, 'cap_refill', '', '5-8 secs', 0],
            [4, 'cap_refill', '', 'grey/mottled', 0],

            [4, 'pain', '', 'no', 0],
            [4, 'pain', '', 'mild', 0],
            [4, 'pain', '', 'moderate', 0],
            [4, 'pain', '', 'severe', 0],
            [4, 'pain', '', 'excruciating', 0],

        ]

    pass


# instantiate the PEWS model DataFrame
PEWS_model = pd.DataFrame(PEWS().limits, columns=['chart', 'par', 'lower', 'upper', 'score'])
# print(PEWS_model)


""" Sequential Function Call """

PEWS_df = load_saved_data('PEWS_data_clean')
# NPEWS_limits = load_saved_data('NPEWS_limits')
# UHL_PEWS_limits = load_saved_data('UHL_PEWS_limits')

explore_data(PEWS_df)

explore_data(calculate_PEWS(PEWS_df, 'PEWS'))

print(PEWS_df.loc[(PEWS_df['chart'] != 0)&(PEWS_df['chart'] != 1)&(PEWS_df['chart'] != 2)&(PEWS_df['chart'] != 3)&(PEWS_df['chart'] != 4) ])


exit()

