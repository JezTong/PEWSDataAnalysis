"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    Pandas
"""

import pandas as pd
import PEWS_models as PT

""" Import Synthetic Observations dataset """

df = pd.read_csv('Data/synthetic_obs.csv')

""" Data Exploring """
print('\nDisplaying DataFrame Summary:\n')

# set pandas options to display all columns in a DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# explore and examine the DataFrame
print(df.describe())
# print(df.head(10))

# print('\nDisplaying DataFrame column headers and data types:\n')
# print(df.dtypes)
# print('\n')

""" Calculate PEWS scores for different models """

# activate PEWS threshold

""" Calculate the centiles """

# take in a DataFrame of the parameter in question

# create a new column for the centile

# select an an age band 'window'

# take the window and work out its mean and standard deviation

# calculate the sample size for this window (determine what the minimum sample size should be)

# calculate the centile for each age using the 'window' with 'acceptable limits'