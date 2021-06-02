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

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt  # pip install matplotlib
import seaborn as sns  # pip install seaborn
import statsmodels.api as sm  # pip install statsmodels
import statsmodels.formula.api as smf

# Import PEWS models
import PEWS_models as PM

""" Load Data Files """


def load_sharepoint_file(file_scope='full'):
    # function to load PEWS data file from Sharepoint account
    # file_scope: 'half' = limited (faster), 'full' = load full database

    # code to access data files on Sharepoint
    import File_Access as FA

    if file_scope == 'half':
        # load 2 of 4 data files on Sharepoint
        PEWS_df = FA.load_file('PEWS_Data_1.xlsx')
        HISS_df = FA.load_file('HISS_Data_1.xlsx')

    else:
        # Load all 4 data files on Sharepoint
        PEWS_df_1 = FA.load_file('PEWS_Data_1.xlsx')
        PEWS_df_2 = FA.load_file('PEWS_Data_2.xlsx')
        PEWS_df = pd.concat([PEWS_df_1, PEWS_df_2])

        HISS_df_1 = FA.load_file('HISS_Data_1.xlsx')
        HISS_df_2 = FA.load_file('HISS_Data_2.xlsx')
        HISS_df = pd.concat([HISS_df_1, HISS_df_2])

    # Merge the PEWS and HISS Data files
    print('\nMerging Data Files...')
    df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')
    return df


def load_synthetic_data():
    # function to import Synthetic Observations dataset
    df = pd.read_csv('Data/synthetic_obs.csv')
    return df


def load_saved_data():
    # function to load previously processed and saved data (rapid file load for code development)
    df = pd.read_csv('data/PEWS_data.csv')
    # df.rename(columns={'age': 'age_in_days'}, inplace=True)
    return df


""" Initial Data Explore """


def explore_data(df):
    # function to explore the raw dataframe
    print('\nDisplaying DataFrame Summary:\n')

    # set pandas options to display all columns in a DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # explore and examine the DataFrame
    print(df.describe())
    print(df.info(verbose=True, memory_usage='deep'))
    print(df.head(10))
    print('\n')
    return(df)

""" Select the columns used to calculate the PEWS score """

def select_PEWS_data_columns(df):
    PEWS_df = df[[
        'age_in_days',
        'concern',
        'RR', 'sats', 'receiving_o2', 'receiving_o2_units',
        'HR', 'BP', 'cap_refill',
        'temp', 'ACVPU',
    ]]
    print('\nNPEWS DataFrame created from PEWS Data Files ...')
    return PEWS_df


""" Data cleaning """

def convert_decimal_age(PEWS_df):
    # converts the age in days to a decimal age
    PEWS_df['age'] = PEWS_df['age_in_days']/365.25
    PEWS_df = PEWS_df.drop(['age_in_days'], axis=1)
    print('\n...Converted age in days to decimal age...')
    # print(PEWS_df.head(10))
    return PEWS_df

""" Calculate PEWS scores for different models """

# activate PEWS threshold

""" Save files """


def save_as_csv(df):
    # saves the PEWS dataframe as a csv file for quick analysis later
    df.to_csv(f'data/PEWS_data.csv')
    return PEWS_df


""" Sequential Function Call """
# use this to load the PEWS sharepoint files, select the relevant columns and save locally as a csv file for quick access
df = load_sharepoint_file(file_scope='half')

process = (
    explore_data(df)
        .pipe(select_PEWS_data_columns)
        .pipe(convert_decimal_age)
        .pipe(explore_data)
        .pipe(save_as_csv())

)

# .pipe(save_as_csv)

exit()

# use this for analysing files on Sharepoint
# parameter_list = ['HR', 'RR', 'BP']
# for parameter in parameter_list:
#     # takes the dataframe and processes in sequence
#     df = load_sharepoint_file(file_scope='half')
#     process = (
#
#         select_parameter(df, parameter)
#             .pipe(split_BP)
#             .pipe(clean_data)
#             .pipe(convert_decimal_age)
#             .pipe(print_data)
#             .pipe(plot_scatter)
#             .pipe(poly_quantile_regression)
#             .pipe(save_as_csv)
#     )