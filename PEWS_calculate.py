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
import datetime as dt

# Import PEWS models
# import PEWS_models as PM

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
    raw_df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')
    print('\n...PEWS data files merged...')
    return raw_df


def load_synthetic_data():
    # function to import Synthetic Observations dataset
    raw_df = pd.read_csv('Data/synthetic_obs.csv')
    return raw_df


def load_saved_data():
    # function to load previously processed and saved data (rapid file load for code development)
    raw_df = pd.read_csv('data/PEWS_data.csv', header='infer', index_col=None)
    # raw_df.rename(columns={'age': 'age_in_days'}, inplace=True)
    return raw_df


""" Select the columns used to calculate the PEWS score """

def select_PEWS_data_columns(raw_df):
    # selects the relevant columns for calculating age and PEWS score
    # takes in the raw dataframe and returns a compact PEWS dataframe
    PEWS_df = raw_df.filter([
        'dob', 'obs_date',
        'concern',
        'RR', 'sats', 'receiving_o2', 'receiving_o2_units',
        'HR', 'BP', 'cap_refill',
        'temp'
    ], axis=1)
    PEWS_df = PEWS_df.reset_index(drop=True)
    print('\n...NPEWS DataFrame created from PEWS Data Files...')
    # print(PEWS_df.info())
    return PEWS_df
    # 'age_in_days',


""" Save files """


def save_as_csv(df, file_name):
    # saves the PEWS dataframe as a csv file for quick analysis later
    df.to_csv(f'data/{file_name}.csv')
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
    print(df.head(10))
    print('\n')
    return(df)

def list_unique_values(df):
    for column in list(df.columns.values):
        if (df[column].dtype == object ):
            print(df[column].unique().tolist())
    return df

""" Data cleaning """

def calculate_age(df):
    # calculates age in days based on obs date and date of birth
    # drops age_in_days and dob columns afterwards
    df['age_in_days'] = (df['obs_date'] - df['dob']).dt.days
    df = df.drop(['obs_date', 'dob'], axis=1)
    print('\n...Age calculated from dob and obs date...')
    # print(df.info())
    return df


def convert_to_decimal_age(df):
    # converts the age in days column to a decimal age column
    df['age'] = df['age_in_days']/365.25
    df = df.drop(['age_in_days'], axis=1)
    print('\n...Converted age in days to decimal age...')
    # print(df.head(10))
    return df


def drop_column(df, column_list):
    df = df.drop(column_list, axis=1)
    print('\n...Redundant columns deleted...\n')
    print(df.info())
    return df


def split_BP(df):
    # if the parameter is BP, splits the BP data into systolic BP and diastolic BP columns.
    # removes text from sBP column and replaces with np.NaN
    if 'BP' in df:
        BP = df['BP'].str.split('/', n=1, expand=True)
        df['sBP'] = BP[0]
        # df['dBP'] = BP[1]
        df.drop(columns=['BP'], inplace=True)
        df['sBP'] = df['sBP'].replace('\D+', np.NaN, regex=True)
        print(f'\n...Systolic BP extracted from BP column...')
        return df
    else:
        print('\n...No BP column to extract systolic BP from...')
        return df

def clean_data(df):
    # takes the parameter dataframe and converts text to NaN values
    # removes missing values
    # converts data types from objects to integers
    print('\n...Data cleaning in progress...')
    par_list = ['RR', 'sats', 'HR', 'sBP']

    for par in par_list:
        df[par] = pd.to_numeric(df[par], errors='coerce')
        print(f'\n    Count of {par} NaN to delete: ', df[par].isna().sum())
        df.dropna(subset=[par], inplace=True)
        print(f'\n    Final count of {par} NaN: ', df[par].isna().sum())

    df.reset_index(inplace=True)
    print('\n...Data cleaning in complete...')
    return df

def convert_dtypes(df):
    convert_dict = {
                    'RR': int,
                    'sats': int,
                    'HR': int,
                    'temp': float,

                    }

    df = df.astype(convert_dict)
    print('\n')
    print(df.dtypes)
    print('/nDatatypes converted...')
    return df

# 'concern': str,                    'cap_refill': str,
# 'receiving_o2': object,
# 'receiving_o2_units': object,
# 'BP': object,
# 'ACVPU': object

""" Calculate PEWS scores for different models """

# activate PEWS threshold


""" Sequential Function Call """
# use this to load the PEWS sharepoint files, select the relevant columns and save locally as a csv file for quick access

raw_df = load_sharepoint_file(file_scope='half')
# explore_data(raw_df)
PEWS_df = select_PEWS_data_columns(raw_df)
PEWS_df = calculate_age(PEWS_df)
PEWS_df = convert_to_decimal_age(PEWS_df)
# PEWS_df = drop_column(PEWS_df, column_list=['dob', 'obs_date'])
# explore_data(PEWS_df)
PEWS_df = split_BP(PEWS_df)
explore_data(PEWS_df)
PEWS_df = clean_data(PEWS_df)
explore_data(PEWS_df)


# process = (
#     explore_data(df)
#         .pipe(select_PEWS_data_columns)
#         .pipe(calculate_age)
#         .pipe(drop_column(df=PEWS_df, column_list=['dob', 'obs_date', 'age_in_days']))
#         .pipe(convert_decimal_age)
#         .pipe(explore_data)
#             )

# .pipe(save_as_csv(PEWS_df, file_name='PEWS_data'))
# use this for analysing the saved csv data file

# process = (
#     explore_data(df=load_saved_data())
#
#         .pipe(list_unique_values)
#
# )

# .pipe(convert_dtypes)


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