"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    Numpy, Pandas
"""

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas


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
        'RR', 'sats', 'receiving_o2',
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


def replace_nan(df):
    # replaces Nan values
    # assumes blank value for concern = no concern
    # assumes blank value for receiving_o2 = breathing Air
    # assumes blank value for cap_refill = brisk response (<3sec)
    df['concern'].fillna('No', inplace=True)
    df['receiving_o2'].fillna('21', inplace=True)
    df['cap_refill'].fillna('0-2 secs', inplace=True)
    return df


def clean_receiving_o2_data(df):
    #  cleans the receiving_O2 data column
    # assumes NaN or nonsensical values to be Air (replace_nan())
    # final values: 21 = Air
    # final values: any number > 21 is % FiO2
    # final values: any number < 21 is in L/min
    print('\n...Cleaning \"Receiving O2\" Data...')

    # print(sorted(df['receiving_o2'].unique().tolist()))

    #  first convert to string so regex will work
    df['receiving_o2'] = df['receiving_o2'].astype(str)
    # replace Air with 21
    df['receiving_o2'] = df['receiving_o2'].replace('Air', '21')
    # replace @ = or ' with 21
    df['receiving_o2'] = df['receiving_o2'].replace('[=@\']', '21', regex=True)

    # removes L from values
    df['receiving_o2'] = df['receiving_o2'].replace('[lL.]$', '', regex=True)

    # replaces 00. with 0.0
    df['receiving_o2'] = df['receiving_o2'].replace('^00.', '0.0', regex=True)
    # replace 00 with 0.0
    df['receiving_o2'] = df['receiving_o2'].replace('^00', '0.0', regex=True)

    # convert values back to float numbers
    df['receiving_o2'] = df['receiving_o2'].astype(float)

    # print(sorted(df['receiving_o2'].unique().tolist()))
    print('\n...Receiving O2 Data cleaning complete...')
    return df


def clean_categorical_data(df):
    # takes the dataframe and converts categorical data parameters to codes
    # categorical parameters are listed in cat_list
    print('\n...Processing Categorical Data...')
    cat_list = ['concern', 'cap_refill']

    for cat in cat_list:
        df[cat] = df[cat].astype('category')
        # print(df.dtypes)
        # df[f'{cat}_cat'] = df[cat].cat.codes
        # print(df.head())

    print('\n...Categorical Data cleaning complete...')
    return df


def clean_continuous_data(df):
    # takes the dataframe and converts continuous data parameters to
    # continuous parameters are listed in par_list
    # any text strings are converted to NaN values
    # NaN values are removed
    print('\n...Processing Continuous Data...')
    par_list = ['age', 'RR', 'sats',  'HR', 'sBP', 'temp']

    for par in par_list:
        df[par] = pd.to_numeric(df[par], errors='coerce')
        print(f'\n    Count of {par} NaN to delete: ', df[par].isna().sum())
        df.dropna(subset=[par], inplace=True)
        print(f'\n    Final count of {par} NaN: ', df[par].isna().sum())

    df.reset_index(drop=True, inplace=True)
    print('\n...Continuous Data cleaning complete...')
    return df


def list_unique_values(df):
    for column in list(df.columns.values):
        # if (df[column].dtype == object ):
        print(df[column].unique().tolist())
    return df



""" Sequential Function Call """
# use this to load the PEWS sharepoint files, select the relevant columns and save locally as a csv file for quick access

raw_df = load_sharepoint_file(file_scope='full')
# explore_data(raw_df)
PEWS_df = select_PEWS_data_columns(raw_df)
PEWS_df = calculate_age(PEWS_df)
PEWS_df = convert_to_decimal_age(PEWS_df)

PEWS_df = split_BP(PEWS_df)
PEWS_df = replace_nan(PEWS_df)
# explore_data(PEWS_df)

PEWS_df = clean_receiving_o2_data(PEWS_df)
PEWS_df = clean_categorical_data(PEWS_df)
PEWS_df = clean_continuous_data(PEWS_df)

explore_data(PEWS_df)
# list_unique_values(PEWS_df)
# save_as_csv(PEWS_df, 'PEWS_data_clean')


""" pipe doesn't seem to work for this """

# raw_df = load_sharepoint_file(file_scope='half')
# PEWS_df = select_PEWS_data_columns(raw_df)
# process = (
#     calculate_age(PEWS_df)
#         .pipe(calculate_age)
#         .pipe(convert_to_decimal_age)
#
#         .pipe(split_BP)
#         .pipe(replace_nan)
#
#         .pipe(clean_receiving_o2_data)
#         .pipe(clean_categorical_data)
#         .pipe(clean_continuous_data)
#             )
# explore_data(PEWS_df)
# list_unique_values(PEWS_df)




exit()

