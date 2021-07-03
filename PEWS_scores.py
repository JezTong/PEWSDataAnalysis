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


def load_synthetic_data():
    # function to import Synthetic Observations dataset
    raw_df = pd.read_csv('Data/synthetic_obs.csv')
    return raw_df


def load_saved_data(filename):
    # function to load previously processed and saved PEWS data (using PEWS_data_clean.py)
    df = pd.read_csv(f'data/{filename}.csv', header='infer', index_col=None)

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

""" Calculate PEWS scores for different models """


# def check_age():
#     # work out what age range
#
# def check_parameter():
#     # work out which parameter to calculate
#
#
# def score_it(parameter, value):
#     if value > NPEWS_limits.lower_lim & value < upper_lim:
#     score = score

#


def get_parameter_limit():
    # looks up the lower limit for scoring range in the model table
    limit_row = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == '0-11m') &
        (UHL_PEWS_limits['parameter'] == 'HR') &
        (UHL_PEWS_limits['score'] == 0)
    ]
    lower = int(limit_row['lower_lim'].item())
    upper = int(limit_row['upper_lim'].item())
    return lower, upper

def calculate_score(PEWS_df):
    # create a new scored dataframe
    PEWS_scored = PEWS_df
    # create the scoring columns
    upper_lim = get_parameter_limit()[1]
    lower_lim = get_parameter_limit()[0]


    #  bin ages by chart age ranges
    PEWS_bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
    PEWS_bin_labels = ['0-11m', '1-4y', '5-11y', '>12y']  # Age bin category labels

    # classify age according to age bins and add an age_bin column to the PEWS_scored Dataframe
    PEWS_scored['age_bin'] = pd.cut(PEWS_scored.age, PEWS_bins, labels=PEWS_bin_labels)


    filters = [
        (PEWS_df.age_bin == '0-11m') & (PEWS_df.HR > lower_lim) & (PEWS_df.HR < upper_lim),         # score = 0
        (PEWS_df.age_bin == '0-11m') & ((PEWS_df.HR < lower_lim) | (PEWS_df.HR > upper_lim))      # score = 1
               ]

    values = [0, 1]

    # apply filters to calculate the scores
    PEWS_df['HR_score'] = np.select(filters, values, default='error')

    print(PEWS_scored.head(50))
    print(PEWS_scored.describe())

    return PEWS_scored




""" Sequential Function Call """

PEWS_df = load_saved_data('PEWS_data_clean')
NPEWS_limits = load_saved_data('NPEWS_limits')
UHL_PEWS_limits = load_saved_data('UHL_PEWS_limits')

calculate_score(PEWS_df)



""" Test code """


def get_upper_limit():
    upper_lim_row = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == '0-11m') &
        (UHL_PEWS_limits['parameter'] == 'HR') &
        (UHL_PEWS_limits['score'] == 0)
    ]
    upper_limit = int(upper_lim_row['upper_lim'].item())
    return upper_limit

upper_lim = UHL_PEWS_limits.loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0)]['upper_lim'].values.astype(int)

print(upper_lim)

upper_lim_row = UHL_PEWS_limits.loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0)]

upper_lim_2 = int(upper_lim_row['upper_lim'].item())
print(upper_lim_2)
# print(upper_lim_2.type())


lower_lim = UHL_PEWS_limits['lower_lim'].loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0)].astype(int)

print(lower_lim)

HR_limits = UHL_PEWS_limits.loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0), ['upper_lim', 'lower_lim']].reset_index(drop=True)

print(HR_limits)






exit()

# for parameter in NPEWS_limits['parameter'].unique():
#     # print(parameter)
#     if parameter in PEWS_df:
#         print('ok')
#         PEWS_scored[f'{parameter}_score'] = PEWS_scored[f'{parameter}']


""" Bin Data by age """

PEWS_bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
PEWS_bin_labels = ['0-11m', '1-4y', '5-11y', '>12y']  # Age bin category labels

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
df['PEWS_bins'] = pd.cut(df.age, PEWS_bins, labels=PEWS_bin_labels)







PEWS_scored = calculate_score(PEWS_df, model='UHL_PEWS')
explore_data(PEWS_scored)

for parameter in NPEWS_limits['parameter'].unique():
    print(parameter)


def list_unique_values(df):
    for column in list(df.columns.values):
        # if (df[column].dtype == object ):
        print(df[column].unique().tolist())
    return df




