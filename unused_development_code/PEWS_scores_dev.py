"""
    NAME:          PEWSDataAnalysis - PEWS_scores.py
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



"""
This code is development code and does not work. It is used for testing concepts. 
Working code is stored in the main directory.
"""



# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas



""" Test code """






""" use this if decide to use class for getting thresholds """
# chart_list = ['chart_1', 'chart_2', 'chart_3', 'chart_4']
#
# chart_1 = pd.DataFrame(PEWS_model().chart_1, columns = ['age_range', 'lower', 'upper', 'score'])
# chart_2 = pd.DataFrame(PEWS_model().chart_1, columns = ['age_range', 'lower', 'upper', 'score'])
# chart_3 = pd.DataFrame(PEWS_model().chart_1, columns = ['age_range', 'lower', 'upper', 'score'])
# chart_4 = pd.DataFrame(PEWS_model().chart_1, columns = ['age_range', 'lower', 'upper', 'score'])
# print(chart_1)
# print(chart_2)

""" PEWS Model Dictonaries """
# Python Dictionary to store and recall limits and corresponding scores
# 'chart_1' = 0-11m, 'chart_2' = 1-4y, 'chart_3' = 5-12y, 'chart_4' = >13y

# chart_1 = {
#     'HR': {'lower': 91, 'upper': 159, 'score': 0},
#     'RR': {'lower': 31, 'upper': 59, 'score': 0}
# }
# PEWS = {
#     'chart_1': {
#         'age': [0, -1, 1],
#         'HR': [0, 91, 160],
#         'RR': [0, 31, 59]
#     },
#     'chart_2': {
#         'age': [0, 1, 5],
#         'HR': [0, 91, 140],
#         'RR': [0, 21, 40]
#     }
# }



# def calculate_EWS(PEWS_df):
#     #  bin ages by chart age ranges
#     PEWS_bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
#     chart_list = ['chart_1', 'chart_2', 'chart_3', 'chart_4']
#
#     # classify age according to age bins and add an age_bin column to the PEWS_scored Dataframe
#     PEWS_df['age_bin'] = pd.cut(PEWS_df.age, PEWS_bins, labels=chart_list)
#
#     parameter_list = ['HR']
#
#     temp_1 = pd.DataFrame()
#
#     for par in parameter_list:
#
#         score = PEWS['chart_1'][par][2]
#
#         score_it = lambda row: score if (row['age_bin'] == 'chart_1')  & (row[par] > PEWS['chart_1'][par][0]) & (row[par] < PEWS['chart_1'][par][1]) else ''
#
#         temp_1['PEWS_'+par] = temp_1.apply(score_it, axis=1)
#
#
#     return PEWS_df


# def calculate_score(PEWS_df):
#
#     #  bin ages by chart age ranges
#     PEWS_bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
#     chart_list = ['chart_1', 'chart_2', 'chart_3', 'chart_4']
#     # PEWS_bin_labels = ['chart_1', 'chart_2', 'chart_3', 'chart_4']  # Age bin category labels
#
#     # classify age according to age bins and add an age_bin column to the PEWS_scored Dataframe
#     PEWS_df['age_bin'] = pd.cut(PEWS_df.age, PEWS_bins, labels=chart_list)
#
#     temp_df_1 = PEWS_df[PEWS_df.age_bin == 'chart_1'].copy()
#     temp_df_2 = PEWS_df[PEWS_df.age_bin == 'chart_2']
#     temp_df_3 = PEWS_df[PEWS_df.age_bin == 'chart_3']
#     temp_df_4 = PEWS_df[PEWS_df.age_bin == 'chart_4']
#
#     df_list = [temp_df_1, temp_df_2, temp_df_3, temp_df_4]
#
#
#     score_it = lambda row: chart_1['HR'][2] if (row.HR > chart_1['HR'][0]) & (row.HR < chart_1['HR'][1]) else 1
#     temp_df_1['PEWS_HR'] = temp_df_1.apply(score_it, axis=1)


    # test = (temp_df_1.HR > chart_1['HR'][0]) & (temp_df_1.HR < chart_1['HR'][1])
    # score = chart_1['HR'][2]
    #
    # temp_df_1['PEWS_HR'] = np.where(test, score, '')

    # for df in df_list:
    #
    #     score = lambda row: 0 if  (row.HR >= 90) & (row.HR < get_upper_limit()) else 1
    #     df['PEWS_HR'] = df.apply(score, axis=1)
    #
    # PEWS_scored = pd.concat(df_list)
    # PEWS_scored = PEWS_scored.drop('Unnamed: 0', axis=1, inplace=True)





    # print(PEWS_scored.head(50))
    # print(PEWS_scored.describe())

    # return temp_df_1
def __init__(self):
    self.limits = [

        [1, 'age', -1, 1, 0],
        [2, 'age', 1, 5, 0],

        [1, 'HR', 201, 310, 1],
        [1, 'HR', 161, 200, 1],
        [1, 'HR', 91, 160, 0],
        [1, 'HR', 61, 90, 1],
        [1, 'HR', -1, 60, 1]
    ]

def get_upper_limit():
    # looks up the lower limit for scoring range in the model table
    limit_row = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == '0-11m') &
        (UHL_PEWS_limits['parameter'] == 'HR') &
        (UHL_PEWS_limits['score'] == 0)
    ]
    upper = int(limit_row['upper_lim'].values.item())
    return upper


def get_parameter_limit(age_bin):
    # looks up the lower limit for scoring range in the model table
    limit_row = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == age_bin) &
        (UHL_PEWS_limits['parameter'] == 'HR') &
        (UHL_PEWS_limits['score'] == 0)
        ]
    lower = int(limit_row['lower_lim'].values.item())
    upper = int(limit_row['upper_lim'].values.item())
    return lower, upper


def get_parameter_bins(age_bin, parameter):
    parameter_bins = UHL_PEWS_limits.loc[
                         (UHL_PEWS_limits['age_range'] == age_bin) &
                         (UHL_PEWS_limits['parameter'] == parameter)
                         ]['upper_lim'].values.astype(int).tolist() + [0]

    parameter_bins.reverse()
    print(parameter_bins)
    return parameter_bins


def get_parameter_scores(age_bin, parameter):
    parameter_scores = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == age_bin) &
        (UHL_PEWS_limits['parameter'] == parameter)
        ]['score'].values.astype(int).tolist()

    parameter_scores.reverse()
    print(parameter_scores)
    return parameter_scores


def get_lower_limit():
    # looks up the lower limit for scoring range in the model table
    limit_row = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == '0-11m') &
        (UHL_PEWS_limits['parameter'] == 'HR') &
        (UHL_PEWS_limits['score'] == 0)
        ]
    lower = int(limit_row['lower_lim'].values.item())
    print(lower)
    return lower


def get_upper_limit():
    upper_lim_row = UHL_PEWS_limits.loc[
        (UHL_PEWS_limits['age_range'] == '0-11m') &
        (UHL_PEWS_limits['parameter'] == 'HR') &
        (UHL_PEWS_limits['score'] == 0)
    ]
    upper_limit = int(upper_lim_row['upper_lim'].item())
    return upper_limit

""" calculate score function test code """

# lower = get_lower_limit(age_bin=age_bin)
# upper = get_upper_limit(age_bin='0-11m')
# score = UHL_PEWS_limits.loc[
#     (UHL_PEWS_limits['age_range'] == age_bin) &
#     (UHL_PEWS_limits['parameter'] == parameter)
#     ]['score'].values.astype(int).tolist()

# apply_score = lambda row: 0 if (row.age_bin == '0-11m') & (row.HR >= get_lower_limit(age_bin=row.age_bin)) & (row.HR < 160) else 1
#
# PEWS_scored['UHL_PEWS_HR'] = PEWS_scored.apply(apply_score, axis=1)


# this works
# lower = get_lower_limit(age_bin='0-11m')
# upper = get_upper_limit(age_bin='0-11m')
#
# apply_score = lambda row: 'score' if (row.age_bin == '0-11m') & (row.HR > lower) & (row.HR <= upper) else 'something'
#
# PEWS_scored['UHL_PEWS_HR'] = PEWS_scored.apply(apply_score, axis=1)


# this maybe works using pd.cut
# HR_bins = UHL_PEWS_limits.loc[
#     (UHL_PEWS_limits['age_range'] == '0-11m') &
#     (UHL_PEWS_limits['parameter'] == 'HR')
#     ]['upper_lim'].values.astype(int).tolist() + [0]
#
# HR_bins.reverse()
# # print(HR_bins)
#
# HR_scores = UHL_PEWS_limits.loc[
#     (UHL_PEWS_limits['age_range'] == '0-11m') &
#     (UHL_PEWS_limits['parameter'] == 'HR')
#     ]['score'].values.astype(int).tolist()
#
# HR_scores.reverse()
# # print(HR_scores)
#
# PEWS_scored['score_HR'] = pd.cut(PEWS_scored['HR'], bins= HR_bins, labels= HR_scores, ordered=False)




# this works using np.select
# get the parameter limits
# lower_lim = get_parameter_limit(age_bin='0-11m')[0]
# upper_lim = get_parameter_limit(age_bin='0-11m')[1]
#
# filters = [
#     (PEWS_df.age_bin == '0-11m') & (PEWS_df.HR > lower_lim) & (PEWS_df.HR < upper_lim),         # score = 0
#     (PEWS_df.age_bin == '0-11m') & ((PEWS_df.HR < lower_lim) | (PEWS_df.HR > upper_lim))      # score = 1
#            ]
#
# values = [0, 1]
#
# # apply filters to calculate the scores
# PEWS_scored['HR_score'] = np.select(filters, values, default='error')

# doesn't work really
# parameter_list = ['HR', 'RR']
# chart_list_2 = ['chart_1', 'chart_2']
#
# for par in parameter_list:
#     for chart in chart_list_2:
#         # get the parameter limits
#         lower_lim = PEWS[chart][par][0]
#         upper_lim = PEWS[chart][par][1]
#
#         conditions = [
#             (PEWS_df['age_bin'] == chart) & (PEWS_df[par] > lower_lim) & (PEWS_df[par] < upper_lim),  # score = 0
#             (PEWS_df['age_bin'] == chart)  # score = 1
#         ]
#
#         scores = [PEWS[chart][par][2], 1]
#
#         # apply filters to calculate the scores
#         PEWS_df[par+'_PEWS'] = np.select(conditions, scores, default='error')


# chart = 'chart_1'
# par = 'HR'
#
# # get the parameter limits
# lower_lim = PEWS[chart][par][0]
# upper_lim = PEWS[chart][par][1]
#
# conditions = [
#     (PEWS_df['age_bin'] == chart) & (PEWS_df[par] > lower_lim) & (PEWS_df[par] < upper_lim),         # score = 0
#     (PEWS_df['age_bin'] == chart)      # score = 1
#            ]
#
# scores = [0, 1]
#
# # apply filters to calculate the scores
# PEWS_df['HR_PEWS'] = np.select(conditions, scores, default='error')





""" other test """

# upper_lim = UHL_PEWS_limits.loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0)]['upper_lim'].values.astype(int)
#
# print(upper_lim)
#
# upper_lim_row = UHL_PEWS_limits.loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0)]
#
# upper_lim_2 = int(upper_lim_row['upper_lim'].item())
# print(upper_lim_2)
# # print(upper_lim_2.type())
#
#
# lower_lim = UHL_PEWS_limits['lower_lim'].loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0)].astype(int)
#
# print(lower_lim)
#
# HR_limits = UHL_PEWS_limits.loc[(UHL_PEWS_limits['age_range'] == '0-11m') & (UHL_PEWS_limits['parameter'] == 'HR') & (UHL_PEWS_limits['score'] == 0), ['upper_lim', 'lower_lim']].reset_index(drop=True)
#
# print(HR_limits)






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




