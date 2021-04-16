"""
    NAME:          PEWSDataAnalysis: PEWS_models.py
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:

"""

import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection

""" National PEWS model thresholds """

class nat_PEWS(object):

    thresholds = []

    def __init__(self):
        self.thresholds = [

            # age ranges: 0 = 'all_ages', 1 = '0-11m', 2 = '1-4y', 3 = '5-11y', 4 = '>12y'

            [0, 'concern', '', 'Worse', 0, False],
            [0, 'concern', '', 'Same', 0, False],
            [0, 'concern', '', 'Better', 0, False],
            [0, 'concern', '', 'Parent Away', 0, False],
            [0, 'concern', '', 'Parent Asleep', 0, False],

            [1, 'RR', 81, 100, 4, False],
            [1, 'RR', 71, 80, 2, False],
            [1, 'RR', 61, 70, 1, False],
            [1, 'RR', 41, 60, 0, True],
            [1, 'RR', 31, 40, 1, True],
            [1, 'RR', 21, 30, 2, False],
            [1, 'RR', 0, 20, 4, False],

            [2, 'RR', 81, 100, 4, False],
            [2, 'RR', 71, 80, 2, False],
            [2, 'RR', 51, 70, 1, False],
            [2, 'RR', 31, 50, 0, True],
            [2, 'RR', 21, 30, 1, True],
            [2, 'RR', 11, 20, 2, False],
            [2, 'RR', 0, 10, 4, False],

            [3, 'RR', 41, 100, 4, False],
            [3, 'RR', 31, 40, 2, False],
            [3, 'RR', 26, 30, 1, False],
            [3, 'RR', 16, 25, 0, True],
            [3, 'RR', 11, 15, 1, True],
            [3, 'RR', 0, 10, 4, False],

            [4, 'RR', 41, 100, 4, False],
            [4, 'RR', 31, 40, 2, False],
            [4, 'RR', 26, 30, 1, False],
            [4, 'RR', 16, 25, 0, True],
            [4, 'RR', 11, 15, 1, True],
            [4, 'RR', 0, 10, 4, False],

            [0, 'Resp_Dis', '', 'severe', 4, False],
            [0, 'Resp_Dis', '', 'moderate', 2, False],
            [0, 'Resp_Dis', '', 'mild', 1, False],
            [0, 'Resp_Dis', '', 'none', 0, False],

            [0, 'Sats', 0, 91, 4, False],
            [0, 'Sats', 92, 94, 1, False],
            [0, 'Sats', 95, 100, 0, False],

            [0, 'FiO2', 50, 100, 4, False],
            [0, 'FiO2', 22, 49, 1, False],
            [0, 'FiO2', 21, 21, 0, False],
            [0, 'O2liters', 4, 15, 4, False],
            [0, 'O2litres', 0.1, 4, 1, False],

            [1, 'HR', 181, 310, 4, False],
            [1, 'HR', 171, 180, 2, True],
            [1, 'HR', 151, 170, 1, True],
            [1, 'HR', 111, 150, 0, True],
            [1, 'HR', 91, 110, 1, True],
            [1, 'HR', 81, 90, 2, True],
            [1, 'HR', 0, 80, 4, False],

            [2, 'HR', 171, 310, 4, False],
            [2, 'HR', 151, 170, 2, True],
            [2, 'HR', 121, 150, 1, True],
            [2, 'HR', 91, 120, 0, True],
            [2, 'HR', 71, 90, 1, True],
            [2, 'HR', 61, 70, 2, True],
            [2, 'HR', 0, 60, 4, False],

            [3, 'HR', 161, 310, 4, False],
            [3, 'HR', 141, 160, 2, True],
            [3, 'HR', 121, 140, 1, True],
            [3, 'HR', 81, 120, 0, True],
            [3, 'HR', 71, 80, 1, True],
            [3, 'HR', 61, 70, 2, True],
            [3, 'HR', 0, 60, 4, False],

            [4, 'HR', 131, 310, 4, False],
            [4, 'HR', 121, 130, 2, True],
            [4, 'HR', 101, 120, 1, True],
            [4, 'HR', 71, 100, 0, True],
            [4, 'HR', 61, 70, 1, True],
            [4, 'HR', 51, 60, 2, True],
            [4, 'HR', -1, 50, 4, False],

            [1, 'sBP', 111, 250, 4, False],
            [1, 'sBP', 101, 110, 2, True],
            [1, 'sBP', 91, 100, 1, True],
            [1, 'sBP', 71, 90, 0, True],
            [1, 'sBP', 61, 70, 1, True],
            [1, 'sBP', 51, 60, 2, True],
            [1, 'sBP', 0, 50, 4, False],

            [2, 'sBP', 131, 250, 4, False],
            [2, 'sBP', 121, 130, 2, False],
            [2, 'sBP', 101, 120, 1, False],
            [2, 'sBP', 81, 100, 0, False],
            [2, 'sBP', 61, 80, 1, False],
            [2, 'sBP', 51, 60, 2, False],
            [2, 'sBP', 0, 50, 4, False],

            [3, 'sBP', 141, 250, 4, False],
            [3, 'sBP', 131, 140, 2, False],
            [3, 'sBP', 121, 130, 1, False],
            [3, 'sBP', 101, 120, 0, False],
            [3, 'sBP', 91, 100, 1, False],
            [3, 'sBP', 81, 90, 2, False],
            [3, 'sBP', 0, 80, 4, False],

            [4, 'sBP', 141, 250, 4, False],
            [4, 'sBP', 131, 140, 2, False],
            [4, 'sBP', 121, 130, 1, False],
            [4, 'sBP', 101, 120, 0, False],
            [4, 'sBP', 91, 100, 1, False],
            [4, 'sBP', 81, 90, 2, False],
            [4, 'sBP', 0, 80, 4, False],

            [0, 'cap_refill', 3, 10, 2, False],
            [0, 'cap_refill', 0, 2, 0, False],

            [0, 'temp', 38.1, 44.0, 0, False],
            [0, 'temp', 36.1, 38, 0, False],
            [0, 'temp', 0, 36.0, 0, False],

            [0, 'Sepsis6', 'N', 'Y', 0, False],

            [0, 'AVPU', '', 'unresponsive', 0, False],
            [0, 'AVPU', '', 'pain', 0, False],
            [0, 'AVPU', '', 'voice', 0, False],
            [0, 'AVPU', '', 'alert', 0, False],
            [0, 'AVPU', '', 'asleep', 0, False],

            [0, 'pain', '', 'no', 0, False],
            [0, 'pain', '', 'mild', 0, False],
            [0, 'pain', '', 'moderate', 0, False],
            [0, 'pain', '', 'severe', 0, False],
            [0, 'pain', '', 'excruciating', 0, False]

        ]

    pass



""" UHL Nervecentre PEWS model """

class UHL_PEWS(object):
    thresholds = []

    def __init__(self):
        self.thresholds = [

            # age ranges: 0 = 'all_ages', 1 = '0-11m', 2 = '1-4y', 3 = '5-11y', 4 = '>12y'

            [1, 'HR', 201, 310, 1, False],
            [1, 'HR', 161, 200, 1, False],
            [1, 'HR', 91, 160, 0, True],
            [1, 'HR', 61, 90, 1, True],
            [1, 'HR', 20, 60, 1, False],

            [2, 'HR', 181, 310, 1, False],
            [2, 'HR', 141, 180, 1, False],
            [2, 'HR', 91, 140, 0, True],
            [2, 'HR', 51, 90, 1, True],
            [2, 'HR', 20, 50, 1, False],

            [3, 'HR', 161, 250, 1, False],
            [3, 'HR', 121, 160, 1, False],
            [3, 'HR', 71, 120, 0, True],
            [3, 'HR', 45, 70, 1, True],
            [3, 'HR', 20, 44, 1, False],

            [4, 'HR', 101, 250, 1, False],
            [4, 'HR', 61, 100, 0, True],
            [4, 'HR', 20, 60, 1, True],

            [1, 'RR', 61, 110, 1, False],
            [1, 'RR', 31, 60, 0, True],
            [1, 'RR', 21, 30, 1, True],
            [1, 'RR', 11, 20, 1, False],
            [1, 'RR', 0, 10, 1, False],

            [2, 'RR', 51, 80, 1, False],
            [2, 'RR', 41, 50, 1, False],
            [2, 'RR', 21, 40, 0, True],
            [2, 'RR', 17, 20, 1, True],
            [2, 'RR', 11, 16, 1, False],
            [2, 'RR', 0, 10, 1, False],

            [3, 'RR', 41, 70, 1, False],
            [3, 'RR', 31, 40, 1, False],
            [3, 'RR', 21, 30, 0, True],
            [3, 'RR', 15, 20, 1, True],
            [3, 'RR', 9, 14, 1, False],
            [3, 'RR', 0, 8, 1, False],

            [4, 'RR', 25, 60, 1, False],
            [4, 'RR', 21, 25, 1, False],
            [4, 'RR', 11, 20, 0, True],
            [4, 'RR', 7, 10, 1, True],
            [4, 'RR', 0, 6, 1, False],

            [1, 'sBP', 110, 250, 0, False],
            [1, 'sBP', 71, 109, 0, True],
            [1, 'sBP', 61, 70, 0, True],
            [1, 'sBP', 51, 60, 0, False],
            [1, 'sBP', 30, 50, 0, False],

            [2, 'sBP', 111, 250, 0, False],
            [2, 'sBP', 81, 110, 0, True],
            [2, 'sBP', 71, 80, 0, True],
            [2, 'sBP', 61, 70, 0, False],
            [2, 'sBP', 30, 60, 0, False],

            [3, 'sBP', 121, 250, 0, False],
            [3, 'sBP', 91, 120, 0, True],
            [3, 'sBP', 81, 90, 0, True],
            [3, 'sBP', 71, 80, 0, False],
            [3, 'sBP', 30, 70, 0, False],

            [4, 'sBP', 151, 250, 0, False],
            [4, 'sBP', 101, 150, 0, True],
            [4, 'sBP', 76, 100, 0, True],
            [4, 'sBP', 30, 75, 0, False],

            [0, 'FiO2', 24, 100, 1, False],
            [0, 'FiO2', 20, 23, 0, False],

            [0, 'Sats', 94, 100, 0, False],
            [0, 'Sats', 30, 93, 0, False],

            [0, 'WoB', '', 'stridor', 1, False],
            [0, 'WoB', '', 'grunting', 1, False],
            [0, 'WoB', '', 'severe', 1, False],
            [0, 'WoB', '', 'moderate', 1, False],
            [0, 'WoB', '', 'mild', 0, False],
            [0, 'WoB', '', 'none', 0, False],

            [0, 'UHL_concern', '', 'Nurse', 1, False],
            [0, 'UHL_concern', '', 'parent', 1, False],

            [0, 'AVPU', '', 'unresponsive', 1, False],
            [0, 'AVPU', '', 'pain', 1, False],
            [0, 'AVPU', '', 'voice', 1, False],
            [0, 'AVPU', '', 'alert', 0, False],
            [0, 'AVPU', '', 'asleep', 0, False],

            [0, 'temp', 28.0, 36.0, 0, False],
            [0, 'temp', 36.1, 38.4, 0, False],
            [0, 'temp', 38.5, 42.0, 0, False],

            [0, 'cap_refill', '', '0-2', 0, False],
            [0, 'cap_refill', '', '3-4', 0, False],
            [0, 'cap_refill', '', '5-8', 0, False],
            [0, 'cap_refill', '', 'grey/mottled', 0, False],

            [0, 'pain', '', 'no', 0, False],
            [0, 'pain', '', 'mild', 0, False],
            [0, 'pain', '', 'moderate', 0, False],
            [0, 'pain', '', 'severe', 0, False],
            [0, 'pain', '', 'excruciating', 0, False],

        ]

    pass


# Function to generate a table containing the PEWS model
def generate_model_table(model):
    columns = ['age_range', 'parameter', 'lower_lim', 'upper_lim', 'score', 'plot?']
    age_bin_labels = ['all_ages', '0-11m', '1-4y', '5-11y', '>12y']

    if model == 'UHL_PEWS':
        table = pd.DataFrame(UHL_PEWS().thresholds, columns=columns)

    elif model == 'nat_PEWS':
        table = pd.DataFrame(nat_PEWS().thresholds, columns=columns)

    table.age_range = table.age_range.apply(lambda x: age_bin_labels[x])
    table = table[['age_range', 'parameter', 'lower_lim', 'upper_lim', 'score']]

    return table


# Function to generate a table containing the PEWS model limits (better for plotting)
def generate_thresholds_table(model, parameter, score):
    columns = ['bins', 'parameter', 'lo_limit', 'up_limit', 'score', 'plot?']

    if model == 'UHL_PEWS':
        table = pd.DataFrame(UHL_PEWS().thresholds, columns=columns)

    elif model == 'nat_PEWS':
        table = pd.DataFrame(nat_PEWS().thresholds, columns=columns)

    # select the correct parameter and cut the threshold table down to size
    table = table.loc[(table['parameter'] == parameter) & (table['score'] == score)].reset_index()
    table = table[['bins', 'lo_limit', 'up_limit']]

    age_bins = [0, 365, 1826, 4383, 6575]

    # create new table and map thresholds to age on a continuous scale
    mapping = pd.DataFrame({'age': range(1, 6575),
                            'bins': pd.cut(range(1, 6575), bins=age_bins, labels=[1, 2, 3, 4])})
    new_table = mapping.merge(table)

    return new_table

# print(generate_model_table('nat_PEWS', 'HR', 0))
# exit()

# Function for looking up thresholds for a specified parameter of a specified PEWS model
def get_thresolds(model, parameter):
    columns = ['bin', 'var', 'mini', 'maxi', 'score', 'lim']

    if model == 'UHL_PEWS':
        thresholds = pd.DataFrame(UHL_PEWS().thresholds, columns=columns)

    elif model == 'nat_PEWS':
        thresholds = pd.DataFrame(nat_PEWS().thresholds, columns=columns)

    limits = thresholds.loc[(thresholds['var'] == parameter) & (thresholds['lim'] == True), ['bin', 'maxi']]
    limits.reset_index(inplace=True, drop=True)

    return limits

# Function to generate threshold limit line coordinates for plotting in a chart
def generate_lines(model, parameter, color = 'red', linewidth = 1):
    age_brackets = [0, 0, 1, 5, 12, 18]
    age_limits = [age * 365 for age in age_brackets]

    table = get_thresolds(model, parameter)

    table['a1'] = table['bin']
    table['a2'] = table['bin']

    table.a1 = table.a1.apply(lambda x: age_limits[x])
    table.a2 = table.a2.apply(lambda x: age_limits[x + 1] - 1)

    line_list = []

    for i in range(len(table)):
        line = [table.a1[i], table.maxi[i]], [table.a2[i], table.maxi[i]]
        line_list.append(line)

    return  LineCollection(line_list, linewidth=linewidth, color=color)


""" view threshold tables and generate .csv files """

# print(generate_model_table('UHL_PEWS'))
# print('\n')
# print(generate_model_table('nat_PEWS'))
# generate_model_table('nat_PEWS').to_csv('nat_PEWS_limits.csv', index=False)
# generate_model_table('UHL_PEWS').to_csv('UHL_PEWS_limits.csv', index=False)
