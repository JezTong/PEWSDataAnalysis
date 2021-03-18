"""
    NAME:          PEWSDataAnalysis
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

""" National PEWS model thresholds """

class nat_PEWS(object):

    thresholds = []

    def __init__(self):
        self.thresholds = [

            [0, 'HR', -1, 80, 4, False],
            [0, 'HR', 81, 90, 2, True],
            [0, 'HR', 91, 110, 1, True],
            [0, 'HR', 111, 150, 0, True],
            [0, 'HR', 151, 170, 1, True],
            [0, 'HR', 171, 180, 2, True],
            [0, 'HR', 181, 310, 4, False],

            [1, 'HR', -1, 60, 4, False],
            [1, 'HR', 61, 70, 2, True],
            [1, 'HR', 71, 90, 1, True],
            [1, 'HR', 91, 120, 0, True],
            [1, 'HR', 121, 150, 1, True],
            [1, 'HR', 151, 170, 2, True],
            [1, 'HR', 171, 310, 4, False],

            [2, 'HR', -1, 60, 4, False],
            [2, 'HR', 61, 70, 2, True],
            [2, 'HR', 71, 80, 1, True],
            [2, 'HR', 81, 120, 0, True],
            [2, 'HR', 121, 140, 1, True],
            [2, 'HR', 141, 160, 2, True],
            [2, 'HR', 161, 310, 4, False],

            [3, 'HR', -1, 50, 4, False],
            [3, 'HR', 51, 60, 2, True],
            [3, 'HR', 61, 70, 1, True],
            [3, 'HR', 71, 100, 0, True],
            [3, 'HR', 101, 120, 1, True],
            [3, 'HR', 121, 130, 2, True],
            [3, 'HR', 131, 310, 4, False],

            [3, 'RR', -1, 10, 4, False],
            [3, 'RR', 11, 15, 1, True],
            [3, 'RR', 16, 25, 0, True],
            [3, 'RR', 26, 30, 1, False],
            [3, 'RR', 31, 40, 2, False],
            [3, 'RR', 41, 65, 4, False]

        ]

    pass



""" UHL Nervecentre PEWS model """

class UHL_PEWS(object):
    thresholds = []

    def __init__(self):
        self.thresholds = [
            [0, 'HR', 20, 60, 1, False],
            [0, 'HR', 61, 90, 1, True],
            [0, 'HR', 91, 160, 0, True],
            [0, 'HR', 161, 200, 1, False],
            [0, 'HR', 201, 310, 1, False],

            [1, 'HR', 20, 50, 1, False],
            [1, 'HR', 51, 90, 1, True],
            [1, 'HR', 91, 140, 0, True],
            [1, 'HR', 141, 180, 1, False],
            [1, 'HR', 181, 310, 1, False],

            [2, 'HR', 20, 44, 1, False],
            [2, 'HR', 45, 70, 1, True],
            [2, 'HR', 71, 120, 0, True],
            [2, 'HR', 121, 160, 1, False],
            [2, 'HR', 161, 250, 1, False],

            [3, 'HR', 20, 60, 1, True],
            [3, 'HR', 61, 100, 0, True],
            [3, 'HR', 101, 250, 1, False],

            [0, 'RR', 0, 10, 1, False],
            [0, 'RR', 11, 20, 1, False],
            [0, 'RR', 21, 30, 1, True],
            [0, 'RR', 31, 60, 0, True],
            [0, 'RR', 61, 110, 1, False],

            [1, 'RR', 0, 10, 1, False],
            [1, 'RR', 11, 16, 1, False],
            [1, 'RR', 17, 20, 1, True],
            [1, 'RR', 21, 40, 0, True],
            [1, 'RR', 41, 50, 1, False],
            [1, 'RR', 51, 80, 1, False],

            [2, 'RR', 0, 8, 1, False],
            [2, 'RR', 9, 14, 1, False],
            [2, 'RR', 15, 20, 1, True],
            [2, 'RR', 21, 30, 0, True],
            [2, 'RR', 31, 40, 1, False],
            [2, 'RR', 41, 70, 1, False],

            [3, 'RR', 0, 6, 1, False],
            [3, 'RR', 7, 10, 1, True],
            [3, 'RR', 11, 20, 0, True],
            [3, 'RR', 21, 25, 1, False],
            [3, 'RR', 25, 60, 1, False],

            [0, 'sBP', 30, 50, 0, False],
            [0, 'sBP', 51, 60, 0, False],
            [0, 'sBP', 61, 70, 0, True],
            [0, 'sBP', 71, 109, 0, True],
            [0, 'sBP', 110, 250, 0, True],

            [1, 'sBP', 30, 60, 0, False],
            [1, 'sBP', 61, 70, 0, False],
            [1, 'sBP', 71, 80, 0, True],
            [1, 'sBP', 81, 110, 0, True],
            [1, 'sBP', 111, 250, 0, True],

            [2, 'sBP', 30, 70, 0, False],
            [2, 'sBP', 71, 80, 0, False],
            [2, 'sBP', 81, 90, 0, True],
            [2, 'sBP', 91, 120, 0, True],
            [2, 'sBP', 121, 250, 0, True],

            [3, 'sBP', 30, 75, 0, False],
            [3, 'sBP', 76, 100, 0, True],
            [3, 'sBP', 101, 150, 0, True],
            [3, 'sBP', 151, 250, 0, False],

            [0, 'O2', 24, 100, 1, False],
            [0, 'O2', 20, 23, 0, False],

            [1, 'O2', 24, 100, 1, False],
            [1, 'O2', 20, 23, 0, False],

            [2, 'O2', 24, 100, 1, False],
            [2, 'O2', 20, 23, 0, False],

            [3, 'O2', 24, 100, 1, False],
            [3, 'O2', 20, 23, 0, False],

            [0, 'Sats', 30, 93, 0, False],
            [0, 'Sats', 94, 100, 0, False],

            [1, 'Sats', 30, 93, 0, False],
            [1, 'Sats', 94, 100, 0, False],

            [2, 'Sats', 30, 93, 0, False],
            [2, 'Sats', 94, 100, 0, False],

            [3, 'Sats', 30, 93, 0, False],
            [3, 'Sats', 94, 100, 0, False],

            [0, 'WoB', '', 'stridor', 1, False],
            [0, 'WoB', '', 'grunting', 1, False],
            [0, 'WoB', '', 'severe', 1, False],
            [0, 'WoB', '', 'moderate', 1, False],
            [0, 'WoB', '', 'mild', 0, False],
            [0, 'WoB', '', 'none', 0, False],

            [1, 'WoB', '', 'stridor', 1, False],
            [1, 'WoB', '', 'grunting', 1, False],
            [1, 'WoB', '', 'severe', 1, False],
            [1, 'WoB', '', 'moderate', 1, False],
            [1, 'WoB', '', 'mild', 0, False],
            [1, 'WoB', '', 'none', 0, False],

            [2, 'WoB', '', 'stridor', 1, False],
            [2, 'WoB', '', 'grunting', 1, False],
            [2, 'WoB', '', 'severe', 1, False],
            [2, 'WoB', '', 'moderate', 1, False],
            [2, 'WoB', '', 'mild', 0, False],
            [2, 'WoB', '', 'none', 0, False],

            [3, 'WoB', '', 'stridor', 1, False],
            [3, 'WoB', '', 'grunting', 1, False],
            [3, 'WoB', '', 'severe', 1, False],
            [3, 'WoB', '', 'moderate', 1, False],
            [3, 'WoB', '', 'mild', 0, False],
            [3, 'WoB', '', 'none', 0, False],

            [0, 'concern', '', 'Nurse', 1, False],
            [0, 'concern', '', 'parent', 1, False],

            [1, 'concern', '', 'Nurse', 1, False],
            [1, 'concern', '', 'parent', 1, False],

            [2, 'concern', '', 'Nurse', 1, False],
            [2, 'concern', '', 'parent', 1, False],

            [3, 'concern', '', 'Nurse', 1, False],
            [3, 'concern', '', 'parent', 1, False],

            [0, 'ACVPU', '', 'unresponsive', 1, False],
            [0, 'ACVPU', '', 'pain', 1, False],
            [0, 'ACVPU', '', 'voice', 1, False],
            [0, 'ACVPU', '', 'alert', 0, False],
            [0, 'ACVPU', '', 'asleep', 0, False],

            [1, 'ACVPU', '', 'unresponsive', 1, False],
            [1, 'ACVPU', '', 'pain', 1, False],
            [1, 'ACVPU', '', 'voice', 1, False],
            [1, 'ACVPU', '', 'alert', 0, False],
            [1, 'ACVPU', '', 'asleep', 0, False],

            [2, 'ACVPU', '', 'unresponsive', 1, False],
            [2, 'ACVPU', '', 'pain', 1, False],
            [2, 'ACVPU', '', 'voice', 1, False],
            [2, 'ACVPU', '', 'alert', 0, False],
            [2, 'ACVPU', '', 'asleep', 0, False],

            [3, 'ACVPU', '', 'unresponsive', 1, False],
            [3, 'ACVPU', '', 'pain', 1, False],
            [3, 'ACVPU', '', 'voice', 1, False],
            [3, 'ACVPU', '', 'alert', 0, False],
            [3, 'ACVPU', '', 'asleep', 0, False],

            [0, 'temp', 28.0, 36.0, 0, False],
            [0, 'temp', 36.1, 38.4, 0, False],
            [0, 'temp', 38.5, 42.0, 0, False],

            [1, 'temp', 28.0, 36.0, 0, False],
            [1, 'temp', 36.1, 38.4, 0, False],
            [1, 'temp', 38.5, 42.0, 0, False],

            [2, 'temp', 28.0, 36.0, 0, False],
            [2, 'temp', 36.1, 38.4, 0, False],
            [2, 'temp', 38.5, 42.0, 0, False],

            [3, 'temp', 28.0, 36.0, 0, False],
            [3, 'temp', 36.1, 38.4, 0, False],
            [3, 'temp', 38.5, 42.0, 0, False],

            [0, 'cap_refill', '', '0-2', 0, False],
            [0, 'cap_refill', '', '3-4', 0, False],
            [0, 'cap_refill', '', '5-8', 0, False],
            [0, 'cap_refill', '', 'grey/mottled', 0, False],

            [1, 'cap_refill', '', '0-2', 0, False],
            [1, 'cap_refill', '', '3-4', 0, False],
            [1, 'cap_refill', '', '5-8', 0, False],
            [1, 'cap_refill', '', 'grey/mottled', 0, False],

            [2, 'cap_refill', '', '0-2', 0, False],
            [2, 'cap_refill', '', '3-4', 0, False],
            [2, 'cap_refill', '', '5-8', 0, False],
            [2, 'cap_refill', '', 'grey/mottled', 0, False],

            [3, 'cap_refill', '', '0-2', 0, False],
            [3, 'cap_refill', '', '3-4', 0, False],
            [3, 'cap_refill', '', '5-8', 0, False],
            [3, 'cap_refill', '', 'grey/mottled', 0, False],

            [0, 'pain', '', 'no', 0, False],
            [0, 'pain', '', 'mild', 0, False],
            [0, 'pain', '', 'moderate', 0, False],
            [0, 'pain', '', 'severe', 0, False],
            [0, 'pain', '', 'excruciating', 0, False],

            [1, 'pain', '', 'no', 0, False],
            [1, 'pain', '', 'mild', 0, False],
            [1, 'pain', '', 'moderate', 0, False],
            [1, 'pain', '', 'severe', 0, False],
            [1, 'pain', '', 'excruciating', 0, False],

            [2, 'pain', '', 'no', 0, False],
            [2, 'pain', '', 'mild', 0, False],
            [2, 'pain', '', 'moderate', 0, False],
            [2, 'pain', '', 'severe', 0, False],
            [2, 'pain', '', 'excruciating', 0, False],

            [3, 'pain', '', 'no', 0, False],
            [3, 'pain', '', 'mild', 0, False],
            [3, 'pain', '', 'moderate', 0, False],
            [3, 'pain', '', 'severe', 0, False],
            [3, 'pain', '', 'excruciating', 0, False]

        ]


# for plotting thresholds
def get_thresolds(model, parameter):
    columns = ['bin', 'var', 'mini', 'maxi', 'score', 'lim']

    if model == 'UHL_PEWS':
        thresholds = pd.DataFrame(UHL_PEWS().thresholds, columns=columns)

    elif model == 'nat_PEWS':
        thresholds = pd.DataFrame(nat_PEWS().thresholds, columns=columns)

    limits = thresholds.loc[(thresholds['var'] == parameter) & (thresholds['lim'] == True), ['bin', 'maxi']]
    limits.reset_index(inplace=True, drop=True)

    return limits

# print(get_thresolds('UHL_PEWS', 'HR'))
# table = get_thresolds('UHL_PEWS', 'HR')

def generate_lines(table):
    age_brackets = [0, 1, 5, 12, 18]
    age_limits = [age * 365 for age in age_brackets]

    table['a1'] = table['bin']
    table['a2'] = table['bin']

    table.a1 = table.a1.apply(lambda x: age_limits[x])
    table.a2 = table.a2.apply(lambda x: age_limits[x + 1] - 1)

    line_list = []

    for i in range(len(table.a1)):
        line = [table.a1[i], table.maxi[i]], [table.a2[i], table.maxi[i]]
        line_list.append(line)

    return line_list


