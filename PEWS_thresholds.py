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





""" UHL Nervecentre PEWS model """

class UHL_PEWS(object):
    thresholds = []

    def __init__(self):
        self.thresholds = [
            [0, 'HR', 20, 60, 1],
            [0, 'HR', 61, 90, 1],
            [0, 'HR', 91, 160, 0],
            [0, 'HR', 161, 200, 1],
            [0, 'HR', 201, 310, 1],

            [1, 'HR', 20, 50, 1],
            [1, 'HR', 51, 90, 1],
            [1, 'HR', 91, 140, 0],
            [1, 'HR', 141, 180, 1],
            [1, 'HR', 181, 310, 1],

            [2, 'HR', 20, 44, 1],
            [2, 'HR', 45, 70, 1],
            [2, 'HR', 71, 120, 0],
            [2, 'HR', 121, 160, 1],
            [2, 'HR', 161, 250, 1],

            [3, 'HR', 20, 60, 1],
            [3, 'HR', 61, 100, 0],
            [3, 'HR', 101, 250, 1]
        ]
# print(get_thresholds('nat_PEWS', 3,'HR'))

# for plotting thresholds

def get_thresolds(model, parameter):
    age_brackets = [0, 1, 5, 12, 18]
    age_limits = [age * 365 for age in age_brackets]
    columns = ['bin', 'var', 'min', 'max', 'score']

    if model == 'UHL_PEWS':
        thresholds = pd.DataFrame(UHL_PEWS().thresholds, columns=columns)

    elif model == 'nat_PEWS':
        thresholds = pd.DataFrame(nat_PEWS().thresholds, columns=columns)

    limits = thresholds.loc[thresholds['var'] == parameter, ['bin', 'min', 'max']]

    # limits.bin = limits.bin.replace([0, age_limits[1]])
    limits.bin = limits.bin.apply(lambda x: age_limits[x+1])

    return limits

print(get_thresolds('UHL_PEWS', 'HR'))


# """ UHL Nervecentre PEWS model """
#
# class UHL_PEWS(object):
#     bin_0 = []
#     bin_1 = []
#     bin_2 = []
#     bin_3 = []
#
#     def __init__(self):
#         self.bin_0 = [
#             ['HR', 20, 60, 1],
#             ['HR', 61, 90, 1],
#             ['HR', 91, 160, 0],
#             ['HR', 161, 200, 1],
#             ['HR', 201, 310, 1]
#         ]
#         self.bin_1 = [
#             ['HR', 20, 50, 1],
#             ['HR', 51, 90, 1],
#             ['HR', 91, 140, 0],
#             ['HR', 141, 180, 1],
#             ['HR', 181, 310, 1]
#         ]
#         self.bin_2 = [
#             ['HR', 20, 44, 1],
#             ['HR', 45, 70, 1],
#             ['HR', 71, 120, 0],
#             ['HR', 121, 160, 1],
#             ['HR', 161, 250, 1]
#         ]
#         self.bin_3 = [
#             ['HR', 20, 60, 1],
#             ['HR', 61, 100, 0],
#             ['HR', 101, 250, 1]
#         ]

# """ National PEWS model thresholds """
#
# class nat_PEWS(object):
#
#     bin_0 = []
#     bin_1 = []
#     bin_2 = []
#     bin_3 = []
#
#     def __init__(self):
#         self.bin_0 = [
#
#             ['HR', -1, 50, 4],
#             ['HR', 51, 60, 2]
#
#         ]
#         self.bin_1 = [
#
#             ['HR', -1, 50, 4],
#             ['HR', 51, 60, 2],
#             ['HR', 61, 70, 1]
#
#         ]
#         self.bin_2 = [
#
#             ['HR', -1, 50, 4],
#             ['HR', 51, 60, 2],
#             ['HR', 61, 70, 1],
#             ['HR', 71, 100, 0]
#
#         ]
#         self.bin_3 = [
#
#             ['HR', -1, 50, 4],
#             ['HR', 51, 60, 2],
#             ['HR', 61, 70, 1],
#             ['HR', 71, 100, 0],
#             ['HR', 101, 120, 1],
#             ['HR', 121, 130, 2],
#             ['HR', 131, 300, 4],
#
#             ['RR', -1, 10, 4],
#             ['RR', 11, 15, 1],
#             ['RR', 16, 25, 0],
#             ['RR', 26, 30, 1],
#             ['RR', 31, 40, 2],
#             ['RR', 41, 65, 1]
#
#
#         ]
#
#     pass

# def get_thres(model, bin_num, parameter):
#     temp_df = []
#     columns = ['var', 'min', 'max', 'score']
#
#     if model == 'nat_PEWS':
#         if bin_num == 0:
#             temp_df = pd.DataFrame(nat_PEWS().bin_0, columns=columns)
#         elif bin_num == 1:
#             temp_df = pd.DataFrame(nat_PEWS().bin_1, columns=columns)
#         elif bin_num == 2:
#             temp_df = pd.DataFrame(nat_PEWS().bin_2, columns=columns)
#         elif bin_num == 3:
#             temp_df = pd.DataFrame(nat_PEWS().bin_3, columns=columns)
#
#     elif model == 'UHL_PEWS':
#         if bin_num == 0:
#             temp_df = pd.DataFrame(UHL_PEWS().bin_0, columns=columns)
#         elif bin_num == 1:
#             temp_df = pd.DataFrame(UHL_PEWS().bin_1, columns=columns)
#         elif bin_num == 2:
#             temp_df = pd.DataFrame(UHL_PEWS().bin_2, columns=columns)
#         elif bin_num == 3:
#             temp_df = pd.DataFrame(UHL_PEWS().bin_3, columns=columns)
#
#     temp_df = temp_df[temp_df['var'] == parameter]
#     return temp_df