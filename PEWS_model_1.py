"""
    NAME:          PEWSDataAnalysis/PEWS_centiles.py
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.1
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DETAILS:       Functions sued to process data and polt the data on charts with regression analysis
    DEPENDENCIES:  This program requires the following modules:
                    io, Numpy, Pandas, Mathplotlib, Seaborn, Statsmodels
"""

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas

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

            [1, 'sats', 94, 100, 0],
            [1, 'sats', 30, 93, 0],

            [2, 'sats', 94, 100, 0],
            [2, 'sats', 30, 93, 0],

            [3, 'sats', 94, 100, 0],
            [3, 'sats', 30, 93, 0],

            [4, 'sats', 94, 100, 0],
            [4, 'sats', 30, 93, 0],

            [4, 'receiving_o2', 24, 100, 1],
            [4, 'receiving_o2', 20, 23, 0],
            [4, 'receiving_o2', -1, 19, 1],

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