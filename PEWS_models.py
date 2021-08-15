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

class UPEWS(object):

    """
    age ranges: { 0: '0-11m', 1: '1-4y', 2: '5-12y', 3: '>13y', 4: 'all_ages' }
    columns are: { 0: 'chart', 1: 'parameter', 2: 'lower', 3: 'upper', 4: 'score' }
    lower = lower limit, upper = upper limit
    scores are calculated based on paper chart scoring convention:
        If (value >= lower) & (value <= upper) == True: score = score for that range
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

            [0, 'sats', 94, 100, 0],
            [0, 'sats', 30, 93, 0],

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

""" National PEWS model thresholds """

class NPEWS(object):

    """
    age ranges: { 0: '0-11m', 1: '1-4y', 2: '5-12y', 3: '>13y', 4: 'all_ages' }
    concern = { 'w': 'Worse , 's': 'Same', 'b': 'Better', 'pa': 'Parent Away', 'a': 'Parent Asleep' }
    AVPU = { 'asleep':'asleep', 'a':'alert','v':'responsive to voice', 'p': 'responsive to pain', 'u': 'unresponsive' }
    lower = lower limit, upper = upper limit
    scores are calculated based on paper chart scoring convention:
        (value >= lower) & (value <= upper) = score as per score for that range
    """

    limits = []

    def __init__(self):
        self.limits = [

            [4, 'concern', '', 'w', 0],
            [4, 'concern', '', 's', 0],
            [4, 'concern', '', 'b', 0],
            [4, 'concern', '', 'pa', 0],
            [4, 'concern', '', 'a', 0],

            [0, 'RR', 80, 150, 4],
            [0, 'RR', 70, 79, 2],
            [0, 'RR', 60, 69, 1],
            [0, 'RR', 41, 59, 0],
            [0, 'RR', 31, 40, 1],
            [0, 'RR', 21, 30, 2],
            [0, 'RR', 0, 20, 4],

            [1, 'RR', 80, 150, 4],
            [1, 'RR', 70, 79, 2],
            [1, 'RR', 50, 69, 1],
            [1, 'RR', 31, 49, 0],
            [1, 'RR', 21, 30, 1],
            [1, 'RR', 11, 20, 2],
            [1, 'RR', 0, 10, 4],

            [2, 'RR', 60, 100, 4],
            [2, 'RR', 50, 59, 2],
            [2, 'RR', 40, 49, 1],
            [2, 'RR', 21, 39, 0],
            [2, 'RR', 16, 20, 1],
            [2, 'RR', 11, 15, 2],
            [2, 'RR', 0, 10, 4],

            [3, 'RR', 40, 100, 4],
            [3, 'RR', 30, 39, 2],
            [3, 'RR', 25, 29, 1],
            [3, 'RR', 16, 24, 0],
            [3, 'RR', 11, 15, 1],
            [3, 'RR', 0, 10, 4],

            [4, 'Resp_Dis', '', 'severe', 4],
            [4, 'Resp_Dis', '', 'moderate', 2],
            [4, 'Resp_Dis', '', 'mild', 1],
            [4, 'Resp_Dis', '', 'none', 0],

            [0, 'sats', 0, 91, 4],
            [0, 'sats', 92, 94, 1],
            [0, 'sats', 95, 100, 0],

            [1, 'sats', 0, 91, 4],
            [1, 'sats', 92, 94, 1],
            [1, 'sats', 95, 100, 0],

            [2, 'sats', 0, 91, 4],
            [2, 'sats', 92, 94, 1],
            [2, 'sats', 95, 100, 0],

            [3, 'sats', 0, 91, 4],
            [3, 'sats', 92, 94, 1],
            [3, 'sats', 95, 100, 0],

            [4, 'FiO2', 50, 100, 4],
            [4, 'FiO2', 22, 49, 2],
            [4, 'FiO2', 21, 21, 0],
            [4, 'O2liters', 4, 40, 4],
            [4, 'O2litres', 0.01, 3.99, 2],
            [4, 'O2Delivery', '', 'np', 2],
            [4, 'O2Delivery', '', 'fm', np.NaN],
            [4, 'O2Delivery', '', 'hb', np.NaN],

            [0, 'HR', 180, 300, 4],
            [0, 'HR', 170, 179, 2],
            [0, 'HR', 150, 169, 1],
            [0, 'HR', 111, 149, 0],
            [0, 'HR', 91, 110, 1],
            [0, 'HR', 81, 90, 2],
            [0, 'HR', 0, 80, 4],

            [1, 'HR', 170, 300, 4],
            [1, 'HR', 150, 169, 2],
            [1, 'HR', 120, 149, 1],
            [1, 'HR', 91, 119, 0],
            [1, 'HR', 71, 90, 1],
            [1, 'HR', 61, 70, 2],
            [1, 'HR', 0, 60, 4],

            [2, 'HR', 160, 300, 4],
            [2, 'HR', 140, 159, 2],
            [2, 'HR', 120, 139, 1],
            [2, 'HR', 81, 119, 0],
            [2, 'HR', 71, 80, 1],
            [2, 'HR', 61, 70, 2],
            [2, 'HR', 0, 60, 4],

            [3, 'HR', 130, 300, 4],
            [3, 'HR', 120, 129, 2],
            [3, 'HR', 100, 119, 1],
            [3, 'HR', 71, 99, 0],
            [3, 'HR', 61, 70, 1],
            [3, 'HR', 51, 60, 2],
            [3, 'HR', 0, 50, 4],

            [0, 'sBP', 110, 250, 4],
            [0, 'sBP', 100, 109, 2],
            [0, 'sBP', 90, 99, 1],
            [0, 'sBP', 71, 89, 0],
            [0, 'sBP', 61, 70, 1],
            [0, 'sBP', 51, 60, 2],
            [0, 'sBP', 0, 50, 4],

            [1, 'sBP', 130, 250, 4],
            [1, 'sBP', 120, 129, 2],
            [1, 'sBP', 100, 119, 1],
            [1, 'sBP', 81, 99, 0],
            [1, 'sBP', 61, 80, 1],
            [1, 'sBP', 51, 60, 2],
            [1, 'sBP', 0, 50, 4],

            [2, 'sBP', 130, 250, 4],
            [2, 'sBP', 120, 129, 2],
            [2, 'sBP', 110, 119, 1],
            [2, 'sBP', 91, 109, 0],
            [2, 'sBP', 81, 90, 1],
            [2, 'sBP', 71, 80, 2],
            [2, 'sBP', 0, 70, 4],

            [3, 'sBP', 140, 250, 4],
            [3, 'sBP', 130, 139, 2],
            [3, 'sBP', 120, 129, 1],
            [3, 'sBP', 101, 119, 0],
            [3, 'sBP', 91, 100, 1],
            [3, 'sBP', 81, 90, 2],
            [3, 'sBP', 0, 80, 4],

            [4, 'cap_refill', 3, 10, 2],
            [4, 'cap_refill', 0, 2, 0],

            [4, 'temp', 38.0, 44.0, 0],
            [4, 'temp', 36.1, 37.9, 0],
            [4, 'temp', 0, 36.0, 0],

            [4, 'Sepsis6', 'N', 'Y', 0],

            [4, 'AVPU', '', 'u', 0],
            [4, 'AVPU', '', 'p', 0],
            [4, 'AVPU', '', 'v', 0],
            [4, 'AVPU', '', 'a', 0],
            [4, 'AVPU', '', 'as', 0],

            [4, 'pain', '', 'none', 0],
            [4, 'pain', '', 'mild', 0],
            [4, 'pain', '', 'moderate', 0],
            [4, 'pain', '', 'severe', 0],
            [4, 'pain', '', 'excruciating', 0]

        ]

    pass



# instantiate the PEWS model DataFrame
UPEWS_model = pd.DataFrame(UPEWS().limits, columns=['chart', 'par', 'lower', 'upper', 'score'])
# print(PEWS_model)

# instantiate the National PEWS model DataFrame
NPEWS_model = pd.DataFrame(NPEWS().limits, columns=['chart', 'par', 'lower', 'upper', 'score'])
# print(NPEWS_model)