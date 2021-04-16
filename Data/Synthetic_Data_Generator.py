"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for creating a synthetic PEWS Dataset for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    Numpy, Pandas, Random
"""

# create a synthetic data table of Observations for PEWS calculations

import pandas as pd
import numpy as np
import random

# seed random number generator
random.seed(1)

# parameters = ['concern', 'UHL_concern', 'RR', 'Resp_Dis', 'WoB', 'Sats', 'FiO2', 'HR', 'sBP', 'cap_refill', 'temp', 'AVPU', 'pain']

#TODO Add in age variation to synthetic data variables

# parameter distribution as {'parameter' : [mean, std]} (may need adjusting - base on mean and std from real dataset)
cont_parameters = {

    'RR' : [50, 10],
    'Sats' : [95, 5],
    'HR' : [130, 30],
    'sBP' : [80, 40],
    'temp' : [36.5, 2]

}

# categorical parameters
cat_parameters = {
    'concern': [0, 1, 2, 3, 4],     # Parent Asleep, Parent Away, Better, Same, Worse
    'U_concern' : [0, 1, 2],        # no concern, parent concern, nurse concern
    'Resp_Dis': [0, 1, 2, 3],       # none, mild, moderate, severe
    'U_WoB': [0, 1, 2, 3, 4, 5],    # none, mild, moderate, severe, grunting, stridor
    'FiO2': [0, 1, 2],              # 21%, 22-49%, 50-100%
    'U_FiO2' : [0, 1],              # 20-23%, 24-100%
    'cap_refill' : [0, 1, 2, 3],    # 0-2s, 3-10s,
    'U_cap_refill' : [0, 1, 2, 3],  # 0-2s, 3-4s, 5-8s, grey/mottled
    'AVPU' : [0, 1, 2, 3, 4],       # asleep, alert, voice, pain, unresponsive
    'pain' : [0, 1, 2, 3, 4]        # no, mild, moderate, severe, excruciating

}

# number of children in dataset
num_children = 10000

# create the data
data = {}

# create a list of children of different ages (0 - 18 y)
data['age_in_days'] = random.choices(list(range(0, 6570)), k = num_children)
data['age'] = list(map(lambda a: np.round(a/365), data['age_in_days']))

# create the continuous data parameters
for parameter in cont_parameters:
    data[parameter] = [np.round(random.gauss(cont_parameters[parameter][0], cont_parameters[parameter][1])) for child in range(num_children)]

# create the category data parameters
for parameter in cat_parameters:
    data[parameter] = random.choices(cat_parameters[parameter], k = num_children)

# Convert to a DataFrame
global df
df = pd.DataFrame(data).reset_index(drop=True)
df.to_csv('synthetic_obs.csv', index=False)


# set pandas options to display all columns in a DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.head())
print(df.describe())
