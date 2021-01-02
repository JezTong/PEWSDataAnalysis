"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
"""

import csv

# Open the PEWS .csv file: replace 'my_PEWS_file.csv' with correct file path.
with open('my_PEWS_file.csv') as PEWS_file:

    # read the PEWS .csv file
    PEWS_raw_data = csv.reader(PEWS_file)
    for row in PEWS_raw_data:
        print(row)

# create a DataFrame form the PEWS data

