"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, csv, Numpy, Pandas, Requests, getpass
                    Office365-REST-Python-Client 2.3.1 https://pypi.org/project/Office365-REST-Python-Client/
"""


# Import Python Modules
import io
import requests  # pip install requests
import getpass
import csv
import numpy as np # pip install numpy
import pandas as pd # pip install pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Import office365 share point API elements # pip install Office365-REST-Python-Client
from office365.runtime.auth.user_credential import UserCredential
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.runtime.client_request import ClientRequest
from office365.sharepoint.files.file import File

 # TODO: check if request and usercredential is needed

""" File Access """

sharepoint_url = "https://uhltrnhsuk.sharepoint.com"
username = input('username: ')
password = getpass.getpass()
site_url = "https://uhltrnhsuk.sharepoint.com/sites/case"
folder_url = "/sites/case/Shared%20Documents/PEWSDataAnalysis/"
filename = input("\nEnter file name: ")

# Access sharepoint folder with authentication
ctx_auth = AuthenticationContext(sharepoint_url)
if ctx_auth.acquire_token_for_user(username, password):
  request = ClientRequest(ctx_auth)
  ctx = ClientContext(site_url, ctx_auth)
  web = ctx.web
  ctx.load(web)
  ctx.execute_query()
  print("\nLogged into: {0}".format(web.properties['Title']))

else:
  print (ctx_auth.get_last_error())

# Function to load a file from sharepoint onto python
def Load_file(filename):

  # Open the file in temporary memory
  response = File.open_binary(ctx, str(folder_url + filename))
  bytes_file_obj = io.BytesIO()
  bytes_file_obj.write(response.content)

  # set file object to start
  bytes_file_obj.seek(0)

  global PEWS_df

  if filename.endswith(".xlsx"):
    print('\nLoading excel file...')
    PEWS_dict = pd.read_excel(bytes_file_obj, sheet_name='Sheet1')  # read the excel file in python as a dictonary
    PEWS_df = pd.DataFrame.from_dict(PEWS_dict)                     # convert dictionary to a dataframe using pandas
    return PEWS_df

  elif filename.endswith(".csv"):
    print("\nLoading .csv file...")
    PEWS_df = pd.read_csv(bytes_file_obj)    # read the .csv file in python as a dataframe
    return PEWS_df

  else:
    file = input("\nFile not recognised, please try again: ")
    Load_file(file)

# load the sharepoint file specified by user
Load_file(filename)



""" Data Cleaning """

# explore and examine the DataFrame
print('\nDisplaying DataFrame column headers and data types:\n')
print(PEWS_df.dtypes)
# print(PEWS_df)
# print('\n')
# print(PEWS_df.describe())
# print(PEWS_df.head(10))
# columns = PEWS_df.columns.tolist()


# Function to clean the PEWS parameter Data
# def data_clean(PEWS_df):
#
#   # if parameter in PEWS_df.select_dtypes(include=['float64']):
#   print('\nCleaning PEWS {} Data...'.format(parameter))
#   PEWS_df.observation = PEWS_df["observation"].replace("\D+", np.NaN, regex=True)  # replace text with null
#   PEWS_df.observation = pd.to_numeric(PEWS_df.observation)  # convert to python float/int
#   PEWS_df.observation = PEWS_df["observation"].dropna()  # remove null values
#   # else:
#   #   parameter = input("\nData type is not float, cannot clean data. Please select a valid parameter: ")
#   #   data_clean(parameter)
#
# # Clean the selected PEWS parameter data
# observation = input("\nSelect an observation to analyse: ")
# data_clean(observation)


# Clean the PEWS Data
PEWS_df.HR = PEWS_df['HR'].replace('\D+', np.NaN, regex = True)   # replace text with null
PEWS_df.HR = pd.to_numeric(PEWS_df.HR)                            # convert to python float/int
PEWS_df.HR = PEWS_df['HR'].dropna()                               # remove null values
Heart_Rates = PEWS_df['HR'].values


print(PEWS_df.dtypes)
# print(PEWS_df.describe())
print('\n')


""" Bin Data by age """

PEWS_bins = [0, 1, 5, 12, 18]                          # Age bins according to UHL PEWS chart categories
PEWS_bin_labels = ['0-11m', '1-4y', '5-11y', '>12']    # Age bin category labels

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
PEWS_df['PEWS_bins'] = pd.cut(PEWS_df['Age at Obs'], PEWS_bins, labels = PEWS_bin_labels)


""" Data Analysis """

# Seaborn method of plotting histogram
# plt.clf()
# sns.distplot(Heart_Rates)
# plt.show()

# Mathplotlib method test
# plt.clf()
# PEWS_df.plot(x="Age at Obs", y=["HR"])
# plt.show()

HR_PEWS_bin_one = PEWS_df.loc[PEWS_df['PEWS_bins'] == '0-11m']
HR_PEWS_bin_two = PEWS_df.loc[PEWS_df['PEWS_bins'] == '1-4y']
HR_PEWS_bin_three = PEWS_df.loc[PEWS_df['PEWS_bins'] == '5-11y']
HR_PEWS_bin_four = PEWS_df.loc[PEWS_df['PEWS_bins'] == '>12']

HR_PEWS_series = [HR_PEWS_bin_one.HR, HR_PEWS_bin_two.HR, HR_PEWS_bin_three.HR, HR_PEWS_bin_four.HR]

# Test plot the HR histograms
# Title = '{0} distribution for {1}'.format('HR', age_bin_labels[0])
Title = 'Heart rate distribution for 4 age groups'
Range = (50, 200)
# Range = (0, 250)
# plt.clf()
# plt.hist(HR_PEWS_bin_one.HR, range = Range, bins = 50, alpha = 0.4, label = PEWS_bin_labels[0] ) # edgecolor = 'white'
# plt.hist(HR_PEWS_bin_two.HR, range = Range, bins = 50, alpha = 0.4, label = PEWS_bin_labels[1])
# plt.hist(HR_PEWS_bin_three.HR, range = Range, bins = 50, alpha = 0.4, label = PEWS_bin_labels[2])
# plt.hist(HR_PEWS_bin_four.HR, range = Range, bins = 50, alpha = 0.4, label = PEWS_bin_labels[3])
# plt.title(Title)
# plt.legend()
# plt.show()

# Boxplot of Heart Rate for 4 age bins
plt.clf()
plt.title('Heart rates for 4 different age groups')
sns.boxplot(x = PEWS_df.PEWS_bins, y = PEWS_df.HR, width = 0.3)
plt.show()

# Boxplots of Heart Rate for each age
plt.clf()
plt.title('Heart rate range at different ages')
sns.boxplot(x = PEWS_df['Age at Obs'], y = PEWS_df.HR, width = 0.2)
plt.show()

# explore the Heart Rate data and plot a histogram of heart rates
# HR_min = np.amin(PEWS_df.HR)
# HR_max = np.amax(PEWS_df.HR)
# range = HR_max - HR_min
# print('\nThe lowest heart rate is {:.2f} beats per min'.format(HR_min))
# print('\nThe highest heart rate is {:.2f} beats per min'.format(HR_max))

HR_centiles = np.quantile(PEWS_df['HR'], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

# centile_line_color = ['red', 'orange', 'yellow', 'green', 'yellow', 'orange', 'red']
# plt.clf()
# plt.hist(PEWS_df['HR'], range = (0, 250), bins = 25, edgecolor = 'white')

# for n in [0, 1, 2, 3, 4, 5, 6]:
#   plt.axvline(x = HR_centiles[n], color = centile_line_color[n])

# for centile in HR_centiles:
#   plt.axvline(x = centile, color = 'red')

# plt.axvline(x = HR_centiles[0], c = 'red') # 5th centile
# plt.axvline(x = HR_centiles[1], c = 'orange') # 10th centile
# plt.axvline(x = HR_centiles[2], c = 'yellow') # 25th centile
# plt.axvline(x = HR_centiles[3], c = 'green') # 50th centile
# plt.axvline(x = HR_centiles[4], c = 'yellow') # 75th centile
# plt.axvline(x = HR_centiles[5], c = 'orange') # 90th centile
# plt.axvline(x = HR_centiles[6], c = 'red') # 95th centile
# plt.title('Heart Rate')
# plt.show()





