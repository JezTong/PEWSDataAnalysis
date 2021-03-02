"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, Numpy, Pandas, Mathplotlib, Seaborn, getpass
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

  global df

  if filename.endswith(".xlsx"):
    print('\nLoading {}...'.format(filename))
    dict = pd.read_excel(bytes_file_obj, sheet_name='Sheet1')  # read the excel file in python as a dictonary
    df = pd.DataFrame.from_dict(dict)                     # convert dictionary to a dataframe using pandas
    return df

  elif filename.endswith(".csv"):
    print("\nLoading {}...".format(filename))
    df = pd.read_csv(bytes_file_obj)    # read the .csv file in python as a dataframe
    return df

  else:
    file = input("\nFile not recognised, please try again: ")
    Load_file(file)

""" Load the Sharepoint Files """

PEWS_df = Load_file('PEWS_Data_1.xlsx')
# HISS_df = Load_file('HISS_Data_1.xlsx')

# Load all 4 data files on Sharepoint
# PEWS_df_1 = Load_file('PEWS_Data_1.xlsx')
# PEWS_df_2 = Load_file('PEWS_Data_2.xlsx')
# PEWS_df = pd.concat([PEWS_df_1, PEWS_df_2])

# HISS_df_1 = Load_file('HISS_Data_1.xlsx')
# HISS_df_2 = Load_file('HISS_Data_2.xlsx')
# HISS_df = pd.concat([HISS_df_1, HISS_df_2])

# Merge the PEWS and HISS Data files
# full_df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')


""" Data Cleaning """

# explore and examine the DataFrame
print('\nDisplaying DataFrame column headers and data types:\n')
print(PEWS_df.dtypes)
# print(PEWS_df.describe())
# print(PEWS_df.head(10))
# print('\n')

# Clean the PEWS Heart Rate Data
PEWS_df.HR = PEWS_df['HR'].replace('\D+', np.NaN, regex = True)   # replace text with null
PEWS_df.HR = pd.to_numeric(PEWS_df.HR)                            # convert to python float/int
PEWS_df.HR = PEWS_df['HR'].dropna()                               # remove null values

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


""" Bin Data by age """

PEWS_bins = [0, 1, 5, 12, 18]                          # Age bins according to UHL PEWS chart categories
PEWS_bin_labels = ['0-11m', '1-4y', '5-11y', '>12']    # Age bin category labels

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
PEWS_df['PEWS_bins'] = pd.cut(PEWS_df.age, PEWS_bins, labels = PEWS_bin_labels)


""" Select the Heart Rate Data """

Heart_Rates = PEWS_df[['HR', 'PEWS_bins', 'age', 'obs_sequence']].values
Heart_Rates = pd.DataFrame(Heart_Rates)

""" Data Analysis """

# Get the Heart rate data for each PEWS Age bin
HR_PEWS_bin_1 = PEWS_df.loc[PEWS_df['PEWS_bins'] == '0-11m']
HR_PEWS_bin_2 = PEWS_df.loc[PEWS_df['PEWS_bins'] == '1-4y']
HR_PEWS_bin_3 = PEWS_df.loc[PEWS_df['PEWS_bins'] == '5-11y']
HR_PEWS_bin_4 = PEWS_df.loc[PEWS_df['PEWS_bins'] == '>12']

HR_PEWS_series = [HR_PEWS_bin_1.HR, HR_PEWS_bin_2.HR, HR_PEWS_bin_3.HR, HR_PEWS_bin_4.HR]

# Plot the HR by PEWS age bins as a histogram
Title = 'Heart rate distribution for 4 age groups'
Range = (50, 200)
# Range = (0, 250)
plot1 = plt.figure(1)
for i in range(len(HR_PEWS_series)):
  plt.hist(HR_PEWS_series[i], range = Range, bins = 50, alpha = 0.4, label = PEWS_bin_labels[i])
plt.title(Title)
plt.legend()
# plt.show()

# Boxplot of Heart Rate for 4 age bins
plot2 = plt.figure(2)
plt.title('Heart rates for 4 different age groups')
sns.boxplot(x = PEWS_df.PEWS_bins, y = PEWS_df.HR, width = 0.3)
# plt.show()

# Boxplots of Heart Rate for each age
plot3 = plt.figure(3)
plt.title('Heart rate range at different ages')
sns.boxplot(x = PEWS_df.age, y = PEWS_df.HR, width = 0.2)
plt.show()

""" Experimenting with adding PEWS Thresholds in plots"""

# HR_centiles = np.quantile(PEWS_df['HR'], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
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


""" UHL PEWS Model Thresholds """

class PEWS_thresholds(object):

  bin_1 = []
  bin_2 = []
  bin_3 = []
  bin_4 = []

  def __init__(self):
    self.bin_1 = [
      ['HR', 20, 90, 1],
      ['HR', 91, 160, 0],
      ['HR', 161, 310, 1]
    ]

    self.bin_2 = [
      ['HR', 20, 90, 1],
      ['HR', 91, 140, 0],
      ['HR', 141, 310, 1]
    ]

    self.bin_3 = [
      ['HR', 20, 70, 1],
      ['HR', 71, 120, 0],
      ['HR', 121, 250, 1]
    ]

    self.bin_4 = [
      ['HR', 20, 60, 1],
      ['HR', 61, 100, 0],
      ['HR', 101, 250, 1]

    ]


exit()