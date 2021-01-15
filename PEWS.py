"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.1
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, csv, Numpy, Pandas, Requests, getpass
                    Office365-REST-Python-Client 2.3.1 https://pypi.org/project/Office365-REST-Python-Client/
"""

# Import Python Modules
import io
import csv
import numpy as np # pip install numpy
import pandas as pd # pip install pandas
import requests # pip install requests
import getpass
from matplotlib import pyplot as plt

# Import office365 share point API elements # pip install Office365-REST-Python-Client
from office365.runtime.auth.user_credential import UserCredential
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.runtime.client_request import ClientRequest
from office365.sharepoint.files.file import File


sharepoint_url = "https://uhltrnhsuk.sharepoint.com"
username = 'jeremy.tong@uhl-tr.nhs.uk' # input('username: ')
password = 'EdWhite.1' # getpass.getpass()
site_url = "https://uhltrnhsuk.sharepoint.com/sites/case"
folder_url = "/sites/case/Shared%20Documents/PEWSDataAnalysis/"
file_url = "/sites/case/Shared%20Documents/PEWSDataAnalysis/PEWS_NC_Test_200.csv"
filename = "PEWS_NC_TEST_200.csv"


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


# Download the file to temporary memory
response = File.open_binary(ctx, file_url)
# save data to BytesIO stream
bytes_file_obj = io.BytesIO()
bytes_file_obj.write(response.content)
bytes_file_obj.seek(0)  # set file object to start

# create a DataFrame form the PEWS data (can be excel file with .read_excel)
PEWS_df = pd.read_csv(bytes_file_obj)

# explore and examine the DataFrame
print(PEWS_df.describe())
print(PEWS_df.columns)
print(PEWS_df.head(10))

# explore the Heart Rate data and plot a histogram of heart rates
HR = PEWS_df['Heart Rate'].values

HR_min = np.amin(HR)
HR_max = np.amax(HR)
range = HR_max - HR_min
print('\nThe lowest heart rate is {:.2f} beats per min'.format(HR_min))
print('\nThe highest heart rate is {:.2f} beats per min'.format(HR_max))

HR_centiles = np.quantile(HR, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
centile_line_color = ['red', 'orange', 'yellow', 'green', 'yellow', 'orange', 'red']

plt.hist(HR, range = (30, 150), bins = 12, edgecolor = 'white')

for n in [0, 1, 2, 3, 4, 5, 6]:
  plt.axvline(x = HR_centiles[n], color = centile_line_color[n])

# for centile in HR_centiles:
#   plt.axvline(x = centile, color = 'red')

# plt.axvline(x = HR_centiles[0], c = 'red') # 5th centile
# plt.axvline(x = HR_centiles[1], c = 'orange') # 10th centile
# plt.axvline(x = HR_centiles[2], c = 'yellow') # 25th centile
# plt.axvline(x = HR_centiles[3], c = 'green') # 50th centile
# plt.axvline(x = HR_centiles[4], c = 'yellow') # 75th centile
# plt.axvline(x = HR_centiles[5], c = 'orange') # 90th centile
# plt.axvline(x = HR_centiles[6], c = 'red') # 95th centile

plt.show()

""" Some sample code to work with """
# Example code to calculate standard deviation
# standard_deviation = np.std(array)

# Example code to calculate min, max & range
# minimum = np.amin(array)
# maximum = np.amax(array)
# range = maximum - minimum

# Example code to calculate percentiles
# fiftieth_centile = np.quantile(array, 0.5)
# centiles = np.quantile(array, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
# Plot the centiles on a histogram
# for centile in centiles:
#   plt.axvline(x = centile, color = 'red')

# Example code to plot a histogram
# Save transaction times to a separate numpy array
# array = dataframe["Column name"].values
# plt.figure(1) # multiple plots in same figure
# plt.subplot(211) # sub plot in same figure
# plt.hist(array, range = (min_value, max_value), bins = num_of_bars)
# plt.title('Chart Title')
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.tight_layout() # prevent the labels from overlapping with the graphs
# plt.show()

