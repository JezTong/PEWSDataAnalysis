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

# Import office365 share point API elements # pip install Office365-REST-Python-Client
from office365.runtime.auth.user_credential import UserCredential
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.runtime.client_request import ClientRequest
from office365.sharepoint.files.file import File


sharepoint_url = "https://uhltrnhsuk.sharepoint.com"
username = input('username: ')
password = getpass.getpass()
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
  print("Logged into: {0}".format(web.properties['Title']))

else:
  print (ctx_auth.get_last_error())


# Download the file to memory
response = File.open_binary(ctx, file_url)
# save data to BytesIO stream
bytes_file_obj = io.BytesIO()
bytes_file_obj.write(response.content)
bytes_file_obj.seek(0)  # set file object to start

# create a DataFrame form the PEWS data (can be excel file with .read_excel)
PEWS_df = pd.read_csv(bytes_file_obj)
print(PEWS_df.head(10))
