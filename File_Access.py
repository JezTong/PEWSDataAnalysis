"""
    NAME:          PEWSDataAnalysis: File_Access.py
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for accessing PEWS Data on remote server for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, Numpy, Pandas, Mathplotlib, Seaborn, python-decouple,
                    Office365-REST-Python-Client 2.3.1 https://pypi.org/project/Office365-REST-Python-Client/
"""

# Import Python Modules
import io
import pandas as pd  # pip install pandas

# Import office365 share point API elements # pip install Office365-REST-Python-Client
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.runtime.client_request import ClientRequest
from office365.sharepoint.files.file import File

# Import Authentication tokens
from decouple import config

""" Sharepoint File Access """

sharepoint_url = config('sharepoint')
username = config('username')   #input('username: ')
password = config('password')   #getpass.getpass()
site_url = config('site')
folder_url = config('folder')

# Access sharepoint folder with authentication
ctx_auth = AuthenticationContext(sharepoint_url)
if ctx_auth.acquire_token_for_user(username, password):
    request = ClientRequest(ctx_auth)
    ctx = ClientContext(site_url, ctx_auth)
    web = ctx.web
    ctx.load(web)
    ctx.execute_query()
    print(f"\n...Logged into: {web.properties['Title']}")

else:
    print(ctx_auth.get_last_error())


# Function to load a file from sharepoint onto python
def load_file(filename):
    # Open the file in temporary memory
    response = File.open_binary(ctx, str(folder_url + filename))
    bytes_file_obj = io.BytesIO()
    bytes_file_obj.write(response.content)

    # set file object to start
    bytes_file_obj.seek(0)

    global df
    print(f'\n...Loading {filename}...')

    if filename.endswith('.xlsx'):
        dict = pd.read_excel(bytes_file_obj, sheet_name='Sheet1')  # read the excel file in python as a dictionary
        df = pd.DataFrame.from_dict(dict)  # convert dictionary to a dataframe using pandas
        return df

    elif filename.endswith('.csv'):
        df = pd.read_csv(bytes_file_obj)  # read the .csv file in python as a dataframe
        return df

    else:
        file = input("\n*** File not recognised, please try again: ")
        Load_file(file)
