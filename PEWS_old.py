"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, Numpy, Pandas, Mathplotlib, Seaborn,
                    Office365-REST-Python-Client 2.3.1 https://pypi.org/project/Office365-REST-Python-Client/
"""

#TODO Create a virtual Python environment and requirements.txt to allow installation of necessary packages using: python -m pip install -r requirements.txt

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt  # pip install matplotlib
from matplotlib.collections import LineCollection
import seaborn as sns  # pip install seaborn

# code to access data files on sharepoint
import File_Access as FA

# Import PEWS models
import PEWS_models as PM

""" Load the Sharepoint Files """

# limited file load (faster)
# PEWS_df = FA.load_file('PEWS_Data_1.xlsx')
# HISS_df = FA.load_file('HISS_Data_1.xlsx')

# Load all 4 data files on Sharepoint
# PEWS_df_1 = FA.load_file('PEWS_Data_1.xlsx')
# PEWS_df_2 = FA.load_file('PEWS_Data_2.xlsx')
# PEWS_df = pd.concat([PEWS_df_1, PEWS_df_2])
#
# HISS_df_1 = FA.load_file('HISS_Data_1.xlsx')
# HISS_df_2 = FA.load_file('HISS_Data_2.xlsx')
# HISS_df = pd.concat([HISS_df_1, HISS_df_2])

# Merge the PEWS and HISS Data files
# print('\nMerging Data Files...')
# df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')

""" Import Synthetic Observations dataset """

df = pd.read_csv('Data/synthetic_obs.csv')


""" Data Exploring """

# set pandas options to display all columns in a DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# explore and examine the DataFrame
print('\nDisplaying DataFrame Summary:\n')
print(df.describe())
# print(df.head(10))
# print('\nDisplaying DataFrame column headers and data types:\n')
# print(df.dtypes)
print('\n')

# exit()

""" Bin Data by age """

PEWS_bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
PEWS_bin_labels = ['0-11m', '1-4y', '5-11y', '>12y']  # Age bin category labels

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
df['PEWS_bins'] = pd.cut(df.age, PEWS_bins, labels=PEWS_bin_labels)

# TODO change to select the data first before cleaning - faster

""" Data Cleaning """

# Clean the PEWS Heart Rate Data
df.HR = df.HR.replace('\D+', np.NaN, regex=True)  # replace text with null
df.HR = pd.to_numeric(df.HR)  # convert to python float/int
# df.HR = df.HR.dropna()  # remove null values

# Clean the PEWS Respiratory Rate Data
df.RR = df.RR.replace('\D+', np.NaN, regex=True)
df.RR = pd.to_numeric(df.RR)
# df.RR = df.RR.dropna()

# Clean the PEWS Blood Pressure Data
df.BP = df.BP.replace('^\D', np.NaN, regex=True)

""" Select the Heart Rate Data """

HR = df[['HR', 'age_in_days', 'age', 'PEWS_bins']].values
HR = pd.DataFrame(HR, columns=['HR', 'age_in_days', 'age', 'PEWS_bins'])
HR.dropna(inplace=True)
# print(HR.head())
print(HR.describe())
print('\n')

""" Select the Respiratory Rate Data """

RR = df[['RR', 'age_in_days', 'age', 'PEWS_bins']].values
RR = pd.DataFrame(RR, columns=['RR', 'age_in_days', 'age', 'PEWS_bins'])
RR.dropna(inplace=True)
# print(RR.head())
print(RR.describe())
print('\n')

""" Select the Blood Pressure Data """

# Select the BP data and split BP to systolic and diastolic BP columns
BP = df[['BP', 'age_in_days', 'age', 'PEWS_bins']].values
BP = pd.DataFrame(BP, columns=['BP', 'age_in_days', 'age', 'PEWS_bins'])
BP_temp = BP['BP'].str.split('/', n=1, expand=True)
BP['sBP'] = BP_temp[0]
BP['dBP'] = BP_temp[1]
BP.drop(columns=['BP'], inplace=True)
BP.dropna(inplace=True)
BP.sBP = BP.sBP.apply(pd.to_numeric)
BP.dBP = BP.dBP.apply(pd.to_numeric)

# print(BP)
print(BP.describe())
print('\n')



""" Plot a histogram of all heart rates and add PEWS limits """

# age_ticks = np.arange(0, 6570, 365).tolist()
# # print(age_ticks)
# age_labels = list(range(18))

# PLot the histogram
plot1= plt.figure(1)
sns.scatterplot(x=HR.age_in_days, y=HR.HR, alpha=0.2, s=5)  #hue=HR.admit_status
plt.ylim([0, 250])

# Plot the thresholds - option 1 brute force
# plt.gca().add_collection(PM.generate_lines('UHL_PEWS', 'HR'))

# generate the thresholds - option 2 computabale table
scores = [0, 1, 2, 4]
for score in scores:
    # generate the threshold tables
    threshold = PM.generate_thresholds_table('nat_PEWS', 'HR', score)
    # Plot the thresholds
    plt.plot(threshold.age, threshold.lower, color='red', linewidth=0.5)
    plt.plot(threshold.age, threshold.upper, color='red', linewidth=0.5)

plt.xlabel('Age in Days')
plt.ylabel('Heart Rates per min')
plt.title('Heart rates with National PEWS Thresholds')
# plt.savefig('Nat_PEWS_HR.png')

# PLot the histogram
plot2= plt.figure(2)
sns.scatterplot(x=HR.age_in_days, y=HR.HR, alpha=0.2, s=5)  #hue=HR.admit_status
plt.ylim([0, 250])

# generate the threshold tables
threshold_UHL = PM.generate_thresholds_table('UHL_PEWS', 'HR', 0)

# Plot the thresholds - option 2 computabale table
plt.plot(threshold_UHL.age, threshold_UHL.lower, color='purple', linewidth=0.5)
plt.plot(threshold_UHL.age, threshold_UHL.upper, color='purple', linewidth=0.5)

plt.xlabel('Age in Days')
plt.ylabel('Heart Rates per min')
plt.title('Heart Rates with UHL PEWS Thresholds')
# plt.savefig('UHL_PEWS_HR.png')


# plt.show()

# exit()


""" Plot a histogram of all respiratory rates and add PEWS limits """

# PLot the histogram
plot3 = plt.figure(3)
sns.scatterplot(x=RR.age_in_days, y=RR.RR, alpha=0.2, s=5, color='mediumseagreen' )  #hue=RR.admit_status

# generate the thresholds
scores = [0, 1, 2, 4]
for score in scores:
    # generate the threshold tables
    threshold = PM.generate_thresholds_table('nat_PEWS', 'RR', score)
    # Plot the thresholds - option 2 computabale table
    plt.plot(threshold.age, threshold.lower, color='red', linewidth=0.5)
    plt.plot(threshold.age, threshold.upper, color='red', linewidth=0.5)

plt.xlabel('Age in days')
plt.ylabel('Respiratory Rates per min)')
plt.title('Respiratory Rates with National PEWS Thresholds')
# plt.savefig('Nat_PEWS_RR.png')

# PLot the histogram
plot4 = plt.figure(4)
sns.scatterplot(x=RR.age_in_days, y=RR.RR, alpha=0.2, s=5, color='mediumseagreen' )  #hue=HR.admit_status

# generate the threshold tables
threshold_UHL = PM.generate_thresholds_table('UHL_PEWS', 'RR', 0)

# Plot the thresholds - option 2 computabale table
plt.plot(threshold_UHL.age, threshold_UHL.lower, color='purple', linewidth=0.5)
plt.plot(threshold_UHL.age, threshold_UHL.upper, color='purple', linewidth=0.5)

plt.xlabel('Age in Days')
plt.ylabel('Respiratory Rates per min')
plt.title('Respiratory Rates with UHL PEWS Thresholds')
# plt.savefig('UHL_PEWS_RR.png')
plt.show()

exit()

# """ Plot a histogram of all Systolic Blood Pressure data and add PEWS limits """
#
# # PLot the histogram
# plot5 = plt.figure(5)
# sns.scatterplot(x=BP.age_in_days, y=BP.sBP, alpha=0.2, s=5, color='mediumpurple' )  #hue=sBP.admit_status
# # plt.yticks(np.arange(0, 160, 10))
# plt.ylim([20, 180])
#
# # generate the thresholds
# scores = [0, 1, 2, 4]
# for score in scores:
#     # generate the threshold tables
#     threshold = PM.generate_thresholds_table('nat_PEWS', 'sBP', score)
#     # Plot the thresholds - option 2 computabale table
#     plt.plot(threshold.age, threshold.lower, color='red', linewidth=0.5)
#     plt.plot(threshold.age, threshold.upper, color='red', linewidth=0.5)
#
# plt.xlabel('Age in days')
# plt.ylabel('Systolic Blood Pressure in mmHg')
# plt.title('Systolic Blood Pressure with National PEWS Thresholds')
# plt.savefig('Nat_PEWS_sBP.png')
#
# # PLot the histogram
# plot6 = plt.figure(6)
# sns.scatterplot(x=BP.age_in_days, y=BP.sBP, alpha=0.2, s=5, color='mediumpurple' )  #hue=sBP.admit_status
# # plt.yticks(np.arange(0, 160, 10))
# plt.ylim([20, 180])
#
# # generate the threshold tables
# threshold_UHL = PM.generate_thresholds_table('UHL_PEWS', 'sBP', 0)
#
# # Plot the thresholds - option 2 computabale table
# plt.plot(threshold_UHL.age, threshold_UHL.lower, color='red', linewidth=0.5)
# plt.plot(threshold_UHL.age, threshold_UHL.upper, color='red', linewidth=0.5)
#
# plt.xlabel('Age in days')
# plt.ylabel('Systolic Blood Pressure in mmHg')
# plt.title('Systolic Blood Pressure with UHL PEWS Thresholds')
# plt.savefig('UHL_PEWS_sBP.png')
# plt.show()

exit()



""" Data Analysis """

# Get the Heart rate data for each PEWS Age bin
HR_PEWS_bin_0 = df.loc[df.PEWS_bins == '0-11m']
HR_PEWS_bin_1 = df.loc[df.PEWS_bins == '1-4y']
HR_PEWS_bin_2 = df.loc[df.PEWS_bins == '5-11y']
HR_PEWS_bin_3 = df.loc[df.PEWS_bins == '>12']

HR_PEWS_series = [HR_PEWS_bin_0.HR, HR_PEWS_bin_1.HR, HR_PEWS_bin_2.HR, HR_PEWS_bin_3.HR]

# Plot the HR by PEWS age bins as a histogram
Title = 'Heart rate distribution for 4 age groups'
Range = (50, 200)
# Range = (0, 250)
fig1 = plt.figure(1)
for i in range(len(HR_PEWS_series)):
    plt.hist(HR_PEWS_series[i], range=Range, bins=50, alpha=0.4, label=PEWS_bin_labels[i])
plt.title(Title)
plt.legend()
# plt.show()

# Boxplot of Heart Rate for 4 age bins
fig2 = plt.figure(2)
plt.title('Heart rates for 4 different age groups')
sns.boxplot(x=df.PEWS_bins, y=df.HR, width=0.3)
# plt.show()

# Boxplots of Heart Rate for each age
fig3 = plt.figure(3)
plt.title('Heart rate range at different ages')
sns.boxplot(x=df.age, y=df.HR, width=0.2)
plt.show()

""" Experimenting with adding centiles in plots"""

# HR_centiles = np.quantile(df['HR'], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
# centile_line_color = ['red', 'orange', 'yellow', 'green', 'yellow', 'orange', 'red']
# plt.clf()
# plt.hist(df['HR'], range = (0, 250), bins = 25, edgecolor = 'white')

# for n in [0, 1, 2, 3, 4, 5, 6]:
#   plt.axvline(x = HR_centiles[n], color = centile_line_color[n])

# for centile in HR_centiles:
#   plt.axvline(x = centile, color = 'red')

# plt.title('Heart Rate')
# plt.show()







