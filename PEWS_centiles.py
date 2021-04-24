"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.2.3
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, Numpy, Pandas, Mathplotlib, Seaborn
"""

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
PEWS_df_1 = FA.load_file('PEWS_Data_1.xlsx')
PEWS_df_2 = FA.load_file('PEWS_Data_2.xlsx')
PEWS_df = pd.concat([PEWS_df_1, PEWS_df_2])

HISS_df_1 = FA.load_file('HISS_Data_1.xlsx')
HISS_df_2 = FA.load_file('HISS_Data_2.xlsx')
HISS_df = pd.concat([HISS_df_1, HISS_df_2])

# Merge the PEWS and HISS Data files
print('\nMerging Data Files...')
df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')

""" Import Synthetic Observations dataset """

# df = pd.read_csv('Data/synthetic_obs.csv')


""" Data Exploring """
print('\nDisplaying DataFrame Summary:\n')

# set pandas options to display all columns in a DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# explore and examine the DataFrame
print(df.describe())
# print(df.columns)
# print(df.head(10))
print('\n')

""" Calculate the centiles """

# take in a DataFrame of the parameter in question

# select HR as first parameter
df_HR = df[['age_in_days', 'HR']].values
df_HR = pd.DataFrame(df_HR, columns=['age', 'HR'])

# clean the HR data
df_HR.HR = df_HR.HR.replace('\D+', np.NaN, regex=True)  # replace text with null
df_HR.dropna(inplace=True) # remove NaN values
df_HR.HR = df_HR.HR.astype(float) # convert object to float
df_HR.age = df_HR.age.astype(int) # convert object to integer

# print('\nHR Dataframe 1\n')
# print(df_HR)

# bin the HR data by age
bin_size = 28
bins = 18 # 1 month = 216, 2 months = 108, 4 months = 54, 6 months = 36
    # [np.arange(0, 6574, 28)]  # Age bins according to PEWS chart categories
# bin_labels = [np.arange(0, len(bins))]  # Age bin category labels

# print('\nbins\n')
# print(bins)
# print(bin_labels)

# classify age according to age bins and add an Ag e bin column to the PEWS Dataframe
df_HR['bin'] = pd.cut(df_HR.age, bins=bins)
# print('\nHR Dataframe 2\n')
# print(df_HR)

# calculate the mean values for each age bin
HR_mean = df_HR.groupby('bin', as_index=False, sort=True).mean()
HR_mean = pd.DataFrame(HR_mean, columns=['age', 'bin', 'HR'])
# print('\nHR means\n')
# print(HR_mean)

# TODO figure out how to plot the standard deviation

HR_std = df_HR.groupby('bin', as_index=False, sort=True).std()
HR_std = pd.DataFrame(HR_std)# , columns=['age', 'bin', 'HR']
print('\nHR standard deviations\n')
print(HR_std)


# TODO match up cenyiles with the age markers better

# calculate the centiles for HR
HR_qtiile_95 = df_HR.groupby('bin', as_index=False, sort=True).quantile(0.95)  # 95th centiles
HR_qtiile_95 = pd.DataFrame(HR_qtiile_95)

HR_qtiile_5 = df_HR.groupby('bin', as_index=False, sort=True).quantile(0.05) # 5th centiles
HR_qtiile_5 = pd.DataFrame(HR_qtiile_5)

# print('\nHR centiles\n')
# print(HR_qtiile_5)
# print(HR_qtiile_95)

# plot the HR data with mean and centiles
plot1 = plt.figure(1)

# plot the HR data
sns.scatterplot(x=df_HR.age, y=df_HR.HR, alpha=0.2, s=5)

#plot the HR mean and centiles for age
sns.lineplot(x=HR_qtiile_95.age, y=HR_qtiile_95.HR, data=HR_qtiile_95, linewidth=1, ls='--', color='red', label='95th centile')
sns.lineplot(x=HR_mean.age, y=HR_mean.HR, data=HR_mean, linewidth=1, color='red', label='mean')
sns.lineplot(x=HR_qtiile_5.age, y=HR_qtiile_5.HR, data=HR_qtiile_5, linewidth=1, ls='--', color='red', label='5th centile')

# colour in the area between upper and lower centiles
plt.fill_between(HR_mean.age, HR_qtiile_5.HR, HR_qtiile_95.HR, color='red', alpha=0.05)

plt.ylim([0, 250])

# plot the figure labels
plt.xlabel('Age in Days')
plt.ylabel('Heart Rates per min')
plt.title('Heart Rate Centiles (1 year averages)')
plt.legend(loc='upper right')
plt.savefig('HR_centiles.png')
plt.show()

exit()

# sort the dataframe by age_in_days
HR = HR.sort_values('age_in_days', ascending=True)
print(HR.head())

# select a number samples as a sample window
band = 50

# take the window and work out its mean and standard deviation
test = HR.iloc[0:49].mean()
print(test)





# calculate the sample size for this window (determine what the minimum sample size should be)

# calculate the centile for each age using the 'window' with 'acceptable limits'