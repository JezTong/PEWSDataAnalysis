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

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt  # pip install matplotlib
from matplotlib.collections import LineCollection
import seaborn as sns  # pip install seaborn

# code to access data files on sharepoint
import File_Access as FA

# Import PEWS model thresholds
import PEWS_thresholds as PT

""" Load the Sharepoint Files """

# limited file load (faster)
PEWS_df = FA.load_file('PEWS_Data_1.xlsx')
HISS_df = FA.load_file('HISS_Data_1.xlsx')

# Load all 4 data files on Sharepoint
# PEWS_df_1 = FA.load_file('PEWS_Data_1.xlsx')
# PEWS_df_2 = FA.load_file('PEWS_Data_2.xlsx')
# PEWS_df = pd.concat([PEWS_df_1, PEWS_df_2])
#
# HISS_df_1 = FA.load_file('HISS_Data_1.xlsx')
# HISS_df_2 = FA.load_file('HISS_Data_2.xlsx')
# HISS_df = pd.concat([HISS_df_1, HISS_df_2])

# Merge the PEWS and HISS Data files
print('\nMerging Data Files...')
df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')
print('\nDisplaying DataFrame Summary:\n')
print(df.describe())


""" Data Exploring """

# explore and examine the DataFrame
print('\nDisplaying DataFrame column headers and data types:\n')
print(df.dtypes)
# print(df.describe())
# print(df.head(10))
# print('\n')

exit()

""" Data Cleaning """

# Clean the PEWS Heart Rate Data
df.HR = df['HR'].replace('\D+', np.NaN, regex=True)  # replace text with null
df.HR = pd.to_numeric(df.HR)  # convert to python float/int
df.HR = df['HR'].dropna()  # remove null values

""" Bin Data by age """

PEWS_bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
PEWS_bin_labels = ['0-11m', '1-4y', '5-11y', '>12']  # Age bin category labels

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
df['PEWS_bins'] = pd.cut(df.age, PEWS_bins, labels=PEWS_bin_labels)

""" Select the Heart Rate Data """

HR = df[['HR', 'age_in_days', 'age', 'PEWS_bins', 'obs_sequence', 'admit_status']].values
HR = pd.DataFrame(HR, columns=['HR', 'age_in_days', 'age', 'PEWS_bins', 'obs_sequence', 'admit_status'])
# print(HR.head())
print(HR.describe())

age_ticks = np.arange(0, 6570, 365).tolist()
# print(age_ticks)
age_labels = list(range(18))



""" Plot a histogram of all heart rates and add PEWS limits """

# PLot the histogram
plot4 = plt.figure(4)
sns.scatterplot(x=HR.age_in_days, y=HR.HR, alpha=0.4, s=5)  # hue = HR.admit_status

# Plot the UHL PEWS Thresholds
UHL_t_table = PT.get_thresolds('UHL_PEWS', 'HR')    # Import the UHL PEWS thresholds for heart rate
lc_1 = LineCollection(PT.generate_lines(UHL_t_table), linewidth=1, color='red')
plt.gca().add_collection(lc_1)

# Now plot the National PEWS thresholds
nat_t_table = PT.get_thresolds('nat_PEWS', 'HR')    # Import the National PEWS thresholds for heart rate
lc_2 = LineCollection(PT.generate_lines(nat_t_table), linewidth=1, color='orange')
plt.gca().add_collection(lc_2)


plt.xlabel('age')
plt.ylabel('HR')
plt.show()

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







