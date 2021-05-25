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
import seaborn as sns  # pip install seaborn
import statsmodels.api as sm # pip install statsmodels

# Import PEWS models
import PEWS_models as PM

""" Load Data Files """

def load_sharepoint_file(file_scope='half'):
    # function to load PEWS data file from Sharepoint account
    # file_scope: 'half' = limited (faster), 'full' = load full database

    # code to access data files on Sharepoint
    import File_Access as FA

    if file_scope == 'half':
        # load 2 of 4 data files on Sharepoint
        PEWS_df = FA.load_file('PEWS_Data_1.xlsx')
        HISS_df = FA.load_file('HISS_Data_1.xlsx')

    else:
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
        return df


def load_synthetic_data():
    # function to import Synthetic Observations dataset
    df = pd.read_csv('Data/synthetic_obs.csv')
    return df


def load_saved_data(parameter='HR'):
    # function to load previously processed and saved data (rapid file load for code development)
    df = pd.read_csv(f'data/{parameter}.csv')
    df.rename(columns={'age':'age_in_days'}, inplace=True)
    return df


""" Initial Data Explore """


def explore_data(df):
    # function to explore the raw dataframe
    print('\nDisplaying DataFrame Summary:\n')

    # set pandas options to display all columns in a DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # explore and examine the DataFrame
    print(df.describe())
    # print(df.columns)
    # print(df.head(10))
    print('\n')


""" Data Processing """

def select_parameter(df, parameter):
    # creates a new dataframe with the single parameter from the PEWS dataframe
    # renames the columns as age and parameter name
    parameter_df = df[['age_in_days', parameter]].values
    parameter_df = pd.DataFrame(parameter_df, columns=['age', parameter])
    print(f'\n...{parameter} DataFrame created...')
    return parameter_df


# TODO this is not quite ready yet
def calculate_decimal_age(parameter_df):
    # converts age in days to decimal age
    # drops age_in_days column
    # reorders the columns - important for later functions to work
    parameter_df['age'] = parameter_df['age_in_days'] / 365.25
    parameter_df.drop(columns=['age_in_days'], inplace=True)
    par_name = parameter_df.columns[1]
    parameter_df = parameter_df[['age', par_name]]
    print('\n...Converted age in days to decimal age...')
    print(parameter_df.head(10))
    return parameter_df


def split_BP(parameter_df):
    # if the parameter is BP, splits the BP data into systolic BP and diastolic BP columns.
    # otherwise does not alter the data
    if 'BP' in parameter_df:
        BP = parameter_df['BP'].str.split('/', n=1, expand=True)
        parameter_df['sBP'] = BP[0]
        # parameter_df['dBP'] = BP[1]
        parameter_df.drop(columns=['BP'], inplace=True)
        print(f'\n...Extracting sBP from BP column...')
        return parameter_df
    else:
        return parameter_df


def clean_data(parameter_df):
    # takes the parameter dataframe and converts text to NaN valuse
    # removes missing values
    # converts data types from objects to integers
    par_name = parameter_df.columns[1]
    parameter_df[par_name] = parameter_df[par_name].replace('\D+', np.NaN, regex=True)
    parameter_df.dropna(inplace=True)
    for i in list(parameter_df):
        parameter_df[i] = parameter_df[i].astype(int)
    print(f'\n...{par_name} Data Cleaning Complete...')
    return parameter_df


def print_data(parameter_df):
    # prints the parameter dataframe and its summary statistics
    par_name = parameter_df.columns[1]
    # print(f'\n{par_name} Dataframe:\n')
    # print(parameter_df)
    # print(parameter_df.dtypes)
    print(f'\nSummary Statistics for {par_name}:\n')
    print(parameter_df.describe())
    return parameter_df


def bin_by_age(parameter_df, bins=54):
    # categorises the parameter data by time intervals
    # number of bins for specific time intervals: 1 month = 216, 2 months = 108, 4 months = 54, 6 months = 36
    par_name = parameter_df.columns[1]
    parameter_df['bin'] = pd.cut(parameter_df.age, bins=bins)
    print(f'\n...{par_name} data binned by age intervals...')
    print(parameter_df.head(10))
    return parameter_df


def color_selector(par_name):
    # function to select a color based on the parameter
    # TODO switch to list comprehension
    if par_name == 'HR':
        return 'blue'
    elif par_name == 'RR':
        return 'green'
    else:
        return 'mediumpurple'


def format_plot(par_name, chart_type):
    # function to add chart title, axis labels, show plot and save plot as .png
    plt.xlabel('Age in Days')
    plt.ylabel(f'{par_name}')
    plt.title(f'{par_name} in Children')
    plt.savefig(f'plots/{par_name}_{chart_type}_plot.png')
    plt.show()


def plot_scatter(parameter_df):
    # function to plot a scatter plot of the parameter data
    par_name = parameter_df.columns[1]
    chart_type = 'scatter'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)
    format_plot(par_name, chart_type)
    return parameter_df


""" Calculate the centiles """


def linear_regression(parameter_df):
    # function to plot a linear regression of the parameter
    par_name = parameter_df.columns[1]
    chart_type = 'regression'
    centile_df = parameter_df

    centile_df['median'] = centile_df[par_name].rolling(1).median()
    print(centile_df.head())

    model = sm.OLS.from_formula('median ~ age', data=centile_df).fit()
    print('\n')
    print(model.params)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # ax = plt.plot(centile_df.age, quantile_50,  color='red', linewidth=1)

    ax = plt.plot(centile_df.age, model.params[0] + model.params[1] * centile_df['median'], color='green', linewidth=1)

    format_plot(par_name, chart_type)
    return parameter_df



def plot_centiles(parameter_df, lower_quintile=0.05, mid_quintile=0.5, upper_quintile=0.95):
    # need to plot scatter plot of the parameter first
    par_name = parameter_df.columns[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lmplot(x='age', y=par_name, data=parameter_df, order=2)

    # create a centile dataframe to calculate the centiles
    centile_df = parameter_df
    # bin the HR data by age
    bins = 18  # 1 month = 216, 2 months = 108, 4 months = 54, 6 months = 36
    # classify age according to age bins and add an Age bin column to the PEWS Dataframe
    centile_df['bin'] = pd.cut(centile_df.age, bins=bins)
    # calculate the centiles for HR
    quantile_50 = centile_df.groupby('bin', as_index=False, sort=True).quantile(0.5)  # 50th centiles
    quantile_50 = pd.DataFrame(quantile_50)

    # plot the centile line
    ax = plt.plot(centile_df.age, quantile_50,  color='red', linewidth=1)

    format_plot(par_name)
    return parameter_df


def save_as_csv(parameter_df):
    # saves the parameter dataframe as a csv file for quick analysis later
    par_name = parameter_df.columns[1]
    parameter_df.to_csv(f'data/{par_name}.csv')
    return parameter_df


""" Sequential Function Call """

parameter_list = ['HR']
for parameter in parameter_list:
    # takes the dataframe and processes in sequence
    df = load_saved_data(parameter)
    process = (
        select_parameter(df, parameter)
        .pipe(print_data)
        .pipe(linear_regression)

    )

exit()

# Quick access for .pipe functions (copy & paste these into process above)
        # .pipe(print_data)
        # .pipe(split_BP)
        # .pipe(clean_data)
        # .pipe(print_data)
        # .pipe(bin_by_age)
        # .pipe(plot_scatter)
        # .pipe(plot_centiles)
        # .pipe(linear_regression)
        # .pipe(save_as_csv)





""" Test Code """
# the following is test code used for development and not used in analysis


# bin the HR data by age
bins = 18  # 1 month = 216, 2 months = 108, 4 months = 54, 6 months = 36
# [np.arange(0, 6574, 28)]  # Age bins according to PEWS chart categories
# bin_labels = [np.arange(0, len(bins))]  # Age bin category labels

# print('\nbins\n')
# print(bins)
# print(bin_labels)

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
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
HR_std = pd.DataFrame(HR_std)  # , columns=['age', 'bin', 'HR']
print('\nHR standard deviations\n')
print(HR_std)

# TODO match up centiles with the age markers better

# calculate the centiles for HR
HR_qtiile_95 = df_HR.groupby('bin', as_index=False, sort=True).quantile(0.95)  # 95th centiles
HR_qtiile_95 = pd.DataFrame(HR_qtiile_95)

HR_qtiile_5 = df_HR.groupby('bin', as_index=False, sort=True).quantile(0.05)  # 5th centiles
HR_qtiile_5 = pd.DataFrame(HR_qtiile_5)

# print('\nHR centiles\n')
# print(HR_qtiile_5)
# print(HR_qtiile_95)


centile_df['average'] = centile_df[par_name].rolling(1).mean()
centile_df['average'] = centile_df[par_name].groupby(centile_df.age, sort=True).quantile(0.5)
centile_df = pd.DataFrame(centile_df)

centile_df['age'] = pd.cut(centile_df.age, bins=28)
centile_df = centile_df.groupby('bin', as_index=False, sort=True).mean()
centile_df = pd.DataFrame(centile_df, columns=['age', 'bin', 'HR'])

# centile_df['median'] = centile_df[par_name].rolling(window).median()
# centile_df['P5'] = centile_df[par_name].rolling(window).quantile(0.05)
# centile_df['P95'] = centile_df[par_name].rolling(window).quantile(0.95)
# print(centile_df.head(240))

# plot the centile lines
# ax = sns.lineplot(x=centile_df['age'], y=centile_df['median'].median(), data=centile_df, color='red', label='median')
# ax = sns.lineplot(x=centile_df['age'], y=centile_df['P5'], data=centile_df, color='red', label='P5')
# ax = sns.lineplot(x=centile_df['age'], y=centile_df['P95'], data=centile_df, color='red', label='P95')
# # colour in the area between upper and lower centiles
#plt.fill_between(centile_df['age'], centile_df['P5'], centile_df['P5'], color='red', alpha=0.05)




# plot the HR data
sns.scatterplot(x=df_HR.age, y=df_HR.HR, alpha=0.2, s=5)

# plot the HR mean and centiles for age
sns.lineplot(x=HR_qtiile_95.age, y=HR_qtiile_95.HR, data=HR_qtiile_95, linewidth=1, ls='--', color='red',
             label='95th centile')
sns.lineplot(x=HR_mean.age, y=HR_mean.HR, data=HR_mean, linewidth=1, color='red', label='mean')
sns.lineplot(x=HR_qtiile_5.age, y=HR_qtiile_5.HR, data=HR_qtiile_5, linewidth=1, ls='--', color='red',
             label='5th centile')

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
