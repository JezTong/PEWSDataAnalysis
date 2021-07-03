"""
    NAME:          PEWSDataAnalysis
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.1
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DEPENDENCIES:  This program requires the following modules:
                    io, Numpy, Pandas, Mathplotlib, Seaborn, Statsmodels
"""

# Import Python Modules
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt  # pip install matplotlib
import seaborn as sns  # pip install seaborn
import statsmodels.api as sm  # pip install statsmodels
import statsmodels.formula.api as smf


""" Load Data Files """


def load_sharepoint_file(file_scope='full'):
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
    print('\n...Merging Data Files...')
    df = pd.merge(PEWS_df, HISS_df, on='spell_id', how='outer')
    return df


def load_synthetic_data():
    # function to import Synthetic Observations dataset
    df = pd.read_csv('Data/synthetic_obs.csv')
    return df


def load_saved_data(parameter):
    # function to load previously processed and saved data (rapid file load for code development)
    parameter_df = pd.read_csv(f'data/{parameter}.csv')
    parameter_df = pd.DataFrame(parameter_df)
    # parameter_df.rename(columns={'age': 'age_in_days'}, inplace=True)
    return parameter_df


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
    return(df)


""" Data Processing """


def select_parameter(df, parameter):
    # creates a new dataframe with the single parameter from the PEWS dataframe
    # renames the columns as age and parameter name
    parameter_df = df[['age_in_days', parameter]].values
    parameter_df = pd.DataFrame(parameter_df, columns=['age_in_days', parameter])
    print(f'\n...{parameter} DataFrame created...')
    return parameter_df


def split_BP(parameter_df):
    # if the parameter is BP, splits the BP data into systolic BP and diastolic BP columns.
    # otherwise does not alter the data
    # do this before clean_data of values will be lost
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
    # takes the parameter dataframe and converts text to NaN values
    # removes missing values
    # converts data types from objects to integers
    par_name = parameter_df.columns[1]
    parameter_df[par_name] = parameter_df[par_name].replace(r'\D+', np.NaN, regex=True)
    parameter_df.dropna(inplace=True)
    for i in list(parameter_df):
        parameter_df[i] = parameter_df[i].astype(int)
    print(f'\n...{par_name} Data Cleaning Complete...')
    return parameter_df


def convert_decimal_age(parameter_df):
    # converts the age in days to a decimal age
    parameter_df['age'] = parameter_df['age_in_days'] / 365.25
    parameter_df.drop(['age_in_days'], axis=1)
    print('\n...Converted age in days to decimal age...')
    # print(parameter_df.head(10))
    return parameter_df


def print_data(parameter_df):
    # prints the parameter dataframe and its summary statistics
    par_name = parameter_df.columns[1]
    # print(f'\n{par_name} Dataframe:\n')
    # print(parameter_df)
    # print(parameter_df.dtypes)
    print('\n')
    print('=' * 80)
    print(f'\nSummary Statistics for {par_name}:\n')
    print(parameter_df.describe())
    return parameter_df


""" Simple data pLots """


def color_selector(par_name):
    # function to select a color based on the parameter
    if par_name == 'HR':
        return 'royalblue'
    elif par_name == 'RR':
        return 'mediumseagreen'
    elif par_name == 'sats':
        return 'mediumturquoise'
    else:
        return 'mediumpurple'


def format_plot(par_name, chart_type, ax):
    # function to add chart title, axis labels, show plot and save plot as .png
    plt.xlabel('Age in years')
    plt.ylabel(f'{par_name}')
    plt.title(f'{chart_type} For {par_name} in children')
    plt.savefig(f'plots/{par_name}_{chart_type}_plot.png')
    plt.show()


def plot_scatter(parameter_df):
    # function to plot a scatter plot of the parameter data
    par_name = parameter_df.columns[1]
    chart_type = 'Scatter plot'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)
    format_plot(par_name, chart_type, ax)
    return parameter_df


""" Data plots with regression analysis """


def linear_regression(parameter_df):
    # function to plot a linear regression of the parameter
    par_name = parameter_df.columns[1]
    chart_type = 'regression'

    # fit the median values to a simple linear regression model
    model = sm.OLS.from_formula(f'{par_name} ~ age', data=parameter_df).fit()
    print('\n')
    print(model.params, '\n', model.summary().extra_txt)

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # plot the regression line overlaid on parameter scatter plot
    ax.plot(parameter_df.age, model.params[0] + model.params[1] * parameter_df.age, color='orange', linewidth=1)

    # format the chart and save as .png
    format_plot(par_name, chart_type, ax)
    return parameter_df


def polynomial_regression(parameter_df):
    # function to plot a linear regression of the parameter
    par_name = parameter_df.columns[1]
    chart_type = 'polynomial'

    # fit the median values to a simple linear regression model
    model = sm.OLS.from_formula(f'{par_name} ~ age + np.power(age,2)', data=parameter_df).fit()
    print('\n', model.params, '\n', model.summary().extra_txt)

    """ check the modeling assumptions of normality and homoscedasticity of the residuals """

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # plot the regression line overlaid on parameter scatter plot
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 50)
    y = model.params[0] + model.params[1] * x + model.params[2] * np.power(x, 2)

    ax.plot(x, y, linestyle='--', color='red', linewidth=1)

    # format the chart and save as .png
    format_plot(par_name, chart_type, ax)
    return parameter_df


def quantile_regression(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    chart_type = 'OLS regression centiles'

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age', parameter_df)
    result = model.fit(q=0.5)
    print(result.summary())

    quantiles = [0.05, 0.5, 0.95]

    # function to fit regression model to a quantile 'q'
    def fit_model(q):
        result = model.fit(q=q)
        return [q, result.params['Intercept'], result.params['age']] + result.conf_int().loc['age'].tolist()

    # fit the regression model to quantiles and convert it to a dataframe
    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

    # Ordinary Least Squared model predicting parameter based on age
    ols = smf.ols(f'{par_name} ~ age', parameter_df).fit()
    ols_ci = ols.conf_int().loc['age'].tolist()
    ols = dict(a=ols.params['Intercept'],
               b=ols.params['age'],
               lb=ols_ci[0],
               ub=ols_ci[1])

    print(models)
    print(ols)

    # set up the regression data and plotting it
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 50)
    get_y = lambda a, b: a + b * x

    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i])
        ax.plot(x, y, linestyle='dotted', color='red')

    y = get_y(ols['a'], ols['b'])

    ax.plot(x, y, color='red', label='OLS')

    ax.legend()
    format_plot(par_name, chart_type, ax)
    return parameter_df


def poly_quantile_regression_1(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    chart_type = 'Polynomial quantile regression for y = m + x + x^2 '

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age + np.power(age, 2)', parameter_df)
    result = model.fit(q=.5)
    print('\n')
    print(chart_type)
    print('\n')
    print(result.summary())
    print('\n')
    print(result.params)

    # quantile lines to display
    quantiles = [.95, .75, .5, .25, .05]

    def fit_model(q):
        # function to apply LAD model to the data
        result = model.fit(q=q)
        return [
                   q,
                   result.params['Intercept'],
                   result.params['age'],
                   result.params['np.power(age, 2)'],
               ] + result.conf_int().loc['age'].tolist()

    # apply ALD model for each quantile in list & convert to dataframe for plotting
    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b', 'c', 'lb', 'ub'])

    ols = smf.ols(f'{par_name} ~ age + np.power(age, 2)', parameter_df).fit()
    ols_ci = ols.conf_int().loc['age'].tolist()
    ols = dict(a=ols.params['Intercept'],
               b=ols.params['age'],
               c=ols.params['np.power(age, 2)'],
               lb=ols_ci[0],
               ub=ols_ci[1])

    print('\n')
    print(models)
    print('\n')
    print(ols)

    # prepare a list of values for the prediction model
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 50)
    # prediction model formula
    get_y = lambda a, b, c: a + b * x + c * np.power(x, 2)

    # plot the OLS fit line
    y = get_y(ols['a'], ols['b'], ols['c'])
    ax.plot(x, y, color='darkorange', linewidth=1, label='OLS')

    # plot each of the quantiles in the models dataframe
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i], models.c[i])
        ax.plot(x, y, linestyle='dotted', color='red', label=f'{models.q[i] * 100:.0f}th centile')

    ax.legend(loc='lower right') if par_name == 'sats' else ax.legend(loc='upper right')
    format_plot(par_name, chart_type, ax)
    return parameter_df


def poly_quantile_regression_2(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    chart_type = 'Polynomial quantile regression for y = m + x + x^2 + x^3 '

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age + np.power(age, 2) + np.power(age, 3)', parameter_df)
    result = model.fit(q=.5)
    print('\n')
    print(chart_type)
    print('\n')
    print(result.summary())
    print('\n')
    print(result.params)

    # quantile lines to display
    quantiles = [.95, .75, .5, .25, .05]

    def fit_model(q):
        # function to apply LAD model to the data
        result = model.fit(q=q)
        return [
                   q,
                   result.params['Intercept'],
                   result.params['age'],
                   result.params['np.power(age, 2)'],
                   result.params['np.power(age, 3)'],
               ] + result.conf_int().loc['age'].tolist()

    # apply ALD model for each quantile in list & convert to dataframe for plotting
    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b', 'c', 'd', 'lb', 'ub'])

    ols = smf.ols(f'{par_name} ~ age + np.power(age, 2) + np.power(age, 3)', parameter_df).fit()
    ols_ci = ols.conf_int().loc['age'].tolist()
    ols = dict(a=ols.params['Intercept'],
               b=ols.params['age'],
               c=ols.params['np.power(age, 2)'],
               d=ols.params['np.power(age, 3)'],
               lb=ols_ci[0],
               ub=ols_ci[1])

    print('\n')
    print(models)
    print('\n')
    print(ols)

    # prepare a list of values for the prediction model
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 50)
    # prediction model formula
    get_y = lambda a, b, c, d: a + b * x + c * np.power(x, 2) + d * np.power(x,3)

    # plot the OLS fit line
    y = get_y(ols['a'], ols['b'], ols['c'], ols['d'])
    ax.plot(x, y, color='darkorange', linewidth=1, label='OLS')

    # plot each of the quantiles in the models dataframe
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i], models.c[i], models.d[i])
        ax.plot(x, y, linestyle='dotted', color='red', label=f'{models.q[i] * 100:.0f}th centile')

    ax.legend(loc='lower right') if par_name == 'sats' else ax.legend(loc='upper right')
    format_plot(par_name, chart_type, ax)
    return parameter_df


def poly_quantile_regression_3(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    chart_type = 'Polynomial quantile regression for y = m + x + x^0.5 + x^2 + x^3 '

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age + np.power(age, 0.5) + np.power(age, 2) + np.power(age, 3)', parameter_df)
    result = model.fit(q=.5)
    print('\n')
    print(chart_type)
    print('\n')
    print(result.summary())
    print('\n')
    print(result.params)

    # quantile lines to display
    quantiles = [.95, .75, .5, .25, .05]

    def fit_model(q):
        # function to apply LAD model to the data
        result = model.fit(q=q)
        return [
                   q,
                   result.params['Intercept'],
                   result.params['age'],
                   result.params['np.power(age, 0.5)'],
                   result.params['np.power(age, 2)'],
                   result.params['np.power(age, 3)'],
               ] + result.conf_int().loc['age'].tolist()

    # apply ALD model for each quantile in list & convert to dataframe for plotting
    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b', 'c', 'd', 'e', 'lb', 'ub'])

    ols = smf.ols(f'{par_name} ~ age + + np.power(age, 0.5) + np.power(age, 2) + np.power(age, 3)', parameter_df).fit()
    ols_ci = ols.conf_int().loc['age'].tolist()
    ols = dict(a=ols.params['Intercept'],
               b=ols.params['age'],
               c=ols.params['np.power(age, 0.5)'],
               d=ols.params['np.power(age, 2)'],
               e=ols.params['np.power(age, 3)'],
               lb=ols_ci[0],
               ub=ols_ci[1])

    print('\n')
    print(models)
    print('\n')
    print(ols)

    # prepare a list of values for the prediction model
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 50)
    # prediction model formula
    get_y = lambda a, b, c, d, e: a + b * x + c * np.power(x, 0.5) + d * np.power(x, 2) + e * np.power(x, 3)

    # plot the OLS fit line
    y = get_y(ols['a'], ols['b'], ols['c'], ols['d'], ols['e'])
    ax.plot(x, y, color='darkorange', linewidth=1, label='OLS')

    # plot each of the quantiles in the models dataframe
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i], models.c[i], models.d[i], models.e[i])
        ax.plot(x, y, linestyle='dotted', color='red', label=f'{models.q[i] * 100:.0f}th centile')

    ax.legend(loc='lower right') if par_name == 'sats' else ax.legend(loc='upper right')
    format_plot(par_name, chart_type, ax)
    return parameter_df


""" Save files """


def save_as_csv(parameter_df):
    # saves the parameter dataframe as a csv file for quick analysis later
    par_name = parameter_df.columns[1]
    parameter_df.to_csv(f'data/{par_name}.csv')
    return parameter_df


""" Sequential Function Call """

# use this for analysing files on Sharepoint
# TODO change parameter abbreviations to full descriptions
parameter_list = ['HR', 'RR', 'BP', 'sats']
for parameter in parameter_list:
    # takes the dataframe and processes in sequence
    df = load_sharepoint_file(file_scope='full')
    process = (

        select_parameter(df, parameter)
            .pipe(split_BP)
            .pipe(clean_data)
            .pipe(convert_decimal_age)
            .pipe(print_data)
            .pipe(poly_quantile_regression_1)
            .pipe(poly_quantile_regression_2)
            .pipe(poly_quantile_regression_3)
            .pipe(save_as_csv)
    )

# .pipe(plot_scatter)

# use this for testing - takes in pre-prepared data set
# parameter_list = ['HR', 'RR', 'sBP', 'sats']
# for parameter in parameter_list:
#     # takes the dataframe and processes in sequence
#     parameter_df = load_saved_data(parameter)
#     print_data(parameter_df)
#     poly_quantile_regression(parameter_df)



# use this for analysing synthetic data set
# parameter_list = ['HR', 'RR', 'BP']
# for parameter in parameter_list:
#     # takes the dataframe and processes in sequence
#     df = load_synthetic_data()
#     process = (
#
#         select_parameter(df, parameter)
#             .pipe(split_BP)
#             .pipe(clean_data)
#             .pipe(convert_decimal_age)
#             .pipe(print_data)
#             .pipe(plot_scatter)
#             .pipe(poly_quantile_regression)
#             .pipe(save_as_csv)
#             )

exit()

# Quick access for .pipe functions (copy & paste these into process above)

# .pipe(convert_decimal_age)
# .pipe(print_data)
# .pipe(bin_by_age)

# .pipe(linear_regression)
# .pipe(polynomial_regression)
# .pipe(quantile_regression)
# .pipe(poly_quantile_regression)


""" Test Code """
# the following is test code used for development and not used in analysis


""" check the modeling assumptions of normality and homoscedasticity of the residuals """
# fitted_values = model.predict(parameter_df)
# residuals = parameter_df[par_name] - fitted_values

# Check normality of residuals
# plt.hist(residuals)
# plt.title('Model : Histogram of Residuals', fontsize=12, weight='bold')
# plt.show()

# Check variance of residuals
# plt.scatter(fitted_values, residuals)
# plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
# plt.title('Model : Residuals vs Fitted Values', fontsize=12, weight='bold')
# plt.show()


# bin the HR data by age
# bins = 18  # 1 month = 216, 2 months = 108, 4 months = 54, 6 months = 36
# [np.arange(0, 6574, 28)]  # Age bins according to PEWS chart categories
# bin_labels = [np.arange(0, len(bins))]  # Age bin category labels

# print('\nbins\n')
# print(bins)
# print(bin_labels)

# classify age according to age bins and add an Age bin column to the PEWS Dataframe
# df_HR['bin'] = pd.cut(df_HR.age, bins=bins)
# print('\nHR Dataframe 2\n')
# print(df_HR)

# calculate the mean values for each age bin
# HR_mean = df_HR.groupby('bin', as_index=False, sort=True).mean()
# HR_mean = pd.DataFrame(HR_mean, columns=['age', 'bin', 'HR'])
# print('\nHR means\n')
# print(HR_mean)

# TODO figure out how to plot the standard deviation

# HR_std = df_HR.groupby('bin', as_index=False, sort=True).std()
# HR_std = pd.DataFrame(HR_std)  # , columns=['age', 'bin', 'HR']
# print('\nHR standard deviations\n')
# print(HR_std)

# TODO match up centiles with the age markers better


exit()

"""Saving this for later"""


# def bin_by_age(parameter_df, bins=54):
#     # categorises the parameter data by time intervals
#     # number of bins for specific time intervals: 1 month = 216, 2 months = 108, 4 months = 54, 6 months = 36
#     par_name = parameter_df.columns[1]
#     parameter_df['bin'] = pd.cut(parameter_df.age, bins=bins)
#     print(f'\n...{par_name} data binned by age intervals...')
#     print(parameter_df.head(10))
#     return parameter_df

