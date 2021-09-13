"""
    NAME:          PEWSDataAnalysis/PEWS_centiles.py
    AUTHOR:        Jeremy Tong
    EMAIL:         jeremy.tong.17@ucl.ac.uk
    DATE:          02/01/2021
    VERSION:       0.1
    INSTITUTION:   University College London & University of Manchester
    DESCRIPTION:   Python file for analysing PEWS Data for MSc Dissertation
    DETAILS:       Functions sued to process data and polt the data on charts with regression analysis
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

# Import the PEWS models for calculating scores
import PEWS_models as pm


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
    print(df.columns)
    # print(df.head(10))
    print('\n')
    return(df)


""" Data Processing """


def select_parameter(df, parameter):
    # creates a new dataframe with the single parameter from the PEWS dataframe
    # renames the columns as age and parameter name
    # counts the number of rows and non-null values
    parameter_df = df[['age_in_days', parameter]].values
    parameter_df = pd.DataFrame(parameter_df, columns=['age_in_days', parameter])
    count_1 = len(parameter_df)
    count_2 = parameter_df['age_in_days'].count()
    count_3 = parameter_df[parameter].count()
    print('\n')
    print('=' * 80)
    print(f'\n...{parameter} DataFrame created...')
    print(f'\n...{count_1} rows in total... ')
    print(f'\n...{count_2} age_in_days values found with {count_1 - count_2} missing...')
    print(f'\n...{count_3} {parameter} values found with {count_1 - count_3} missing...')
    return parameter_df


def split_BP(parameter_df):
    # if the parameter is BP, splits the BP data into systolic BP and diastolic BP columns.
    # otherwise does not alter the data
    # do this before clean_data() or values will be lost
    # counts the number of sBP values available
    if 'BP' in parameter_df:
        BP = parameter_df['BP'].str.split('/', n=1, expand=True)
        parameter_df['sBP'] = BP[0]
        # parameter_df['dBP'] = BP[1]
        parameter_df.drop(columns=['BP'], inplace=True)
        count_1 = len(parameter_df)
        count_2 = parameter_df['sBP'].count()
        print(f'\n...Extracting sBP from BP column...')
        print(f'\n...{count_1} BP rows in total...')
        print(f'\n...{count_2} sBP values extracted...')
        return parameter_df
    else:
        return parameter_df


def clean_data(parameter_df):
    # takes the parameter dataframe and converts text to np.NaN values
    # removes missing values
    # converts data types from objects to integers
    # counts the number of rows and number of missing values
    par_name = parameter_df.columns[1]
    count_1 = parameter_df[par_name].count()
    parameter_df[par_name] = parameter_df[par_name].replace(r'\D+', np.NaN, regex=True)
    count_2 = parameter_df[par_name].count()
    parameter_df.dropna(inplace=True)
    count_3 = parameter_df[par_name].count()

    for i in list(parameter_df):
        parameter_df[i] = parameter_df[i].astype(int)
    print('\n')
    print('=' * 80)
    print(f'\n...{count_1} non-null {parameter} values...')
    print(f'\n...{count_1 - count_2} text values in {i} replaced with np.NaN...')
    print(f'\n...{count_1 - count_3} np.NaN values in {i} removed...')
    print(f'\n...{count_3} {i} values in final count...')
    print(f'\n...{par_name} Data Cleaning Complete...')

    return parameter_df


def convert_decimal_age(parameter_df):
    # converts the age in days to a decimal age
    parameter_df['age'] = parameter_df['age_in_days'] / 365.25
    parameter_df.drop(['age_in_days'], axis=1)
    print('\n...Converted age in days to decimal age...')
    # print(parameter_df.head(10))
    return parameter_df


def bin_age_chart(parameter_df):
    # bin ages by chart age range
    bins = [0, 1, 5, 12, 18]  # Age bins according to PEWS chart categories
    charts = [0, 1, 2, 3]  # age ranges are 0: 0-11m, 1: 1-4y, 2: 5-11y, 3: >12y

    # add a chart column to and classify age according to PEWS model age ranges
    print('\n...Charts labelled based on age...')
    parameter_df['chart'] = pd.cut(parameter_df.age, bins=bins, labels=charts)
    # print(parameter_df.columns())
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
    print('\n')
    print('=' * 80)
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


def full_names(par_name):
    # function to return the full name of a parameter
    if par_name == 'HR':
        return 'Heart Rate (beats per minute)'
    elif par_name == 'RR':
        return 'Respiratory Rate (breaths per minute)'
    elif par_name == 'sats':
        return 'Oxygen Saturation (%)'
    elif par_name == 'sBP':
        return 'Systolic Blood Pressure (mmHg)'
    elif par_name == 'temp':
        return 'Temperature (degrees Celcius)'
    else:
        return 'error par_name not in list'


def format_plot(par_name, plot_type):
    # function to add chart title, axis labels, show plot and save plot as .png
    label_1 = full_names(par_name)
    plt.xlabel('Age (years)', fontsize=14)
    plt.ylabel(f'{label_1}', fontsize=14)
    # plt.title(f'{label_1} in children', fontsize=20)
    plt.savefig(f'plots/{par_name}_{plot_type}_plot.png')
    plt.show()
    plt.clf()


def plot_age_distribution(df):
    # function to plot a the age distribution
    # first convert age in days to decimal age
    # then delete duplicate s numbers and admission spells
    convert_decimal_age(df)
    df= df.drop_duplicates(subset=['spell_id'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df, x='age', kde=False, bins=216, element='step', color='lightblue')
    ax.set_xticks(list(range(19)))
    plt.title('Age distribution', fontsize=20)
    plt.xlabel('Age (years)', fontsize=14)
    plt.ylabel('Number of children admitted', fontsize=14)
    plt.savefig('plots/age_dist_plot.png')
    plt.show()
    plt.clf()


def plot_scatter_1(parameter_df):
    # function to plot a scatter plot of the parameter data
    par_name = parameter_df.columns[1]
    plot_type = 'Scatter_plot'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color='deepskyblue', alpha=0.1)
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


""" Composite Scatter plot with PEWS thresholds """


def calculate_UPEWS_score(parameter_df):
    # function to calculate the UHL PEWS score for the parameter

    model = pm.UPEWS_model # calculate scores for this model
    model_name = 'UHL PEWS'
    par_name = parameter_df.columns[1]  # name of the parameter being plotted

    def score(chart, par_name, value):
        # function for retruning the score based on the value of the parameter
        # chart = the PEWS chart corresponding to the age range
        # par = vital sign or observation parameter
        # value = value of the parameter

        # create a mini-DataFrame of parameter limits based on the age (chart) and parameter to be scored
        limits = model.loc[
            (model['chart'] == chart) & (model['par'] == par_name), ['lower', 'upper', 'score']]
        for index, row in limits.iterrows():
            lower = row.lower
            upper = row.upper + 1

            # return the score if the parameter value is within the range of the limits
            if value in range(lower, upper):
                score = row.score
                return score

    # add a chart column and calculate the PEWS score
    print(f'\n...working out the UHL PEWS scores for {par_name}, please wait...')
    parameter_df['PEWS score'] = parameter_df.apply(lambda row: score(row['chart'], par_name, row[par_name]), axis=1)
    print(f'\n...{par_name} scoring complete...')
    print(f'\nDisplay {par_name} score stats for {model_name} model:\n')
    print(parameter_df.groupby('chart')['PEWS score'].value_counts())

    return parameter_df


def calculate_NPEWS_score(parameter_df):
    # function to calculate the National PEWS score for the parameter (duplicate of UPEWS scoring function)

    model = pm.NPEWS_model # calculate scores for this model
    model_name = 'National PEWS'
    par_name = parameter_df.columns[1]  # name of the parameter being plotted

    def score(chart, par_name, value):
        # function for retruning the score based on the value of the parameter
        # chart = the PEWS chart corresponding to the age range
        # par = vital sign or observation parameter
        # value = value of the parameter

        # create a mini-DataFrame of parameter limits based on the age (chart) and parameter to be scored
        limits = model.loc[
            (model['chart'] == chart) & (model['par'] == par_name), ['lower', 'upper', 'score']]
        for index, row in limits.iterrows():
            lower = row.lower
            upper = row.upper + 1

            # return the score if the parameter value is within the range of the limits
            if value in range(lower, upper):
                score = row.score
                return score

    # add a chart column and calculate the PEWS score
    print(f'\n...working out the National PEWS scores for {par_name}, please wait...')
    parameter_df['PEWS score'] = parameter_df.apply(lambda row: score(row['chart'], par_name, row[par_name]), axis=1)
    print(f'\n...{par_name} scoring complete...')

    # display the counts for each score for each PEWS chart (age bin)
    print(f'\nDisplay {par_name} score stats for {model_name} model:\n')
    print(parameter_df.groupby('chart')['PEWS score'].value_counts())

    return parameter_df


def plot_scatter_2(parameter_df):
    # function to plot a scatter plot of the parameter data

    par_name = parameter_df.columns[1] # name of the parameter being plotted

    # plot the scatter data
    print(f'\n...plotting the UHL PEWS Scatter array for {par_name}...')

    plot_type = 'Scatter_UPEWS_overlay'

    color_dict = dict({0: 'black', 1: 'red'})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(data=parameter_df, x='age', y=par_name, marker='.', hue='PEWS score', palette=color_dict, alpha=0.1)
    ax.set_xticks(list(range(19)))

    ax.legend(title='PEWS Score', frameon=False, fontsize=10, loc='lower right') if par_name == 'sats' \
        else ax.legend(title='PEWS Score', frameon=False, fontsize=10, loc='upper right')

    format_plot(par_name, plot_type)
    return parameter_df


def plot_scatter_3(parameter_df):
    # function to plot a scatter plot of the parameter data

    par_name = parameter_df.columns[1] # name of the parameter being plotted

    # plot the scatter data
    print(f'\n...plotting the NPEWS Scatter array for {par_name}...')

    plot_type = 'Scatter_NPEWS_overlay'

    color_dict = dict({0: 'black', 1: 'gold', 2: 'darkorange', 4: 'red'})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(data=parameter_df, x='age', y=par_name, marker='.', hue='PEWS score', palette=color_dict, alpha=0.1)
    ax.set_xticks(list(range(19)))

    ax.legend(title='PEWS Score', frameon=False, fontsize=10, loc='lower right') if par_name == 'sats' \
        else ax.legend(title='PEWS Score', frameon=False, fontsize=10, loc='upper right')

    format_plot(par_name, plot_type)
    return parameter_df


""" Data plots with regression analysis """


def linear_regression(parameter_df):
    # function to plot a linear regression of the parameter
    par_name = parameter_df.columns[1]
    plot_type = 'regression'

    # fit the median values to a simple linear regression model
    model = sm.OLS.from_formula(f'{par_name} ~ age', data=parameter_df).fit()
    print('\n')
    print('=' * 80)
    print(model.params, '\n', model.summary().extra_txt)

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color=color_selector(par_name), alpha=0.1)

    # plot the regression line overlaid on parameter scatter plot
    ax.plot(parameter_df.age, model.params[0] + model.params[1] * parameter_df.age, color='orange', linewidth=1)

    # format the chart and save as .png
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


def polynomial_regression(parameter_df):
    # function to plot a linear regression of the parameter
    par_name = parameter_df.columns[1]
    plot_type = 'polynomial'

    # fit the median values to a simple linear regression model
    model = sm.OLS.from_formula(f'{par_name} ~ age + np.power(age,2)', data=parameter_df).fit()
    print('\n')
    print('=' * 80)
    print('\n', model.params, '\n', model.summary().extra_txt)

    """ check the modeling assumptions of normality and homoscedasticity of the residuals """

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color='lightgrey', alpha=0.1)

    # plot the regression line overlaid on parameter scatter plot
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 50)
    y = model.params[0] + model.params[1] * x + model.params[2] * np.power(x, 2)

    ax.plot(x, y, linestyle='-', color='limegreen', linewidth=1)

    # format the chart and save as .png
    # plt.title(f'Polynomial regression for {par_name}', fontsize=20)
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


def quantile_regression(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    plot_type = 'OLS_regression'

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color='lightgrey', alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age', parameter_df)
    result = model.fit(q=0.5)
    print('\n')
    print('=' * 80)
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
        ax.plot(x, y, linestyle='-', color='royalblue')

    y = get_y(ols['a'], ols['b'])

    ax.plot(x, y, color='limegreen', label='OLS')

    ax.legend()
    plt.title(f'OLS regression for {par_name}', fontsize=20)
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


def poly_quantile_regression_1(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    plot_type = 'Polynomial_quantile_regression_1'

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color='lightgrey', alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age + np.power(age, 2)', parameter_df)
    result = model.fit(q=.5)
    print('\n')
    print('=' * 80)
    print('\n')
    print(plot_type)
    print('\n')
    print(result.summary())
    print('\n')
    print(result.params)

    # quantile lines to display
    # quantiles = [.95, .75, .5, .25, .05]
    quantiles = [.99, .9, .75, .5, .25, .1, .01]

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
    # ax.plot(x, y, linestyle='dotted', color='limegreen', label='OLS')

    # plot each of the quantiles in the models dataframe
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i], models.c[i])
        ax.plot(
            x, y,
            linestyle='-',
            linewidth=1,
            color='limegreen' if models.q[i] == 0.5 else 'deepskyblue',
            label=f'{models.q[i] * 100:.0f}st   percentile' if models.q[
                                                                   i] == 0.01 else f'{models.q[i] * 100:.0f}th percentile'
        )

    # lable the quantile lines to the right of each line
    for line, name in zip(ax.lines, quantiles):
        y = line.get_ydata()[-1]
        ax.annotate(
            f'{name * 100:.0f}',
            xy=(1, y),
            xytext=(-30, 0),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            color='limegreen' if name == 0.5 else 'deepskyblue',
            size=8, va="center",
        )

    # set the legend
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    # plt.title(f'Polynomial quantile regression for {par_name} (y = m + x + x^2)', fontsize=18, y=1.05)
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


def poly_quantile_regression_2(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    plot_type = 'Polynomial_quantile_regression_2'

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color='lightgrey', alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age + np.power(age, 2) + np.power(age, 3)', parameter_df)
    result = model.fit(q=.5)
    print('\n')
    print('=' * 80)
    print('\n')
    print(plot_type)
    print('\n')
    print(result.summary())
    print('\n')
    print(result.params)

    # quantile lines to display
    # quantiles = [.95, .75, .5, .25, .05]
    quantiles = [.99, .9, .75, .5, .25, .1, .01]

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
    # ax.plot(x, y, linestyle='dotted', color='limegreen', label='OLS')

    # plot each of the quantiles in the models dataframe
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i], models.c[i], models.d[i])
        ax.plot(
            x, y,
            linestyle='-',
            linewidth=1,
            color='limegreen' if models.q[i] == 0.5 else 'deepskyblue',
            label=f'{models.q[i]*100:.0f}st   percentile' if models.q[i] == 0.01 else f'{models.q[i]*100:.0f}th percentile'
        )

    # lable the quantile lines to the right of each line
    for line, name in zip(ax.lines, quantiles):
        y = line.get_ydata()[-1]
        ax.annotate(
            f'{name*100:.0f}',
            xy=(1, y),
            xytext=(-30, 0),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            color='limegreen' if name == 0.5 else 'deepskyblue',
            size=8, va="center",
        )

    # set the legend
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    # plt.title(f'Polynomial quantile regression for {par_name} (y = m + x + x^2 + x^3)', fontsize=18, y=1.05)
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


def poly_quantile_regression_3(parameter_df):
    # function to plot centile lines using quantile regression

    par_name = parameter_df.columns[1]
    plot_type = 'Polynomial_quantile_regression_3'

    # plot a scatter graph of the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y=par_name, data=parameter_df, marker='.', color='lightgrey', alpha=0.1)

    # set up Least Absolute Deviation model (quantile regression where q = 0.5) and print results
    model = smf.quantreg(f'{par_name} ~ age + np.power(age, 0.5) + np.power(age, 2) + np.power(age, 3)', parameter_df)
    result = model.fit(q=.5)
    print('\n')
    print('=' * 80)
    print('\n')
    print(plot_type)
    print('\n')
    print(result.summary())
    print('\n')
    print(result.params)

    # quantile lines to display
    quantiles = [.99, .9, .75, .5, .25, .1, .01]

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

    # apply LAD model for each quantile in list & convert to dataframe for plotting
    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b', 'c', 'd', 'e', 'lb', 'ub'])

    ols = smf.ols(f'{par_name} ~ age + np.power(age, 0.5) + np.power(age, 2) + np.power(age, 3)', parameter_df).fit()
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
    x = np.linspace(parameter_df.age.min(), parameter_df.age.max(), 216)
    # prediction model formula
    get_y = lambda a, b, c, d, e: a + b * x + c * np.power(x, 0.5) + d * np.power(x, 2) + e * np.power(x, 3)

    # plot the OLS fit line
    y = get_y(ols['a'], ols['b'], ols['c'], ols['d'], ols['e'])
    # ax.plot(x, y, linestyle='dotted', color='limegreen', label='OLS')

    # plot each of the quantiles in the models dataframe
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i], models.c[i], models.d[i], models.e[i])
        ax.plot(
            x, y,
            linestyle='-',
            linewidth=1,
            color='limegreen' if models.q[i] == 0.5 else 'deepskyblue',
            label=f'{models.q[i] * 100:.0f}st   percentile' if models.q[
                                                                   i] == 0.01 else f'{models.q[i] * 100:.0f}th percentile'
        )

    # lable the quantile lines to the right of each line
    for line, name in zip(ax.lines, quantiles):
        y = line.get_ydata()[-1]
        ax.annotate(
            f'{name * 100:.0f}',
            xy=(1, y),
            xytext=(-30, 0),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            color='limegreen' if name == 0.5 else 'deepskyblue',
            size=8, va="center",
        )

    # set the legend
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    # plt.title(f'Polynomial quantile regression for {par_name} (y = m + x + x^0.5 + x^2 + x^3)', fontsize=18, y=1.05)
    ax.set_xticks(list(range(19)))
    format_plot(par_name, plot_type)
    return parameter_df


""" Save files """


def save_as_csv(parameter_df):
    # saves the parameter dataframe as a csv file for quick analysis later
    par_name = parameter_df.columns[1]
    parameter_df.to_csv(f'data/{par_name}.csv')
    return parameter_df


""" Demographics """
#  Use this section to print out demographics and data cleaning summaries

# def demographics(df):
#
#     def unique(var):
#         return df[var].nunique()
#
#     def summary(var):
#         return df[var].dropna().describe()
#
#     def missing(var):
#         count_1 = len(df)
#         count_2 = df[var].count()
#         return count_1 - count_2
#
#
#     # set pandas options to display all columns in a DataFrame
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#
#     admission = df['spell_id'].str.split('_', n=1, expand=True)
#     df['admission'] = admission[0]
#
#     print('\n')
#     print('=' * 80)
#     print('\nSummary Demographics:')
#     print(f'\n...Number of children: {unique("s_number_x")}...')
#     print(f'\n...Number of admission episodes: {unique("admission")}...')
#     print(f'\n\nSummary Statistics for LoS:')
#     print(f'\n...{len(df)} values available for LoS with {missing("los_hours")} missing\n')
#     print(summary('los_hours'))
#     print(f'\n\nSummary Statistics for UHL PEWS:')
#     print(f'\n...{len(df)} values available for UHL PEWS with {missing("EWS")} missing\n')
#     print(summary('EWS'))
#     print('\nBreakdown of UHL PEWS:')
#     return (df)
#
# # load the data
# df = load_sharepoint_file(file_scope='full')
# # run demographics function
# demographics(df)
#
# parameter_list = ['sats', 'RR', 'HR', 'BP']
# for parameter in parameter_list:
#     # takes the dataframe and processes in sequence
#     process = (
#         select_parameter(df, parameter)
#             .pipe(split_BP)
#             .pipe(clean_data)
#             .pipe(convert_decimal_age)
#             .pipe(print_data)
#     )
#
# exit()

""" Sequential Function Calls """


""" Age Distribution Plot """

# plot an age distribution for the data set
# df = load_sharepoint_file(file_scope='full')
# explore_data(df)
# plot_age_distribution(df)
# explore_data(df)
# exit()


""" Scatter Plots """
# Use this section for plotting scatter graphs

# load the data
df = load_sharepoint_file(file_scope='full')
parameter_list = ['sats', 'RR', 'HR', 'BP']

for parameter in parameter_list:

    # takes the dataframe and processes in sequence for National PEWS
    process = (
        select_parameter(df, parameter)
            .pipe(split_BP)
            .pipe(clean_data)
            .pipe(convert_decimal_age)
            .pipe(bin_age_chart)
            .pipe(calculate_NPEWS_score)
            .pipe(print_data)
            .pipe(plot_scatter_3)
    )

    # takes the dataframe and processes in sequence for UHL PEWS
    process = (
        select_parameter(df, parameter)
            .pipe(split_BP)
            .pipe(clean_data)
            .pipe(convert_decimal_age)
            .pipe(bin_age_chart)
            .pipe(calculate_UPEWS_score)
            .pipe(print_data)
            .pipe(plot_scatter_2)
    )

exit()


""" Quantile Regression PLots """
# use this for plotting quantile regression

# load the data
df = load_sharepoint_file(file_scope='full')
parameter_list = ['RR', 'HR', 'BP']

for parameter in parameter_list:
    # takes the dataframe and processes in sequence

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
exit()

# #
#
""" Code Testing """

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
#             .pipe(plot_scatter_1)
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

# TODO figure out how to plot the standard deviation on centile chart


exit()


