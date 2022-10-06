import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas.io.sql as sqlio
import psycopg2
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tools.eval_measures as ev
from sklearn.model_selection import train_test_split
import scipy.stats as stats


def main():
    data = load_data_from_postgres()
    data = clean_data(data)
    data = one_hot_encoding(data)
    # eda_categorical_variables(data)
    # eda_continuous_variables(data)
    
    correlations(data)
    variables = ['marketing_spend',
          'Promotion Blue', 'Promotion Red',
        #   'week_id', 'month_number', 'month_id', 'year',
        #   'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
        #   'visitors'
          ]
    X = data[variables]
    y = data['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    model, rsq_adj, rmse, mae = calc_sm_ols(model_nbr=1, X=X_train, y=y_train)
    error_on_test_set(model_nbr=1, model=model, X=X_test, y=y_test)
    qqplot(model_nbr=8, model=model)

def load_data_from_postgres():
    with psycopg2.connect(
       database="Endeavor_SQL_tasks", user='Tom', password=os.environ['endeavor_pass'], host='127.0.0.1', port= '5432') as conn:

        # Get data from views
        sql = "select * from joined_tbl;"
        dat = sqlio.read_sql_query(sql, conn, parse_dates=['date'])
    return dat


def clean_data(dat):
    """

    Args:
        dat (_type_): _description_

    Returns:
        _type_: _description_
    """
    # need to fill the missing values in the marketing_spend, promo, revenue and visitors
    # Baseline: remove with missing data -> run on a training set and measure 
    dat = missing_values(dat, option=1)
    dat['promo'] = dat['promo'].str.strip()
    dat['day_name'] = dat['day_name'].str.strip()
    dat['week_id'] = dat['week_id'] - dat['week_id'].min()
    dat['month_id'] = dat['month_id'] - dat['month_id'].min()
    dat['year'] = dat['year'] - dat['year'].min()
    return dat

def missing_values(dat, option=1):
    """Deals with missing values

    Args:
        dat (pd.DataFrame): 
        option (int): option to use to deal with missing values
    
    Options:
    1: drop missing values
    """
    if option == 1:
        dat.dropna(inplace=True)
    return dat

def one_hot_encoding(dat):
    # One-hot encoding
    promo_codes = pd.get_dummies(dat['promo'])
    day_name_codes = pd.get_dummies(dat['day_name'])
    dat = pd.concat([dat, promo_codes, day_name_codes], axis=1)
    return dat
    
def eda_continuous_variables(dat):
    """EDA for continous variables (plots them on a pairplot)
    
    Comments on figure: 
    There is not relationship between week_id and month_number with revenue - double check with pearsons
    Linear relationship of marketing_spend and visitors to revenue
    Args:
        dat (pd.DataFrame): data
    """
    sns.pairplot(dat[['revenue','marketing_spend','visitors','promo', 'week_id','month_number']], hue="promo")

    plt.show()
    plt.close()


def eda_categorical_variables(dat):
    """EDA for cat variables (plots them on a pairplot)
    
    Comments on figure: 
    Some difference in the variance between days
    Promotion blue looks better than red which looks better than no promo.
    No apparent correlation between which promotion did well on which days.
    Args:
        dat (pd.DataFrame): data
    """
    fig, ax  = plt.subplots(1,1)
    sns.swarmplot(data=dat, x="day_name", y="revenue", hue='promo',
                  linewidth=1)
    ax.yaxis.set_major_formatter('£{x:1.0f}')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    fig, ax  = plt.subplots(1,1)
    sns.violinplot(data=dat, x="promo", y="revenue",
                  linewidth=1)
    ax.yaxis.set_major_formatter('£{x:1.0f}')
    plt.tight_layout()
    
    plt.show()
    plt.close()
    
def correlations(data):
    """Look at correlation matrix to see what correlates with the revenue
    And check for multicollinearity

    Args:
        data (_type_): _description_
    """
    # generate full correlation matrix
    corr = data.corr()
    print(corr['revenue'].sort_values( ascending=False))

    
    # check for collinearity among potential predictors by pairs
    df = (
         corr
         .stack()
         .reset_index()
         .sort_values(0, ascending=False)
    )
    
    # zip the default name columns level_0 and level_1
    df['pairs'] = list(zip(df.level_0, df.level_1))
    # set index to pairs
    df.set_index(['pairs'], inplace=True)
    # now drop level columns
    df.drop(columns=['level_1', 'level_0'], inplace=True)
    # rename correlation column rather than 0
    df.columns = ['correlation']
    # drop duplicates and keep only pair correlations above 0.65
    df.drop_duplicates(inplace=True)
    df = df[abs(df.correlation) > .65]
    print(df)
    


# function takes X and y dataframes and generates statsmodel OLS results
def calc_sm_ols(model_nbr, X, y):
    """Runs and returns statsmodel Ordinary Least Squares (OLS) regression model.
       Takes in X predictors and y target, generates model, predictions and 
       performance stats.
    
        Parameters:
        model_nbr (int): sequence number you've created for model iteration
        X (pd.DataFrame): train or test slice contains predictors
        y (pd.DataFrame)): train or test slice contains target
        Returns:
        model (linear_model): statsmodel fitted OLS model object
        rsq_adj (float): model adjusted r-squared
        rmse (float): model root mean squared error
        mae (float): models mean absolute error
   """
    
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # generate model predictions and calculate errors 
    y_pred = model.predict(X)
    if y.name == 'revenue':
        rmse = round(ev.rmse(y, y_pred))
        mae = round(ev.meanabs(y, y_pred))
        print(f'\nModel {model_nbr} Summary Statistics')
        print(f'Root Mean Squared Error (RMSE): {rmse}.')
        print(f'Mean Absolute Error (MAE): {mae}.')
    else :
        print(f'Cannot calculate RMSE and MAE from y variable {y}')
    rsq_adj = model.rsquared_adj                 
    return model, rsq_adj, rmse, mae

def error_on_test_set(model_nbr, model, X, y):
    y_pred = model.predict(X)
    if y.name == 'revenue':
        rmse = round(ev.rmse(y, y_pred))
        mae = round(ev.meanabs(y, y_pred))
        print(f'\nModel {model_nbr} Summary Statistics')
        print(f'Root Mean Squared Error (RMSE): {rmse}.')
        print(f'Mean Absolute Error (MAE): {mae}.')
    else :
        print(f'Cannot calculate RMSE and MAE from y variable {y}')

# function to generate QQ-plot for an OLS model
def qqplot(model_nbr, model):
    """Displays a QQ-plot
    
        Parameters:
        model_nbr (int): sequence number you've created for model iteration
        model (linear_model): statsmodel fitted OLS model object
        Returns: no return
   """
        
    # generate QQ-plot
    residuals = model.resid
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    plt.title(f'\nModel {model_nbr} QQ-plot' )
    plt.tight_layout()
    plt.show()
    plt.close()
    


if __name__ == '__main__':
    main()
