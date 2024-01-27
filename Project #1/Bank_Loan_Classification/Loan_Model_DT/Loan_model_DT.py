# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:55:41 2022

@author: SAghigh
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


df = pd.read_excel(r'C:\Users\saghigh\Desktop\Coding_Visu\Python\Ga ML' \
                   r'\Assignment #1\Loan_Model_DT/Bank_Personal_Loan_Modelling.xlsx', sheet_name='Data')

# EDA (Explanatory data Analysis) Section    
 
def cat_age(x):
    if x <= 35:
        return 'Young'
    if x >35 and x <= 55:
        return 'Middle Age'
    if x > 55:
        return 'Elderly'

def cat_exp(x):
    if x <= 10:
        return 'Low EXP'
    if x > 10 and x <= 30:
        return 'Mid EXP'
    if x > 30:
        return 'High EXP'

def cat_income(x):
    if x <= 39:
        return 'Low Income'
    if x > 39 and x <= 98:
        return 'Mid Income'
    if x > 98:
        return 'High Income'

def cat_CCavg(x):
    if x <= 0.7:
        return 'Low Spending'
    if x > 0.7 and x <= 2.5:
        return 'Mid Spending'
    if x > 2.5:
        return 'High Spending'

def cat_homeloan(x):
    if x == 0:
        return 'No Mortgage'
    if x > 0:
        return 'With Mortgage'
    

# NULL Counts

df.info(verbose=True, null_counts=True)

# Dataframe Stats

df_describe = df.describe()

# Finding the percentage of missing values in all columns 

df_missing = round(df.isnull().mean(), 2).sort_values(ascending=False)

#Converting Int columns to categorical variables based on hostograms and boxplots
func_age = np.vectorize(cat_age)
age_cat = func_age(df['Age'])
df['age_cat'] = age_cat

func_exp = np.vectorize(cat_exp)
exp_cat = func_exp(df['Experience'])
df['exp_cat'] = exp_cat

func_income = np.vectorize(cat_income)
income_cat = func_income(df['Income'])
df['income_cat'] = income_cat

func_ccavg = np.vectorize(cat_CCavg)
ccavg_cat = func_ccavg(df['CCAvg'])
df['ccavg_cat'] = ccavg_cat

func_homeloan = np.vectorize(cat_homeloan)
homeloan_cat = func_homeloan(df['Personal Loan'])
df['homeloan_cat'] = homeloan_cat

### Data Visuallization

for col in df.columns:
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set(style='darkgrid')
    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)    
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    
    # assigning a graph to each ax
    sns.boxplot(df[col], ax= ax_box)
    sns.histplot(data=df, x= col, ax=ax_hist)
    # Remove x axis name for the boxplot
    ax_box.set(xlabel="")
    plt.show()





#Seprating X and y 
y = df.loc[:,'Personal Loan']
X = df.loc[:, df.columns != 'Personal Loan']

#Train_Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)




###Decision Tree algo

