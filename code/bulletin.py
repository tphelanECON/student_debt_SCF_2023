"""
As a preliminary check, this script replicates the main features of Table 1
and Table 2 of https://www.federalreserve.gov/publications/files/scf23.pdf

This is simply a check to ensure we are computing things correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, datetime, sys, os
import data_clean

"""
Set dummy to indicate if plots are shown and suppress warning (from https://github.com/twopirllc/pandas-ta/issues/340)
"""
show=0
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

"""
Import the data, functions and lists that we need
"""
scf = data_clean.scf
scf_debtors = data_clean.scf_debtors
scf_nondebtors = data_clean.scf_nondebtors
age_labels = data_clean.age_labels
age_labels_bulletin = data_clean.age_labels_bulletin
quantile = data_clean.quantile
weight_median = data_clean.weight_median
weight_mean = data_clean.weight_mean
weight_agg = data_clean.weight_agg
years = data_clean.years
c1,c2 = data_clean.c1, data_clean.c2
colorFader = data_clean.colorFader

debt_categories = data_clean.debt_categories
name_dict = data_clean.name_dict
summary_rows = data_clean.summary_rows
summary_cols = data_clean.summary_cols

MM_bull = {}
MM_bull_all = {}
MM_index = {}
MM_index['age_cat_bulletin'] = ['Age ' + age for age in age_labels_bulletin]
MM_index['edcl'] = ['No high school diploma', 'High school diploma', 'Some college', 'Bachelor\'s degree+']
MM_index['racecl4'] = ['White non-Hispanic', 'Black/African-American', 'Hispanic/Latino', 'Other/multiple race']
MM_cols = ['Mean', 'Mean post-cancel.', 'Median', 'Median post-cancel.']

for yr in [2019,2022]:
    MM_bull[yr] = {}
    MM_bull_all[yr] = {}
    df = scf[yr].copy()
    bull_cols = ['Median inc.', 'Mean inc.', 'Median net worth', 'Mean net worth']
    for part_var in ['age_cat_bulletin','edcl','racecl4']:
        MM_bull[yr][part_var] = pd.DataFrame(index=MM_index[part_var],columns=bull_cols)
        MM_bull[yr][part_var].iloc[:,0] = df.groupby(df[part_var])['income'].agg(lambda x: quantile(x,df.loc[x.index,"wgt"],0.5)).values
        MM_bull[yr][part_var].iloc[:,1] = df.groupby(df[part_var])['income'].agg(lambda x: weight_mean(x,df.loc[x.index,"wgt"])).values
        MM_bull[yr][part_var].iloc[:,2] = df.groupby(df[part_var])['networth'].agg(lambda x: quantile(x,df.loc[x.index,"wgt"],0.5)).values
        MM_bull[yr][part_var].iloc[:,3] = df.groupby(df[part_var])['networth'].agg(lambda x: weight_mean(x,df.loc[x.index,"wgt"])).values
        #create table
    MM_bull_all[yr] = pd.concat([MM_bull[yr]['age_cat_bulletin'],MM_bull[yr]['edcl'],MM_bull[yr]['racecl4']],axis=0)
    destin = '../main/figures/MM_bulletin_{0}.tex'.format(yr)
    df_table = (MM_bull_all[yr]/10**3).round(decimals=1)
    with open(destin,'w') as tf:
        df_table=df_table.style.format(precision=1)
        tf.write(df_table.to_latex(column_format='lcccc'))

destin = '../main/figures/MM_bulletin_growth.tex'.format(yr)
df_table = (100*(MM_bull_all[2022]/MM_bull_all[2019]-1)).round(decimals=0)
with open(destin,'w') as tf:
    df_table=df_table.style.format(precision=0)
    tf.write(df_table.to_latex(column_format='lcccc'))
