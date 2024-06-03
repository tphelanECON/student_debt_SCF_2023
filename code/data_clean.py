"""
Data cleaning for "The Evolution of Student Debt 2019–2022: Evidence from the Survey of Consumer Finances."

    Authors: Emily G. Moschini and Tom Phelan.
    Date written: April 2024.
    Email: tom.phelan@clev.frb.org.

This script downloads the 2019 and 2022 waves of the SCF and generates variables used in the Commentary.

It also prints some numbers quoted in the main text that do not appear in any figure.
"""

import numpy as np
import pandas as pd
import os
from io import BytesIO
import requests
import zipfile

"""
Make folder for figures and data if these do not exist
"""

if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

if not os.path.exists('../data'):
    os.makedirs('../data')

"""
Methods: data fetching, colors for graphs, quantiles, means and aggregates.
"""

def data_from_url(url_var):
    r = requests.get(url_var, stream=True)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall('../data/')
    return pd.read_stata('../data/{0}'.format(z.namelist()[0]))

"""
Functions used in data analysis. quantile arguments: series, weights, desired percentile.
Both for a given series and for a given dataframe.
"""

# following interpolates step-function for cdf
def quantile(data, weights, qct):
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    ind_sorted = np.argsort(data)  # gives indices of the sorted array.
    sorted_weights = weights[ind_sorted] # weights associated with sorted observations
    sn = np.cumsum(sorted_weights)
    pn = sn/sn[-1]
    return np.interp(qct, pn, data[ind_sorted])


def weight_mean(data, weights):
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    return np.sum(data*weights)/np.sum(weights)


def weight_agg(data, weights):
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    return np.sum(data*weights)


def weight_median(data, weights):
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    return quantile(data, weights, 0.5)


def weight_mean_df(df, var):
    return np.sum(df[var]*df['wgt'])/np.sum(df['wgt'])


def weight_agg_df(df, var):
    return np.sum(df[var]*df['wgt'])


def weight_median_df(df, var):
    return quantile(df[var], df['wgt'], 0.5)


"""
Download data and define dictionary used.  
"""

years = [2019, 2022]
scf, scf_debtors, scf_young, scf_young_debtors = {}, {}, {}, {}

"""
Fetch data from Board's website if not on file and create variables used in analysis.
"""

for yr in years:
    if os.path.exists('../data/scf{0}.csv'.format(yr)):
        print("File exists for {0} wave.".format(yr))
    else:
        print("No file exists for {0} wave. Now downloading.".format(yr))
        """
        Get summary dataset (only data used in the Commentary)
        """
        url = 'https://www.federalreserve.gov/econres/files/scfp{0}s.zip'.format(yr)
        scf[yr] = data_from_url(url)
        scf[yr].columns = scf[yr].columns.str.lower()
        print("Summary data for {0} wave created.".format(yr))
        """
        Define indicator for education debt
        """
        scf[yr]['SD_ind'] = scf[yr]['edn_inst'] > 0
        """
        Quintiles. Always defined using the WHOLE population. 
        """
        for var in ["income", "networth"]:
            # be sure to set include_lowest==True so that var+'_cat{0}' includes those with no income
            qctiles = np.array([quantile(scf[yr][var], scf[yr]['wgt'], j/5) for j in range(6)])
            scf[yr][var+'_cat{0}'.format(5)] = pd.cut(scf[yr][var], bins=qctiles, labels=range(len(qctiles)-1), include_lowest=True)
        """
        Save and delete unnecessary files
        """
        print("Now saving wave {0} as .csv and deleting unnecessary STATA files.".format(yr))
        scf[yr].to_csv('../data/scf{0}.csv'.format(yr))
        os.remove('../data/rscfp{0}.dta'.format(yr))

"""
Read in .csv files and define the dataframes used in analysis
"""

for yr in years:
    scf[yr] = pd.read_csv('../data/scf{0}.csv'.format(yr))
    scf_debtors[yr] = scf[yr][scf[yr]['edn_inst'] > 0]
    scf_young[yr] = scf[yr][scf[yr]['age'] < 35]
    scf_young_debtors[yr] = scf_young[yr][scf_young[yr]['edn_inst'] > 0]

"""
Some summary statistics not given in a table or figure
"""

"""
Means and median quoted in introduction
"""

print("Median and mean student debt")
for yr in [2019, 2022]:
    print("Year = {0}:".format(yr))
    print("Median student debt AMONG student debtors:", round(weight_median_df(scf_debtors[yr], 'edn_inst')/10**3, 2), "thousands")
    print("Mean student debt AMONG student debtors:", round(weight_mean_df(scf_debtors[yr], 'edn_inst')/10**3, 2), "thousands")

"""
Incidence, aggregates, and ages (quoted in the main text in section 2.1)
"""

print("Incidence and aggregates")
for yr in [2019, 2022]:
    print("Year = {0}:".format(yr))
    print("Incidence (percentage) in whole population:",
          round(100*np.sum(scf[yr]['SD_ind']*scf[yr]['wgt'])/np.sum(scf[yr]['wgt']), 1))
    print("Agg student debt in SCF:", round(weight_agg_df(scf[yr], 'edn_inst')/10**12, 2), "trillion")
    print("As percent of agg income:", round(100*weight_agg_df(scf[yr], 'edn_inst')/weight_agg_df(scf[yr], 'income'), 2))
    print("As percent of agg net worth:", round(100*weight_agg_df(scf[yr], 'edn_inst')/weight_agg_df(scf[yr], 'networth'), 2))

"""
Ages (quoted in the main text in section 2.1)
"""

print("Ages")
for yr in [2019, 2022]:
    print("Median ages for year = {0}:".format(yr))
    print("Whole population:", weight_median(scf[yr]['age'], scf[yr]['wgt']))
    print("Student debtors:", weight_median(scf_debtors[yr]['age'], scf_debtors[yr]['wgt']))
