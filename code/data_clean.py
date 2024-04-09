"""
This script downloads the 2019 and 2022 waves of the SCF and generates variables
used in the analysis if such files do not already exist in memory.

It also prints some checks (comparisons with the official report) and prints some numbers
that are used in the main text.

Documentation links for the 2022 wave released in October 2023:

    * Official report (previously referred to the Bulletin) found at:
    https://www.federalreserve.gov/publications/files/scf23.pdf
    * Summary macros: https://www.federalreserve.gov/econres/files/bulletin.macro.txt
    * Codebook for 2022: https://www.federalreserve.gov/econres/files/codebk2022.txt

With each new wave of the SCF the Board updates the previous summary datasets
to be in current dollars but does NOT do this with the full public dataset.
From page 36 of the Report we adjust nominal figures in the full public dataset as follows:

For 2019:
    * Adjustment factor for assets and debts in survey year = 1.1592
    * Adjustment factor for income in year before survey year = 1.1802

For 2022:
    * Adjustment factor for assets and debts in survey year = 1.0000
    * Adjustment factor for income in year before survey year = 1.0809

Reminders/caveats/concerns:
    * edn_inst in the summary dataset differs from what we obtain when
    summing over the first six loans. However, the difference is negligible and simply noted in a footnote.
    * The SCF asked respondents about the income they earned from the PREVIOUS year. The respondent is
    therefore providing a figure denominated in the previous year's units. It is for this reason that we must
    adjust as per the above factor. This is NOT necessary for the summary dataset.
    * quantiles must be calculated BEFORE any conditioning by debt.
    i.e. when we plot things like average debt among debtors per quintile of
    the distribution of, e.g. net worth, these are not quintiles of the DEBTOR
    population but rather quintiles of the WHOLE population.
    * name_dict is dictionary that relabels variables used in the figures.
    * Per-capita: divide quantities by 2 if married==1. Summary macros: married=1
    if respondent is married or living with their partner.
    NOTE THAT THE CURRENT COMMENTARY DOES NOT USE PER-CAPITA FIGURES.
    * In the public dataset, the PSLF is combined with the forbearance category.
    * Student debt is labelled 'edn_inst' in the summary SCF dataset.

Nothing in the current Commentary requires anything from the loan-level information. This is ALL aggregates.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import os
from io import BytesIO
# from zipfile import ZipFile
# from urllib.request import urlopen
import requests
import zipfile
from warnings import simplefilter

"""
TP: suppress performance warnings. See: https://stackoverflow.com/questions/68292862/
"""

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

"""
Questions and variables from full public dataset. Recall SCF only solicits
loan-level info for the first six loans. Also recall that x42001 are weights
(divided by 5 to be consistent with "wgt" in summary dataset).
"""

# Q1. How much is still owed on this loan? (x7179 represents "all other loans")
# Key: 0. NA; otherwise dollar amount
bal_list = ['x7824', 'x7847', 'x7870', 'x7924', 'x7947', 'x7970', 'x7179']
# Q3. Is this loan a federal student loan such as Stafford, Direct, PLUS, or Perkins?
# Key: 1. *YES; 5. *NO; 0. NA
fed_list = ['x7879', 'x7884', 'x7889', 'x7894', 'x7899', 'x7994']
# Q4. (Are you/Is he/Is she/Is he or she) making payments on this loan now?
# Key: 1. *YES; 5. *NO; 0. NA
paynow_list = ['x7806', 'x7829', 'x7852', 'x7906', 'x7929', 'x7952']
# Q5. What is the reason that (you are/he is/she is/he or she is) not making payments
# on (your/his/her/his or her) loan? (Are you/Is he/Is she/Are they) in
# forebearance, a post-graduation grace period, in a job or public service related
# loan forgiveness program, or simply unable to afford the loan payment?
# Key: 1. IN FORBEARANCE; 2. JOB OR PUBLIC SERVICE LOAN FORGIVENESS PROGRAM;
# 3. UNABLE TO AFFORD LOAN PAYMENT; 4. POST-GRADUATION GRACE PERIOD OR STILL
# ENROLLED -7. OTHER 0. Inap.
whynopay_list = ['x9300', 'x9301', 'x9302', 'x9303', 'x9304', 'x9305']

full_list = ['yy1', 'y1', 'x42001'] + bal_list + fed_list + paynow_list + whynopay_list

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


def quantile(data, weights, qct):
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    ind_sorted = np.argsort(data)
    sorted_weights = weights[ind_sorted]
    sn = np.cumsum(sorted_weights)
    pn = sn/sn[-1]  # : Pn = (sn-0.5*sorted_weights)/sn[-1]
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


# following is just defined because pd.cut is so long:
def cut(df, var, qctiles):
    return pd.cut(df[var], bins=qctiles, labels=range(len(qctiles)-1), include_lowest=True, duplicates='drop')


# 'aliceblue'
c1 = 'crimson'
c2 = 'darkblue'


def colorfader(color1, color2, mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(color1)) + mix*np.array(mpl.colors.to_rgb(color2)))


"""
Download data. Only need the summary dataset. 
"""

"""
Define years used and dictionaries for the two datasets.
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
        Get summary dataset (main source used in Commentary). Label scf_sum.
        """
        url = 'https://www.federalreserve.gov/econres/files/scfp{0}s.zip'.format(yr)
        scf[yr] = data_from_url(url)
        scf[yr].columns = scf[yr].columns.str.lower()
        print("Summary data for {0} wave created.".format(yr))
        """
        Quintiles. Note that quintiles are always defined using the whole population. 
        """
        scf[yr]['SD'] = scf[yr]['edn_inst']
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
Read in .csv files
"""

for yr in years:
    scf[yr] = pd.read_csv('../data/scf{0}.csv'.format(yr))
    scf_debtors[yr] = scf[yr][scf[yr]['SD'] > 0]
    scf_young[yr] = scf[yr][scf[yr]['age'] < 35]
    scf_young_debtors[yr] = scf_young[yr][scf_young[yr]['SD'] > 0]

"""
Some summary statistics not given in a table or figure
"""

"""
Statistics mentioned in introduction and first part of the analysis.
"""

"""
Means and median
"""

for yr in [2019]:
    print("Year = {0}:".format(yr))
    print("Median student debt:")
    print("Among student debtors:", weight_median_df(scf_debtors[yr], 'SD'))
    print("Among <35 population of student debtors:", weight_median_df(scf_young_debtors[yr], 'SD'))
    print("Mean student debt:")
    print("Whole population:", weight_mean_df(scf[yr], 'SD'))
    print("Among student debtors:", weight_mean_df(scf_debtors[yr], 'SD'))
    print("Among <35 population:", weight_mean_df(scf_young[yr], 'SD'))
    print("Among <35 population with student debt:", weight_mean_df(scf_young_debtors[yr], 'SD'))

print("Incidence")

for yr in [2019, 2022]:
    print("Year = {0}:".format(yr))
    print("Incidence in whole population:", weight_agg_df(scf_debtors[yr], 'wgt')/weight_agg_df(scf[yr], 'wgt'))
    print("Incidence in <35 population:", weight_agg_df(scf_young_debtors[yr], 'wgt')/weight_agg_df(scf_young[yr], 'wgt'))

print("Aggregates")

for yr in [2019, 2022]:
    print("Year = {0}:".format(yr))
    print("As percent of agg income:", round(100*weight_agg_df(scf[yr], 'SD')/weight_agg_df(scf[yr], 'income'), 2))
    print("As percent of agg net worth:", round(100*weight_agg_df(scf[yr], 'SD')/weight_agg_df(scf[yr], 'networth'), 2))
    print("As percent of agg income (<35 population):", round(100*weight_agg_df(scf_young[yr], 'SD')/weight_agg_df(scf_young[yr], 'income'), 2))
    print("As percent of agg net worth (<35 population):", round(100*weight_agg_df(scf_young[yr], 'SD')/weight_agg_df(scf_young[yr], 'networth'), 2))

print("Young versus old split")

for yr in [2019, 2022]:
    print("Year = {0}:".format(yr))
    print("Percent of student debt held by <35 population:",
          round(100*weight_agg_df(scf_young[yr], 'SD')/weight_agg_df(scf[yr], 'SD'), 2))
    print("Percent of families <35:",
          round(100 * weight_agg_df(scf_young[yr], 'wgt') / weight_agg_df(scf[yr], 'wgt'), 2))

"""

"""


"""
Ages (quoted in the main text in section 2.1)
"""

for yr in [2019, 2022]:
    print("Median ages for year = {0}:".format(yr))
    print("Whole population:", weight_median(scf[yr]['age'], scf[yr]['wgt']))
    print("Student debtors:", weight_median(scf_debtors[yr]['age'], scf_debtors[yr]['wgt']))

