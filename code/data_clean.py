"""
This script downloads the 2019 and 2022 waves of the SCF and generates variables
used in the analysis if such files do not already exist in memory.

February 2024: get rid of things that are not necessary for the Commentary.
The Commentary makes no mention of race or other kinds of debt. So we can get
rid of this stuff.

name_dict only uses income and networth. Everything else seems superfluous.
No per-capita quantities here either.

Documentation links for most recent wave (2022 wave, released in October 2023):

    * Official report (previously referred to the Bulletin) found at:
    https://www.federalreserve.gov/publications/files/scf23.pdf
    Reference: Aditya Aladangady, Jesse Bricker, Andrew C. Chang, Serena Goodman,
    Jacob Krimmel, Kevin B. Moore, Sarah Reber, Alice H. Volz, and Richard A.
    Windle. Changes in U.S. Family Finances from 2019 to 2022: Evidence from
    the Survey of Consumer Finances. 2023.
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
    summing over the first six loans.
    * The SCF asked respondents about the income they earned from the PREVIOUS year.
    * quantiles must be calculated BEFORE any conditioning by debt.
    i.e. when we plot things like average debt among debtors per quintile of
    the distribution of, e.g. net worth, these are not quintiles of the DEBTOR
    population but rather quintiles of the WHOLE population.
    * name_dict is dictionary that relabels variables used in the figures.
    * Per-capita: divide quantities by 2 if married==1. Summary macros: married=1
    if respondent is married or living with their partner (we typically don't
    use per-capita here though).
    * In the public dataset, the PSLF is combined with the forbearance category.
    * Student debt is labelled 'edn_inst' in the summary SCF dataset.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, datetime, pyreadstat, sys, os
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import statsmodels.formula.api as sm
from numpy import mat
from numpy.linalg import inv
import requests, zipfile

"""
TP: suppress performance warnings.

https://stackoverflow.com/questions/15819050/pandas-dataframe-concat-vs-append
https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
"""

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

"""
Questions and variables from full public dataset. Recall SCF only solicits
loan-level info for the first six loans. Also recall that x42001 are weights
(divided by 5 to be consistent with "wgt" in summary dataset).
"""

#Q1. How much is still owed on this loan? (x7179 represents "all other loans")
#Key: 0. NA; otherwise dollar amount
bal_list = ['x7824', 'x7847', 'x7870', 'x7924', 'x7947', 'x7970', 'x7179']
#Q3. Is this loan a federal student loan such as Stafford, Direct, PLUS, or Perkins?
#Key: 1. *YES; 5. *NO; 0. NA
federal_list = ['x7879', 'x7884', 'x7889', 'x7894', 'x7899', 'x7994']
#Q4. (Are you/Is he/Is she/Is he or she) making payments on this loan now?
#Key: 1. *YES; 5. *NO; 0. NA
paynow_list = ['x7806', 'x7829', 'x7852', 'x7906', 'x7929', 'x7952']
#Q5. What is the reason that (you are/he is/she is/he or she is) not making payments
#on (your/his/her/his or her) loan? (Are you/Is he/Is she/Are they) in
#forebearance, a post-graduation grace period, in a job or public service related
#loan forgiveness program, or simply unable to afford the loan payment?
#Key: 1. IN FORBEARANCE; 2. JOB OR PUBLIC SERVICE LOAN FORGIVENESS PROGRAM;
#3. UNABLE TO AFFORD LOAN PAYMENT; 4. POST-GRADUATION GRACE PERIOD OR STILL
#ENROLLED -7. OTHER 0. Inap.
whynopay_list = ['x9300', 'x9301', 'x9302', 'x9303', 'x9304', 'x9305']

full_list = ['yy1','y1','x42001'] + bal_list + federal_list + paynow_list + whynopay_list

"""
Make folder for figures if none exists
"""

if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

if not os.path.exists('../data'):
    os.makedirs('../data')

"""
Methods: data fetching, colors for graphs, quantiles, means and aggregates.
"""

def data_from_url(url):
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall('../data/')
    return pd.read_stata('../data/{0}'.format(z.namelist()[0]))

"""
Functions used in data analysis. quantile arguments: series, weights, desired percentile.
Both for a given series and for a given dataframe.
"""

def quantile(data, weights, quantile):
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    ind_sorted = np.argsort(data)
    sorted_weights = weights[ind_sorted]
    Sn = np.cumsum(sorted_weights)
    Pn = Sn/Sn[-1] #alternative: Pn = (Sn-0.5*sorted_weights)/Sn[-1]
    return np.interp(quantile, Pn, data[ind_sorted])

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

def weight_mean_df(df,var):
    return np.sum(df[var]*df['wgt'])/np.sum(df['wgt'])

def weight_agg_df(df,var):
    return np.sum(df[var]*df['wgt'])

def weight_median_df(df,var):
    return quantile(df[var], df['wgt'], 0.5)

#following is just defined because pd.cut is so long:
def cut(df,var,qctiles):
    return pd.cut(df[var],bins=qctiles,labels=range(len(qctiles)-1),include_lowest=True, duplicates='drop')

#'aliceblue'
c1 = 'crimson'
c2 = 'darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))

"""
Download data, list variables to keep, and join full public and summary dataset
"""

"""
Dictionaries for inflation adjustment. Taken from page 36 of Report. Need separate
adjustment for income and assets because respondents report income from PREVIOUS YEAR in the SCF.
"""

asset_adj = {}
income_adj = {}

asset_adj[2019] = 1.1592
income_adj[2019] = 1.1802
asset_adj[2022] = 1.0000
income_adj[2022] = 1.0809

"""
We are interested in quintiles. Determined by the following.
"""

num = 5

"""
Define years used and dictionaries for the two datasets.
"""

years = [2019,2022]
scf_full, scf_full_real, scf_sum, scf = {}, {}, {}, {}
scf_debtors, scf_private_debtors, scf_nondebtors, scf_young = {}, {}, {}, {}

"""
Naming dictionary for figures. Only need ONE dictionary for whole analysis.
"""

name_dict = {}
name_dict['income'] = 'Income'
name_dict['networth'] = 'Net worth'

"""
Debt categories and table labels/names
"""

summary_rows = ['Median income', 'Mean income', 'Median net worth', 'Mean net worth']
summary_cols = ['Whole population', 'Debtors', 'Private Debtors']

"""
Fetch data from Board's website if not on file and create variables used in analysis.
"""

for yr in years:
    if os.path.exists('../data/scf{0}.csv'.format(yr)):
        print("File exists for {0} wave.".format(yr))
    else:
        print("No file exists for {0} wave. Now downloading.".format(yr))
        tic = time.time()
        """
        Get summary dataset (main source used in Commentary). Label scf_sum.
        """
        url = 'https://www.federalreserve.gov/econres/files/scfp{0}s.zip'.format(yr)
        scf_sum[yr] = data_from_url(url)
        scf_sum[yr].columns = scf_sum[yr].columns.str.lower()
        toc=time.time()
        print("Data for {0} wave created. Time taken: {1} seconds.".format(yr,toc-tic))
        """
        Get full public dataset for loan-level information. Label scf_full.
        """
        tic = time.time()
        url = 'https://www.federalreserve.gov/econres/files/scf{0}s.zip'.format(yr)
        scf_full[yr] = data_from_url(url)
        scf_full[yr].columns = scf_full[yr].columns.str.lower()
        scf_full[yr] = scf_full[yr][full_list]
        toc = time.time()
        print("Time to download full public dataset p{0}i6:".format(str(yr)[2:]), toc-tic)
        """
        Adjust for inflation. Converted variables indicated with _current suffix.
        _full suffix indicates that quantity taken from the full dataset.
        """
        scf_full[yr]['student_debt_full'] = scf_full[yr][[bal_list[i] for i in range(7)]].sum(axis=1)
        scf_full[yr]['student_debt_full_current'] = scf_full[yr]['student_debt_full']*asset_adj[yr]
        """
        Loan-level constructions:
            * Federal versus private.
            * Amount of debt not in repayment.
            * Amount not in repayment broken down by reason for no payment.
        Note: PSLF combined with forbearance category in public dataset.
        """
        for i in range(6):
            scf_full[yr]['fed_bal{0}'.format(i)] = (scf_full[yr][federal_list[i]]==1)*scf_full[yr][bal_list[i]]
            scf_full[yr]['private_bal{0}'.format(i)] = (scf_full[yr][federal_list[i]]==5)*scf_full[yr][bal_list[i]]
            scf_full[yr]['nopay_bal{0}'.format(i)] = (scf_full[yr][paynow_list[i]]==5)*scf_full[yr][bal_list[i]]
            scf_full[yr]['forbear_bal{0}'.format(i)] = (scf_full[yr][whynopay_list[i]]==1)*scf_full[yr][bal_list[i]]
            scf_full[yr]['noafford_bal{0}'.format(i)] = (scf_full[yr][whynopay_list[i]]==3)*scf_full[yr][bal_list[i]]
            scf_full[yr]['grace_bal{0}'.format(i)] = (scf_full[yr][whynopay_list[i]]==4)*scf_full[yr][bal_list[i]]
            for s in ['nopay','forbear','noafford','grace']:
                scf_full[yr][s+'_bal{0}'.format(i)+'_fed'] = (scf_full[yr][federal_list[i]]==1)*scf_full[yr][s+'_bal{0}'.format(i)]
                scf_full[yr][s+'_bal{0}'.format(i)+'_private'] = (scf_full[yr][federal_list[i]]==5)*scf_full[yr][s+'_bal{0}'.format(i)]
        for s in ['nopay','forbear','noafford','grace']:
            scf_full[yr]['student_debt_'+s] = scf_full[yr][[s+'_bal{0}'.format(i) for i in range(6)]].sum(axis=1)
            scf_full[yr]['student_debt_'+s+'_current'] = scf_full[yr]['student_debt_'+s]*asset_adj[yr]
            scf_full[yr]['student_debt_'+s+'_fed'] = scf_full[yr][[s+'_bal{0}'.format(i)+'_fed' for i in range(6)]].sum(axis=1)
            scf_full[yr]['student_debt_'+s+'_private'] = scf_full[yr][[s+'_bal{0}'.format(i)+'_private' for i in range(6)]].sum(axis=1)
            scf_full[yr]['student_debt_'+s+'_fed_current'] = scf_full[yr]['student_debt_'+s+'_fed']*asset_adj[yr]
            scf_full[yr]['student_debt_'+s+'_private_current'] = scf_full[yr]['student_debt_'+s+'_private']*asset_adj[yr]
        scf_full[yr]['student_debt_fed'] = scf_full[yr][['fed_bal{0}'.format(i) for i in range(6)]].sum(axis=1)
        scf_full[yr]['student_debt_private'] = scf_full[yr][['private_bal{0}'.format(i) for i in range(6)]].sum(axis=1)
        scf_full[yr]['student_debt_fed_current'] = scf_full[yr]['student_debt_fed']*asset_adj[yr]
        scf_full[yr]['student_debt_private_current'] = scf_full[yr]['student_debt_private']*asset_adj[yr]
        scf_full[yr]['student_debt_LL_current'] = scf_full[yr]['student_debt_fed_current'] + scf_full[yr]['student_debt_private_current']
        """
        Now merge
        """
        print("Now merging summary and full public datasets:")
        scf_full[yr].set_index(['yy1','y1'],inplace=True)
        scf_sum[yr].set_index(['yy1','y1'],inplace=True)
        scf[yr] = scf_full[yr].join(scf_sum[yr], how='inner')
        """
        Quintiles and deciles
        """
        scf[yr]['student_debt'] = scf[yr]['edn_inst']
        for var in ["income", "networth"]:
            for num in [10,5]:
                #be sure to set include_lowest==True so that var+'_cat{0}' includes those with no income
                qctiles = np.array([quantile(scf[yr][var], scf[yr]['wgt'], j/num) for j in range(num+1)])
                scf[yr][var+'_cat{0}'.format(num)] = pd.cut(scf[yr][var], bins=qctiles, labels=range(len(qctiles)-1), include_lowest=True)
        """
        Save and delete unnecessary files
        """
        print("Now saving wave {0} as .csv and deleting unnecessary STATA files.".format(yr))
        scf[yr].to_csv('../data/scf{0}.csv'.format(yr))
        os.remove('../data/rscfp{0}.dta'.format(yr))
        os.remove('../data/p{0}i6.dta'.format(str(yr)[2:]))

"""
Read in .csv files
"""

for yr in years:
    scf[yr] = pd.read_csv('../data/scf{0}.csv'.format(yr))
    scf_debtors[yr] = scf[yr][scf[yr]['student_debt']>0]
    scf_private_debtors[yr] = scf[yr][scf[yr]['student_debt_private_current']>0]
    scf_nondebtors[yr] = scf[yr][scf[yr]['student_debt']<=0]

"""
A few checks on the above construction
"""

"""
Inflation adjustment
"""

print("Mean student debt in summary and full dataset:")
for yr in [2019,2022]:
    print("Year:", yr)
    print(weight_mean(scf[yr]['student_debt'],scf[yr]['wgt']))
    print(weight_mean(scf[yr]['student_debt_full_current'],scf[yr]['x42001']/5))

"""
Quintiles calculations
"""

for var in ['income','networth']:
    print('Counts in {0} quintiles'.format(var))
    for yr in [2019,2022]:
        print("Year:", yr)
        print(scf[yr].groupby(var+'_cat'+str(num))['wgt'].sum())

"""
Quantitative significance of "all other loans." We cannot ascertain loan-level
info for these loans. However, they are very small in the aggregate.
"""

print("Percentage of student debt not among first six:")
for yr in [2019,2022]:
    print("Year = {0}:".format(yr))
    other_SD = weight_agg(scf[yr]['x7179'],scf[yr]['wgt'])
    tot_SD = weight_agg(scf[yr]['student_debt'],scf[yr]['wgt'])
    print("Ratio:", 100*other_SD/tot_SD)

"""
Ages (quoted in the main text in section 2.1)
"""
for yr in [2019,2022]:
    print("Mean ages for year = {0}:".format(yr))
    print("Whole population:", weight_median(scf[yr]['age'],scf[yr]['wgt']))
    print("Student debtors:", weight_median(scf_debtors[yr]['age'],scf_debtors[yr]['wgt']))
