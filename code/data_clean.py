"""
This script downloads various waves of the SCF and generates variables used in
the analysis if such files do not already exist in memory.

Documentation links for most recent wave (2022 wave, released in October 2023):

    * Report (previously referred to the Bulletin): https://www.federalreserve.gov/publications/files/scf23.pdf
    * Summary macros: https://www.federalreserve.gov/econres/files/bulletin.macro.txt
    * Codebook for 2022: https://www.federalreserve.gov/econres/files/codebk2022.txt

To search codebook note that analysis of education loans begins with X7801.

In this script, the "Report" refers to the above file. The full reference is

Aditya Aladangady, Jesse Bricker, Andrew C. Chang, Serena Goodman, Jacob Krim-
mel, Kevin B. Moore, Sarah Reber, Alice H. Volz, and Richard A. Windle. Changes
in U.S. Family Finances from 2019 to 2022: Evidence from the Survey of Consumer
Finances. 2023.

With each new wave of the SCF the Board updates the previous summary datasets
to be in current dollars but does NOT do this with the full public dataset.
The following is ad-hoc but suffices for our purposes: from page 36 of the Report
we adjust nominal figures in the full public dataset as follows:

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
    * simple lifetime wealth calculations are placed here. These may or may not
    appear in the Commentary
    * universal cancellation effects here too? Not sure.
    * perhaps place methods elsewhere eventually. they clutter and do not require
    reference to data.

Note that in the public dataset, the PSLF is combined with the forbearance
category.


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
TP: I am not concerned with performance warnings and so I ignore the following.
They seem to come up when we append many series to our dataframes. See e.g.

https://stackoverflow.com/questions/15819050/pandas-dataframe-concat-vs-append
https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
"""

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

"""
Questions and variables taken from the full public dataset. Recall that SCF
only solicits loan-level info for the first six loans.
"""

#Q1. How much is still owed on this loan? (x7179 represents "all other loans")
#Key: 0. NA; otherwise dollar amount
bal_list = ['x7824', 'x7847', 'x7870', 'x7924', 'x7947', 'x7970', 'x7179']
#Q2. Is the payment amount (on this loan) (you/he/she/he or she) owe each month determined
#by (your/his/her/his or her) income, for example an Income-Based Repayment Plan,
#Pay as you Earn Plan, or Income-Contingent Repayment Plan?
#Key: 1. *YES; 5. *NO; 0. NA
IDR_list = ['x9306', 'x9307', 'x9308', 'x9309', 'x9310', 'x9311']
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
#Q6. Is the amount owed on this loan being completely forgiven or partially forgiven?
#Key: 1. *COMPLETELY FORGIVEN; 2. *PARTIALLY FORGIVEN; -7  OTHER; 0. Inap.
forgive_list = ['x7421', 'x7423', 'x7425', 'x7427', 'x7429', 'x7431']
#Q7. What is the annual rate of interest charged on this loan?
#Key. PERCENT * 100. -1. Nothing. 0. Inap.
#interest_list = ['x7822', 'x7845', 'x7868', 'x7922', 'x7945', 'x7968']

"""
For following it does not seem meaningfu to talk about number of loans falling
into particular category. Instead we record dollar value of all such loans.

Now think: what else to depict? Ideas:
    * value of all loans not currently in repayment.
    * value of all loans in an IDR. Typical income/wealth of debtors in an IDR.
    * value of all loans that respondents expect to never repay. Reasons why.
"""

"""
Create full list of variables taken from full public dataset: x42001 are weights
(divided by 5 in order to be consistent with "wgt" in summary dataset).
Also, x5702 is wage income.
"""

#full_list = ['yy1','y1','x42001'] + bal_list + IDR_list + federal_list \
#+ paynow_list + whynopay_list + forgive_list + ['x5702']

full_list = ['yy1','y1','x42001'] + bal_list + IDR_list + federal_list \
+ paynow_list + whynopay_list + ['x5702']


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
Dictionaries for inflation adjustment. Need separate adjustment factor for
income and assets. Following taken from page 36 of the Report.
"""

asset_adj = {}
income_adj = {}

asset_adj[2019] = 1.1592
income_adj[2019] = 1.1802
asset_adj[2022] = 1.0000
income_adj[2022] = 1.0809

debt_list = [0, 1, 1.5*10**4, 4*10**4, np.inf]
debt_brackets = ["No debt","\$1-\$15,000", "\$15,001-\$40,000", "\$40,001+"]

"""
Sample years. Only consider the last two years.
"""
start, end = '1989-03-01', '2023-06-01'
start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
"""
possible_years = [1995,1998,2001,2004,2007,2010,2013,2016,2019,2022]
diff_years = [1995,1998,2001]
years = [yr for yr in possible_years if (yr >= start_dt.year and yr <= end_dt.year)]
"""
years = [2019,2022]
scf_full, scf_full_real, scf_sum, scf = {}, {}, {}, {}
scf_debtors, scf_private_debtors, scf_nondebtors, scf_young = {}, {}, {}, {}
"""
Age distribution (Boundaries: 30-year-old is in group 0.)
"""
age_labels = ["26-30","31-35","36-40","41-45","46-50","51-55","56-60","61-65","66-70"]
age_labels_bulletin = ["Under 35","35-44","45-54","55-64","65-74","Over 75"]
age_values = [25,30,35,40,45,50,55,60,65,70]
age_values_bulletin = [0,34,44,54,64,74,1000]
young_cat = [0,1]
part_dict = ['age', 'edcl', 'racecl4']

"""
Naming dictionary for figures. Only need ONE dictionary for whole analysis.
"""

debt_categories = ['student_debt','res_debt','ccbal','veh_inst']
name_dict = {}
name_dict['income'] = 'Income'
name_dict['networth'] = 'Net worth'
name_dict['asset'] = 'Asset'
name_dict['debt'] = 'Total debt'
name_dict['student_debt'] = 'Student debt'
name_dict['edn_inst'] = 'Student debt'
name_dict['res_debt'] = 'Residential debt'
name_dict['ccbal'] = 'Credit card debt'
name_dict['veh_inst'] = 'Auto loans'
name_dict['age'] = 'Age'
name_dict['edcl'] = 'Education'
name_dict['racecl4'] = 'Race'
name_dict['inc_networth'] = 'Income plus net worth'

"""
Debt categories and table labels/names
"""

summary_rows = ['Median income', 'Mean income', 'Median networth', 'Mean networth']
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
        Get summary dataset (main source in Commentary). Label scf_sum.
        """
        url = 'https://www.federalreserve.gov/econres/files/scfp{0}s.zip'.format(yr)
        scf_sum[yr] = data_from_url(url)
        scf_sum[yr].columns = scf_sum[yr].columns.str.lower()
        toc=time.time()
        print("Data for {0} wave created. Time taken: {1} seconds.".format(yr,toc-tic))
        """
        Get full public dataset for loan-level information.
        """
        tic = time.time()
        url = 'https://www.federalreserve.gov/econres/files/scf{0}s.zip'.format(yr)
        scf_full[yr] = data_from_url(url)
        scf_full[yr].columns = scf_full[yr].columns.str.lower()
        scf_full[yr] = scf_full[yr][full_list]
        toc = time.time()
        print("Time to download full public dataset p{0}i6:".format(str(yr)[2:]), toc-tic)
        """
        Convert certain variables using the inflation adjustment factors.
        """
        scf_full[yr]['student_debt_full'] = scf_full[yr][[bal_list[i] for i in range(7)]].sum(axis=1)
        scf_full[yr]['student_debt_full_current'] = scf_full[yr]['student_debt_full']*asset_adj[yr]
        """
        Loan-level constructions (IDR, federal vs private loans, forbearance, etc).

        Remember PSLF is combined with forbearance category.
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
        #now want dollar value of loans in repayment.
        for i in range(6):
            scf_full[yr]['IDR_ind{0}'.format(i)] = (scf_full[yr][IDR_list[i]]==1).astype(int)
        scf_full[yr]['IDR_count'] = scf_full[yr][['IDR_ind{0}'.format(i) for i in range(6)]].sum(axis=1)
        scf_full[yr]['IDR'] = scf_full[yr]['IDR_count']>0
        """
        Now merge
        """
        print("Now merging summary and full public datasets:")
        scf_full[yr].set_index(['yy1','y1'],inplace=True)
        scf_sum[yr].set_index(['yy1','y1'],inplace=True)
        scf[yr] = scf_full[yr].join(scf_sum[yr], how='inner')
        """
        Additional variables: dummy, debts, age categorial var., per-capita variables.

        NOTE: not clear that we use all of these. Consider deleting before publication.
        """
        scf[yr]['unit'] = 1
        scf[yr]['student_debt'] = scf[yr]['edn_inst']
        scf[yr]['res_debt'] = scf[yr]['nh_mort'] + scf[yr]['heloc'] + scf[yr]['resdbt']
        scf[yr]['nonres_debt'] = scf[yr]['debt'] - scf[yr]['res_debt']
        scf[yr]['nonres_nonSL_debt'] = scf[yr]['nonres_debt'] - scf[yr]['student_debt']
        scf[yr]['age_cat'] = pd.cut(scf[yr]['age'],bins=age_values,labels=range(len(age_values)-1))
        scf[yr]['age_cat_bulletin'] = pd.cut(scf[yr]['age'],bins=age_values_bulletin,labels=range(len(age_values_bulletin)-1))
        scf[yr]['inc_networth'] = scf[yr]['income'] + scf[yr]['networth']
        for debt in debt_categories:
            scf[yr]['has_{0}'.format(debt)] = scf[yr][debt]>0
        for var in ['student_debt','wageinc','income','asset','networth']:
            scf[yr]['percap_' + var] = (1 - (scf[yr]['married']==1)/2)*scf[yr][var]
        for var in ["income", "networth"]:
            for num in [10,5]:
                #be sure to set include_lowest==True so that var+'_cat{0}' includes those with no income
                qctiles = np.array([quantile(scf[yr][var], scf[yr]['wgt'], j/num) for j in range(num+1)])
                scf[yr][var+'_cat{0}'.format(num)] = pd.cut(scf[yr][var], bins=qctiles, labels=range(len(qctiles)-1), include_lowest=True)
                qctiles = np.array([quantile(scf[yr]['percap_'+var], scf[yr]['wgt'], j/num) for j in range(num+1)])
                scf[yr]['percap_'+var+'_cat{0}'.format(num)] = pd.cut(scf[yr]['percap_'+var], bins=qctiles, labels=range(len(qctiles)-1), include_lowest=True)
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
print("Now importing data.")
tic = time.time()
for yr in years:
    print("{0} wave".format(yr))
    scf[yr] = pd.read_csv('../data/scf{0}.csv'.format(yr))
    scf_debtors[yr] = scf[yr][scf[yr]['student_debt']>0]
    scf_private_debtors[yr] = scf[yr][scf[yr]['student_debt_private_current']>0]
    scf_nondebtors[yr] = scf[yr][scf[yr]['student_debt']<=0]
    scf_young[yr] = scf[yr][scf[yr]['age_cat'].isin(young_cat)]
toc = time.time()
print("Time taken:", toc-tic)

"""
Check that we have the correct inflation adjustment
"""

for yr in [2019,2022]:
    print("Year:", yr)
    print("Mean student debt in summary and full dataset:")
    print(weight_mean(scf[yr]['student_debt'],scf[yr]['wgt']))
    print(weight_mean(scf[yr]['student_debt_full_current'],scf[yr]['x42001']/5))

"""
Compute number of people in IDR
"""

for yr in [2019,2022]:
    print("Year = {0}".format(yr))
    print("Fraction of student debtors enrolled in an IDR:")
    print(weight_mean(scf_debtors[yr]['IDR'],scf_debtors[yr]['x42001']/5))
    print("Fraction of student debt that is federal:")
    tot_SD = weight_agg(scf[yr]['student_debt_full_current'],scf[yr]['x42001']/5)
    tot_SD2 = weight_agg(scf[yr]['student_debt'],scf[yr]['wgt'])
    fed_SD = weight_agg(scf[yr]['student_debt_fed_current'],scf[yr]['x42001']/5)
    print(fed_SD/tot_SD)

"""
Some statistics on quantitative significance of "all other loans."
TP: I check this because we cannot ascertain loan-level information for these
loans. Thankfully, they are very small in the aggregate.
"""

print("Average amount of loans beyond first six:")
for yr in [2019,2022]:
    print("Year = {0}:".format(yr))
    print(weight_mean(scf[yr]['x7179'],scf[yr]['wgt']))
    print("Average loans")
    print(weight_mean(scf[yr]['student_debt'],scf[yr]['wgt']))

print("Now among borrowers:")
for yr in [2019,2022]:
    print("Year = {0}:".format(yr))
    print(weight_mean(scf_debtors[yr]['x7179'],scf_debtors[yr]['wgt']))
    print("Average loans")
    print(weight_mean(scf_debtors[yr]['student_debt'],scf_debtors[yr]['wgt']))

"""
Averages check
"""

print("Average debt loan-level versus total:")
for yr in [2019,2022]:
    scf[yr]['student_debt_LL_check'] = scf[yr]['student_debt_LL_current'] + scf[yr]['x7179']
    print("Year = {0}:".format(yr))
    print("Loan-level:",weight_mean(scf[yr]['student_debt_LL_current'],scf[yr]['wgt']))
    print("Loan-level plus misc:",weight_mean(scf[yr]['student_debt_LL_check'],scf[yr]['wgt']))
    print("Full dataset:",weight_mean(scf[yr]['student_debt_full_current'],scf[yr]['wgt']))
    print("Summary dataset:",weight_mean(scf[yr]['student_debt'],scf[yr]['wgt']))
