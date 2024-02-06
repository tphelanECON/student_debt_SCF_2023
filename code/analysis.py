"""
This script performs the analysis. Data is constructed in data_clean.py.

Contains the following:
    * Print summary statistics and make tables.
    * Average student debt by quintiles, broken down by:
        * wave of SCF (2019 or 2022).
        * whole population and population of debtors.
        * income and net worth quintiles.
        * public vs private loans, fraction not in repayment, reason for lack
        of repayment.
    * Incidence of debt by quintiles of income and net worth.
    * Ratio of conditional mean to median student debt. Only for total student
    debt because median PRIVATE student debt among borrowers often zero.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, datetime, pyreadstat, sys, os
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
scf_private_debtors = data_clean.scf_private_debtors
scf_nondebtors = data_clean.scf_nondebtors
quantile = data_clean.quantile
weight_median = data_clean.weight_median
weight_mean = data_clean.weight_mean
weight_agg = data_clean.weight_agg
weight_median_df = data_clean.weight_median_df
weight_mean_df = data_clean.weight_mean_df
weight_agg_df = data_clean.weight_agg_df

years = data_clean.years
c1,c2 = data_clean.c1, data_clean.c2
colorFader = data_clean.colorFader

name_dict = data_clean.name_dict
summary_rows = data_clean.summary_rows
summary_cols = data_clean.summary_cols

num = data_clean.num

"""
Statistics mentioned in introduction and first part of the analysis.
"""

for yr in [2019,2022]:
    print("Year = {0}:".format(yr))
    print("Median student debt among borrowers:", weight_median_df(scf_debtors[yr],'edn_inst'))
    print("Mean student debt:")
    print("Whole population:", weight_mean_df(scf[yr],'edn_inst'))
    print("Among borrowers:", weight_mean_df(scf_debtors[yr],'edn_inst'))
    print("Incidence of student debt:", weight_agg_df(scf_debtors[yr],'wgt')/weight_agg_df(scf[yr],'wgt'))
    print("Aggregate student debt/aggregate income:", weight_agg_df(scf[yr],'edn_inst')/weight_agg_df(scf[yr],'income'))

"""
Summary tables (for the "summary statistics" section)
"""

for yr in [2019,2022]:
    df = pd.DataFrame(data=0,index=summary_rows,columns=summary_cols)
    df_list = [scf[yr],scf_debtors[yr],scf_private_debtors[yr]]
    for i, d in enumerate(df_list):
        df.iloc[0,i] = weight_median_df(d,'income')/10**3
        df.iloc[1,i] = weight_mean_df(d,'income')/10**3
        df.iloc[2,i] = weight_median_df(d,'networth')/10**3
        df.iloc[3,i] = weight_mean_df(d,'networth')/10**3

    destin = '../main/figures/sd_summary_{0}.tex'.format(yr)
    df_table = df.round(decimals=1)
    with open(destin,'w') as tf:
        df_table=df_table.style.format(precision=1)
        tf.write(df_table.to_latex(column_format='lccc'))

"""
Average student debt by quintiles, broken down by:
    * wave of SCF (2019 or 2022).
    * whole population and population of debtors.
    * income and net worth quintiles.
    * public and private loan.
    * mean and median.
"""

"""
Now some fontsize decisions
"""

show=0
xlabelfontsize=14
ylabelfontsize=14
titlefontsize=14
num=5

debt_var_list = ['student_debt','student_debt_private_current']

for debt_var in debt_var_list:
    for var in ['income','networth']:
        for yr in [2019,2022]:
            for measure in ['mean']:
                SD_quintiles = pd.DataFrame(columns=range(1,num+1), index=['borrowers','all'])
                for var2 in ['borrowers','all']:
                    if var2 == 'borrowers':
                        df_temp = scf_debtors[yr]
                    else:
                        df_temp = scf[yr]
                    f = lambda x: np.average(x, weights=df_temp.loc[x.index, "wgt"])
                    g = lambda x: data_clean.weight_median(x, weights=df_temp.loc[x.index, "wgt"])
                    if measure=='mean':
                        SD_quintiles.loc[var2,:] = df_temp.groupby(var+'_cat'+str(num))[debt_var].agg(f).values
                    else:
                        SD_quintiles.loc[var2,:] = df_temp.groupby(var+'_cat'+str(num))[debt_var].agg(g).values
                SD_quintiles = (SD_quintiles/1000).astype(float).round(1)
                width = 1/5
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for i in range(5):
                    if i==0:
                        ax.bar(i+1-width,SD_quintiles.loc['borrowers',i+1],2*width,color=colorFader(c1,c2,0),label="Borrowers")
                        ax.bar(i+1+width,SD_quintiles.loc['all',i+1],2*width,color=colorFader(c1,c2,1),label="All")
                    else:
                        ax.bar(i+1-width,SD_quintiles.loc['borrowers',i+1],2*width,color=colorFader(c1,c2,0))
                        ax.bar(i+1+width,SD_quintiles.loc['all',i+1],2*width,color=colorFader(c1,c2,1))
                if debt_var == 'student_debt':
                    ax.set_title('Average student debt by quintile ({0})'.format(yr), fontsize=titlefontsize)
                    plt.ylim([0,80])
                else:
                    ax.set_title('Average private student debt by quintile ({0})'.format(yr), fontsize=titlefontsize)
                    plt.ylim([0,20])
                ax.set_xlabel('{0} quintiles'.format(name_dict[var]), fontsize=xlabelfontsize)
                ax.set_ylabel('\$000s', fontsize=ylabelfontsize)
                ax.legend()
                destin = '../main/figures/SD_{0}_{1}_quintiles_{2}_{3}.png'.format(measure,var,yr,debt_var)
                plt.savefig(destin, format='png', dpi=1000)
                if show == 1:
                    plt.show()
                plt.close()

print("Borrowers vs non-borrowers figures done")

"""
Above compares borrowers vs all for specific year.
Now opposite: 2019 vs 2022 for borrowers and all separately.
"""

for debt_var in debt_var_list:
    for var in ['income','networth']:
        for var2 in ['borrowers','all']:
            for measure in ['mean']:
                SD_quintiles = pd.DataFrame(columns=range(1,num+1), index=[2019,2022])
                for yr in [2019,2022]:
                    if var2 == 'borrowers':
                        df_temp = scf_debtors[yr]
                    else:
                        df_temp = scf[yr]
                    f = lambda x: np.average(x, weights=df_temp.loc[x.index, "wgt"])
                    g = lambda x: data_clean.weight_median(x, weights=df_temp.loc[x.index, "wgt"])
                    if measure=='mean':
                        SD_quintiles.loc[yr,:] = df_temp.groupby(var+'_cat'+str(num))[debt_var].agg(f).values
                    elif measure=='median':
                        SD_quintiles.loc[yr,:] = df_temp.groupby(var+'_cat'+str(num))[debt_var].agg(g).values
                SD_quintiles = (SD_quintiles/1000).astype(float).round(1)
                width = 1/5
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for i in range(5):
                    if i==0:
                        ax.bar(i+1-width,SD_quintiles.loc[2019,i+1],2*width,color=colorFader(c1,c2,0),label=2019)
                        ax.bar(i+1+width,SD_quintiles.loc[2022,i+1],2*width,color=colorFader(c1,c2,1),label=2022)
                    else:
                        ax.bar(i+1-width,SD_quintiles.loc[2019,i+1],2*width,color=colorFader(c1,c2,0))
                        ax.bar(i+1+width,SD_quintiles.loc[2022,i+1],2*width,color=colorFader(c1,c2,1))
                plt.legend()
                ax.set_xlabel('{0} quintile'.format(name_dict[var]), fontsize=xlabelfontsize)
                ax.set_ylabel('\$000s', fontsize=ylabelfontsize)
                if debt_var == 'student_debt':
                    if var2 == 'all':
                        ax.set_title('Average student debt', fontsize=titlefontsize)
                    else:
                        ax.set_title('Average student debt among borrowers', fontsize=titlefontsize)
                else:
                    if var2 == 'all':
                        ax.set_title('Average private student debt', fontsize=titlefontsize)
                    else:
                        ax.set_title('Average private student debt among borrowers', fontsize=titlefontsize)
                destin = '../main/figures/{0}_debt_ave_{1}_{2}.png'.format(var,debt_var,var2)
                plt.savefig(destin, format='png', dpi=1000)
                if show == 1:
                    plt.show()
                plt.close()

print("2019 vs 2022 figures done")

"""
Incidence of debt by quintiles of income and net worth.
"""

debt_list_binary = [0, 1, np.inf]
for debt_var in debt_var_list:
    for var in ['income','networth']:
        SD_quintiles_frac = pd.DataFrame(columns=range(1,num+1), index=[2019,2022])
        for yr in [2019,2022]:
            data = scf[yr]
            quintiles = np.array([data_clean.quantile(data[var],data['wgt'], j/5) for j in range(6)])
            qct_lists, var_names = [quintiles,debt_list_binary],[var,debt_var]
            d = [pd.cut(data[var_names[i]], bins=qct_lists[i],labels=range(len(qct_lists[i])-1),include_lowest=True,duplicates='drop') for i in range(2)]
            data['pairs'] = list(zip(d[0], d[1]))
            SD_debt = data.groupby(data['pairs'])[debt_var]
            SD_debt_count = data.groupby(data['pairs'])['wgt'].sum()
            SD_quintiles_frac.loc[yr,:] = [SD_debt_count[(i,1)]/(SD_debt_count[(i,0)]+SD_debt_count[(i,1)]) for i in range(num)]
        SD_quintiles_pct = 100*SD_quintiles_frac
        width = 1/5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(5):
            if i==0:
                ax.bar(i+1-width,SD_quintiles_pct.loc[2019,i+1],2*width,color=colorFader(c1,c2,0),label=2019)
                ax.bar(i+1+width,SD_quintiles_pct.loc[2022,i+1],2*width,color=colorFader(c1,c2,1),label=2022)
            else:
                ax.bar(i+1-width,SD_quintiles_pct.loc[2019,i+1],2*width,color=colorFader(c1,c2,0))
                ax.bar(i+1+width,SD_quintiles_pct.loc[2022,i+1],2*width,color=colorFader(c1,c2,1))
        plt.legend()
        ax.set_xlabel('{0} quintile'.format(name_dict[var]),fontsize=xlabelfontsize)
        ax.set_ylabel('Percent (%)',fontsize=ylabelfontsize)
        if debt_var == 'student_debt':
            ax.set_title('Percentage with student debt',fontsize=titlefontsize)
            ax.set_ylim([0, 50])
        elif debt_var == 'student_debt_fed_current':
            ax.set_title('Percentage with federal student debt',fontsize=titlefontsize)
        else:
            ax.set_title('Percentage with private student debt',fontsize=titlefontsize)
        destin = '../main/figures/{0}_debt_inc_{1}.png'.format(var,debt_var)
        plt.savefig(destin, format='png', dpi=1000)
        if show == 1:
            plt.show()
        plt.close()

print("2019 vs 2022 incidence figures done")

"""
Ratio of conditional mean to median student debt. Only for total student debt
because median PRIVATE student debt among borrowers often zero.
"""

for debt_var in ['student_debt']:
    for var in ['income','networth']:
        SD_quintiles_rat = pd.DataFrame(columns=range(1,num+1), index=[2019,2022])
        for yr in [2019,2022]:
            print(debt_var, var, yr)
            data = scf_debtors[yr]
            f = lambda x: np.average(x, weights=data.loc[x.index, "wgt"])
            g = lambda x: data_clean.weight_median(x, weights=data.loc[x.index, "wgt"])
            gb = data.groupby(var+'_cat{0}'.format(num))[debt_var].agg(f).values
            gb_med = data.groupby(var+'_cat{0}'.format(num))[debt_var].agg(g).values
            SD_quintiles_rat.loc[yr,:] = gb/gb_med
            width = 1/5

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(5):
            if i==0:
                ax.bar(i+1-width,SD_quintiles_rat.loc[2019,i+1],2*width,color=colorFader(c1,c2,0),label=2019)
                ax.bar(i+1+width,SD_quintiles_rat.loc[2022,i+1],2*width,color=colorFader(c1,c2,1),label=2022)
            else:
                ax.bar(i+1-width,SD_quintiles_rat.loc[2019,i+1],2*width,color=colorFader(c1,c2,0))
                ax.bar(i+1+width,SD_quintiles_rat.loc[2022,i+1],2*width,color=colorFader(c1,c2,1))
        plt.legend()
        ax.set_xlabel('{0} quintile'.format(name_dict[var]), fontsize=xlabelfontsize)
        ax.set_ylabel('Ratio', fontsize=ylabelfontsize)
        ax.set_ylim([0, 2.2])
        ax.set_title('Ratio of mean-to-median student debt', fontsize=titlefontsize)
        destin = '../main/figures/{0}_MM_rat_{1}.png'.format(var,debt_var)
        plt.savefig(destin, format='png', dpi=1000)
        if show == 1:
            plt.show()
        plt.close()

print("2019 vs 2022 inequality figures done")

"""
Now want to breakdown by private versus public, no pay versus pay, etc.
"""

for yr in [2019]:
    print("Year = {0}:".format(yr))
    print("All student debt")
    frac_nopay = weight_mean(scf[yr]['student_debt_nopay_current'],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_LL_current'],scf[yr]['wgt'])
    print("Fraction not being paid:",frac_nopay)
    for reason in ['forbear','noafford','grace']:
        frac = weight_mean(scf[yr]['student_debt_{0}_current'.format(reason)],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_LL_current'],scf[yr]['wgt'])
        print("   Fraction due to {0}:".format(reason),frac)

    print("Federal student debt:")
    frac_nopay_fed = weight_mean(scf[yr]['student_debt_nopay_fed_current'],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_fed_current'],scf[yr]['wgt'])
    print("Fraction not being paid:",frac_nopay_fed)
    for reason in ['forbear','noafford','grace']:
        frac = weight_mean(scf[yr]['student_debt_{0}_fed_current'.format(reason)],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_fed_current'],scf[yr]['wgt'])
        print("   Fraction due to {0}:".format(reason),frac)

    print("Private student debt:")
    frac_nopay_private = weight_mean(scf[yr]['student_debt_nopay_private_current'],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_private_current'],scf[yr]['wgt'])
    print("Fraction not being paid:",frac_nopay_private)
    for reason in ['forbear','noafford','grace']:
        frac = weight_mean(scf[yr]['student_debt_{0}_private_current'.format(reason)],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_private_current'],scf[yr]['wgt'])
        print("   Fraction due to {0}:".format(reason),frac)

"""
Table summarizing differences between public and private.
"""

for yr in [2019,2022]:
    df = pd.DataFrame(data=0,index=['Total','Federal','Private'],columns=['Total','Forbearance','Cannot afford','Grace period'])
    frac_nopay = weight_mean(scf[yr]['student_debt_nopay_current'],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_LL_current'],scf[yr]['wgt'])
    df.iloc[0,0], i = frac_nopay, 1

    for reason in ['forbear','noafford','grace']:
        frac = weight_mean(scf[yr]['student_debt_{0}_current'.format(reason)],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_LL_current'],scf[yr]['wgt'])
        df.iloc[0,i], i = frac, i+1

    frac_nopay_fed = weight_mean(scf[yr]['student_debt_nopay_fed_current'],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_fed_current'],scf[yr]['wgt'])
    df.iloc[1,0], i = frac_nopay_fed, 1

    for reason in ['forbear','noafford','grace']:
        frac = weight_mean(scf[yr]['student_debt_{0}_fed_current'.format(reason)],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_fed_current'],scf[yr]['wgt'])
        df.iloc[1,i], i = frac, i+1

    frac_nopay_private = weight_mean(scf[yr]['student_debt_nopay_private_current'],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_private_current'],scf[yr]['wgt'])
    df.iloc[2,0], i = frac_nopay_private, 1

    for reason in ['forbear','noafford','grace']:
        frac = weight_mean(scf[yr]['student_debt_{0}_private_current'.format(reason)],scf[yr]['wgt'])/weight_mean(scf[yr]['student_debt_private_current'],scf[yr]['wgt'])
        df.iloc[2,i], i = frac, i+1

    destin = '../main/figures/sd_LL_{0}.tex'.format(yr)
    df_table = df.round(decimals=3)
    with open(destin,'w') as tf:
        df_table_s=df_table.style.format(precision=3)
        tf.write(df_table_s.to_latex(column_format='lcccc'))

"""
Aggregate federal and private student loans
"""

for yr in [2019,2022]:
    print("Year = {0}".format(yr))
    p = scf[yr]['student_debt_private_current']
    f = scf[yr]['student_debt_fed_current']
    print("Private:", weight_agg(p, scf[yr]['wgt'])/10**12, "trillion")
    print("Federal:", weight_agg(f, scf[yr]['wgt'])/10**12, "trillion")
    print("Total (loan-level):", weight_agg(p+f, scf[yr]['wgt'])/10**12, "trillion")
    print("Fraction that is federal:", weight_agg(f, scf[yr]['wgt'])/weight_agg(p+f, scf[yr]['wgt']))
    print("Total (summary):", weight_agg(scf[yr]['student_debt'],scf[yr]['wgt'])/10**12, "trillion")
