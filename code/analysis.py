"""
This script performs the analysis for the Commentary, using the data constructed in data_clean.py.

Tables:
    * Summary tables giving median and means income and net worth for student debtors, the whole population,
    and families in which the respondent is under the age of 35.

Figures:
    * Average student debt by quintiles, broken down by:
        * income and net worth quintiles.
        * whole population and population of debtors.
    * Incidence of debt by quintiles of income and net worth.
    * Ratio of conditional mean of student debt to conditional median of student debt
    by quintiles of income and net worth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import data_clean

"""
Set dummy to indicate if plots are shown and suppress warning (from https://github.com/twopirllc/pandas-ta/issues/340)
"""

show = 0

"""
Import the data, functions and lists that we need
"""

# Dataframes
scf = data_clean.scf
scf_debtors = data_clean.scf_debtors
scf_young = data_clean.scf_young
scf_young_debtors = data_clean.scf_young_debtors

# Functions
quantile = data_clean.quantile
weight_median = data_clean.weight_median
weight_mean = data_clean.weight_mean
weight_agg = data_clean.weight_agg
weight_median_df = data_clean.weight_median_df
weight_mean_df = data_clean.weight_mean_df
weight_agg_df = data_clean.weight_agg_df

# Plots
c1 = 'crimson'
c2 = 'darkblue'


def colorfader(color1, color2, mix):
    return mpl.colors.to_hex(
        (1 - mix) * np.array(mpl.colors.to_rgb(color1)) + mix * np.array(mpl.colors.to_rgb(color2)))


name_dict = {'income': 'Income', 'networth': 'Net worth'}
xlabelfontsize = 15
ylabelfontsize = 15
titlefontsize = 15
width = 1 / 5

"""
Summary tables (for the "summary statistics" section)
"""

summary_rows = ['Median income', 'Mean income', 'Median net worth', 'Mean net worth']
summary_cols = ['Whole population', 'Student debtors']
for yr in [2019, 2022]:
    for demo_var in ['all', 'young']:
        df = pd.DataFrame(data=0, index=summary_rows, columns=summary_cols)
        if demo_var == 'all':
            df_list = [scf[yr], scf_debtors[yr]]
        else:
            df_list = [scf_young[yr], scf_young_debtors[yr]]
        for i, d in enumerate(df_list):
            df.iloc[0, i] = weight_median_df(d, 'income') / 10 ** 3
            df.iloc[1, i] = weight_mean_df(d, 'income') / 10 ** 3
            df.iloc[2, i] = weight_median_df(d, 'networth') / 10 ** 3
            df.iloc[3, i] = weight_mean_df(d, 'networth') / 10 ** 3

        destin = '../main/figures/summary_{0}_{1}.tex'.format(yr, demo_var)
        df_table = df.round(decimals=1)
        with open(destin, 'w') as tf:
            df_table = df_table.style.format(precision=1)
            tf.write(df_table.to_latex(column_format='lccc'))

"""
Average (mean) student debt by quintiles. Indexed by:
    * choice of quintile (income or networth) 
    * population (whole population or population of student debtors)
    
Therefore 2 x 2 = 4 figures.
"""

for var in ['income', 'networth']:
    for pop_var in ['student debtors', 'all']:
        SD_quintiles = pd.DataFrame(columns=range(1, 6), index=[2019, 2022])
        for yr in [2019, 2022]:
            if pop_var == 'student debtors':
                df_temp = scf_debtors[yr]
            else:
                df_temp = scf[yr]
            f = lambda x: np.average(x, weights=df_temp.loc[x.index, "wgt"])
            SD_quintiles.loc[yr, :] = df_temp.groupby(var + '_cat5')['SD'].agg(f).values
        SD_quintiles = (SD_quintiles / 1000).astype(float).round(1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(5):
            if i == 0:
                ax.bar(i + 1 - width, SD_quintiles.loc[2019, i + 1], 2 * width, color=colorfader(c1, c2, 0),
                       label=2019)
                ax.bar(i + 1 + width, SD_quintiles.loc[2022, i + 1], 2 * width, color=colorfader(c1, c2, 1),
                       label=2022)
            else:
                ax.bar(i + 1 - width, SD_quintiles.loc[2019, i + 1], 2 * width, color=colorfader(c1, c2, 0))
                ax.bar(i + 1 + width, SD_quintiles.loc[2022, i + 1], 2 * width, color=colorfader(c1, c2, 1))
        plt.legend()
        ax.set_xlabel('{0} quintile'.format(name_dict[var]), fontsize=xlabelfontsize)
        ax.set_ylabel('Thousands', fontsize=ylabelfontsize)
        if pop_var == 'all':
            ax.set_title('Average student debt', fontsize=titlefontsize)
        else:
            ax.set_title('Average student debt among student debtors', fontsize=titlefontsize)
        destin = '../main/figures/{0}_ave_{1}.png'.format(var[:3], pop_var[:3])
        plt.savefig(destin, format='png', dpi=1000)
        if show == 1:
            plt.show()
        plt.close()

"""
Percentage of families with student debt by quintiles of income and net worth.
"""

for var in ['income', 'networth']:
    SD_quintiles_frac = pd.DataFrame(columns=range(1, 6), index=[2019, 2022])
    for yr in [2019, 2022]:
        data = scf[yr]
        quintiles = np.array([quantile(data[var], data['wgt'], j / 5) for j in range(6)])
        qct_lists, var_names = [quintiles, [0, 1, np.inf]], [var, 'SD']
        d = [pd.cut(data[var_names[i]], bins=qct_lists[i], labels=range(len(qct_lists[i]) - 1),
                    include_lowest=True, duplicates='drop') for i in range(2)]
        data['pairs'] = list(zip(d[0], d[1]))
        SD_debt = data.groupby(data['pairs'])['SD']
        SD_debt_count = data.groupby(data['pairs'])['wgt'].sum()
        SD_quintiles_frac.loc[yr, :] = [SD_debt_count[(i, 1)] / (SD_debt_count[(i, 0)] + SD_debt_count[(i, 1)]) for i in
                                        range(5)]
    SD_quintiles_pct = 100 * SD_quintiles_frac
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(5):
        if i == 0:
            ax.bar(i + 1 - width, SD_quintiles_pct.loc[2019, i + 1], 2 * width, color=colorfader(c1, c2, 0), label=2019)
            ax.bar(i + 1 + width, SD_quintiles_pct.loc[2022, i + 1], 2 * width, color=colorfader(c1, c2, 1), label=2022)
        else:
            ax.bar(i + 1 - width, SD_quintiles_pct.loc[2019, i + 1], 2 * width, color=colorfader(c1, c2, 0))
            ax.bar(i + 1 + width, SD_quintiles_pct.loc[2022, i + 1], 2 * width, color=colorfader(c1, c2, 1))
    plt.legend()
    ax.set_xlabel('{0} quintile'.format(name_dict[var]), fontsize=xlabelfontsize)
    ax.set_ylabel('Percent', fontsize=ylabelfontsize)
    ax.set_title('Percentage of families with student debt', fontsize=titlefontsize)
    ax.set_ylim([0, 50])
    destin = '../main/figures/{0}_pct.png'.format(var[:3])
    plt.savefig(destin, format='png', dpi=1000)
    if show == 1:
        plt.show()
    plt.close()

"""
Ratio of conditional mean to conditional median of student debt. 
"""

for var in ['income', 'networth']:
    SD_quintiles_rat = pd.DataFrame(columns=range(1, 6), index=[2019, 2022])
    for yr in [2019, 2022]:
        data = scf_debtors[yr]
        f = lambda x: np.average(x, weights=data.loc[x.index, "wgt"])
        g = lambda x: weight_median(x, weights=data.loc[x.index, "wgt"])
        gb = data.groupby(var + '_cat{0}'.format(5))['SD'].agg(f).values
        gb_med = data.groupby(var + '_cat{0}'.format(5))['SD'].agg(g).values
        SD_quintiles_rat.loc[yr, :] = gb / gb_med

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(5):
        if i == 0:
            ax.bar(i + 1 - width, SD_quintiles_rat.loc[2019, i + 1], 2 * width, color=colorfader(c1, c2, 0), label=2019)
            ax.bar(i + 1 + width, SD_quintiles_rat.loc[2022, i + 1], 2 * width, color=colorfader(c1, c2, 1), label=2022)
        else:
            ax.bar(i + 1 - width, SD_quintiles_rat.loc[2019, i + 1], 2 * width, color=colorfader(c1, c2, 0))
            ax.bar(i + 1 + width, SD_quintiles_rat.loc[2022, i + 1], 2 * width, color=colorfader(c1, c2, 1))
    plt.legend()
    ax.set_xlabel('{0} quintile'.format(name_dict[var]), fontsize=xlabelfontsize)
    ax.set_ylabel('Ratio', fontsize=ylabelfontsize)
    ax.set_ylim([0, 2.2])
    ax.set_title('Ratio of mean-to-median student debt', fontsize=titlefontsize)
    destin = '../main/figures/{0}_MM.png'.format(var[:3])
    plt.savefig(destin, format='png', dpi=1000)
    if show == 1:
        plt.show()
    plt.close()
