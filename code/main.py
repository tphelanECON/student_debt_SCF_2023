"""
Download data and create the figures and tables used in the Commentary.
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

import data_clean
import bulletin
import analysis
