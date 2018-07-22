import pandas as pd
import requests
import json
import csv
import time
import glob
import os
import math
import re
import string
import itertools
import inspect
from calendar import isleap

#import seaborn as sns
import matplotlib.font_manager as fm
from dateutil.parser import parse
from datetime import datetime, timedelta, date
from pandas.tseries.holiday import get_calendar
from matplotlib import *
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm #color map
import matplotlib.gridspec as gridspec
from scipy.stats.kde import gaussian_kde
from scipy.stats.stats import pearsonr
from pandas.tseries.offsets import CustomBusinessDay
from numpy import linspace
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import getpass
from os.path import expanduser

from read_predictions import readRuns, readInputs
from read_measurements import readGas, readSTM, readSubmetering

def main():
    BuildingList = ['05_MaletPlaceEngineering', '01_CentralHouse', '02_BuroHappold_17',
                    '03_BuroHappold_71']  # Location of DATA
    BuildingHardDisk = ['05_MaletPlaceEngineering_Project', '01_CentralHouse_Project', '02_BuroHappold_17',
                        '03_BuroHappold_71']
    DataFilePaths = ['MPEB', 'CentralHouse', '17', '71']  # Location of STM data and file naming
    BuildingLabels = ['MPEB', 'CH', 'Office 17', 'Office 71']
    BuildingAbbreviations = ['MPEB', 'CH', '17', '71', 'Nothing']
    FloorAreas = [9579, 5876, 1924, 1691]

    building_num = 1  # 0 = MPEB, 1 = CH, 2 = 17, 3 = 71
    base_case = False  # only show a single run from the basecase, or multiple runs (will change both import and plotting)
    simplification = True  # compare simplifications to base_case and compare_models is True
    compare_models = True  # if True, compare calibrated or simplification models with basecase
    calibrated_case = False  # if both base_case and calibrated_case are true and compare_basecase_to_calibration is False, it will show data only on claibrated ccase..
    parallel_simulation = False
    compare_basecase_to_calibration = False
    loadshape_benchmarking = False
    compare_weather = False  # compare weatherfiles for CH and MPEB, compare models need to be true as well

    NO_ITERATIONS = 20
    time_step = 'month'  # 'year', 'month', 'day', 'hour' # this is for loading and plotting the predicted against measured data.
    end_uses = False
    include_weekdays = False  # to include weekdays in the targets/runs for both surrogate model creation and/or calibration, this is for surrogatemodel.py
    write_data = False
    for_sensitivity = True

    building_abr = BuildingAbbreviations[building_num]
    datafile = DataFilePaths[building_num]
    building = BuildingList[building_num]
    building_harddisk = BuildingHardDisk[building_num]
    building_label = BuildingLabels[building_num]
    floor_area = FloorAreas[building_num]

    DataPath_model_real = start_path + 'OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/'
    DataPath = start_path+'OneDrive - BuroHappold/01 - EngD/07 - UCL Study/'
    DataPathSTM = start_path+'OneDrive - BuroHappold/01 - EngD/07 - UCL Study/MonitoringStudyUCL/'

    if building_num in {0, 1}:  # Does it have short term monitoring?
        df_stm = readSTM(DataPathSTM, building, building_num, write_data, datafile,
                         floor_area)  # read short term monitoring
    else:
        df_stm = pd.DataFrame()

    if building_num in {1}:  # does it have separate gas use?
        df_gas = readGas(DataPath, building, building_num, write_data, datafile,
                         floor_area)  # read gas data for Central House
    else:
        df_gas = pd.DataFrame()

    df, df_mains, df_LVL1, df_floorsLP, df_mech, df_stm, df_realweather = readSubmetering(DataPath, building,
                                                                                          building_num, building_abr,
                                                                                          write_data, datafile, df_stm,
                                                                                          floor_area)

    print(df_mains.head())

def assign_colors(cols):
    """
    assign colors for a list of names
    this is to ensure end-uses have the same colours!

    to call when cols are end-uses.

    also change col names
    """

    colors = []
    assigned_colors = {'Equipment': '#1f77b4',
                       'Power': '#1f77b4',
                       'Lights': '#ff7f0e',
                       'Server': '#B675B6',
                       'Servers': '#B675B6',
                       'Gas': '#8c564b',
                       'Heating': '#d62728',
                       'DistrictHeating': '#d62728',
                       #'Canteen': '#9467bd',
                       'Cooling': '#aec7e8',
                       'Chillers': '#aec7e8',
                       'Fans': '#ffbb78',
                       'Pumps': '#98df8a',
                       'WaterSystems': '#c49c94',
                       'Print': '#ff9896',
                       'Canteen': '#c5b0d5',
                       'Lift': '#1f77b4',
                       'Systems': '#669966',
                       'Workshops': '#2ca02c',
                       'L&P': '#006666',
                       'Predicted':'#1f77b4',
                       'Measured':'#ff7f0e'}

    unassigned_colors = ['#9467bd', '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896']

    x = 0
    for col in cols:
        if col in assigned_colors:
            colors.append(assigned_colors[col])
        else:
            colors.append(unassigned_colors[x])
            x += 1

    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


    return colors




if __name__ == '__main__':
    def start__main__():
        print('start')
    start__main__()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    UserName = getpass.getuser()
    if UserName == 'cvdronke':
        start_path = 'C:/Users/' + UserName + '/'
    else:
        start_path = 'D:/'

    main()
