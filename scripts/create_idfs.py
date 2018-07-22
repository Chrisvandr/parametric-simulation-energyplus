__author__ = 'cvdronke'

import sys
import os
import re
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from pyDOE import doe_lhs
import scipy.stats as stats
from collections import defaultdict
import random
import pandas as pd
import eppy
from time import gmtime, strftime
from eppy import modeleditor
from eppy.modeleditor import IDF
import getpass

def main():
    # pathnameto_eppy = 'c:/eppy'
    pathnameto_eppy = '../'
    sys.path.append(pathnameto_eppy)
    UserName = getpass.getuser()

    # own scripts
    sys.path.append('D:\OneDrive - BuroHappold/01 - EngD/07 - UCL Study/UCLDataScripts')
    from PlotSubmetering import lineno

    if sys.platform == 'win32':
        rootdir = os.path.dirname(os.path.abspath(__file__))
        harddrive_idfs = "D:/OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic"
        idfdir = os.path.dirname(os.path.abspath(__file__)) + "\IDFs"
        epdir = "C:/EnergyPlusV8-6-0"
    else:
        if print_statements is True: print("rootdir not found")

    BuildingAbbreviations = ['MPEB', 'CH', '17', '71', 'Nothing']
    BuildingHardDisk = ['05_MaletPlaceEngineering_Project', '01_CentralHouse_Project', '02_BuroHappold_17', '03_BuroHappold_71']

    """ WHICH BUILDING AND NO OF SAMPLES """
    building_num = 1 #0 = MPEB, 1 = CH, 2 = 17, 3 = 71
    n_samples =  1 # how many idfs need to be created
    from_samples = 0 #from what number to start creating files (in case i need to stop creation)

    """ TYPE OF SIMULATION """
    base_case = True #todo run basecase straight away with eppy http://pythonhosted.org/eppy/runningeplus.html
    simplifications = True #and base_case is True todo if simplification is true, it will use the basecase and any simplification applied.
    parallel_simulation = False
    calibrated_case = False

    """ FOR CALIBRATED CASE """
    time_step = 'month'
    end_uses = True
    hourly = True

    """ ADDITIONAL SETTINGS """
    add_variables = False # these are additional variables written to .eso (which are only done when basecase is true, but can be turned of here if not necessary.
    remove_sql = True # when doing MPEB, sql is too big (1gb+)
    save_idfs = True # to create new idfs files or not (in case just to check input generation)
    print_statements = False
    vertical_scaling = False

    iddfile = "C:/EnergyPlusV8-6-0/Energy+.idd"
    IDF.setiddname(iddfile)
    building_harddisk = BuildingHardDisk[building_num]
    building_abr = BuildingAbbreviations[building_num]

    if building_abr == 'MPEB':
        no_variables = 350
        #todo factor by max of monthly l&p consumption and then -.2 so to allow variation...
        #                           Jan   Feb   Mar   Apr   May   jun,  jul,  aug,  sept, oct,  nov,  dec
        seasonal_occ_factor_week = [0.72, 0.66, 0.64, 0.58, 0.74, 0.74, 0.8, 0.77, 0.61, 0.68, 0.66, 0.63] # TODO!!! This is from January to December!!!!
        seasonal_occ_factor_weekend =  [0.72, .66, 0.64, 0.58, 0.74, 0.74, .8, .77, .61, .68, .66, .63]  # TODO!!! This is from January to December!!!!
        overtime_multiplier_equip = 85
        overtime_multiplier_light = 65
        multiplier_variation = 10
        run_periods = [[9, 1, 12, 31, '2016', 'Thursday', 'RunPeriod1'], [1, 1, 8, 31, '2017','Sunday', 'RunPeriod2']]  # first month, first day, last month, last day, 'Start Year', Start week, 'Name'

        building_name = 'MaletPlace'
        if base_case | calibrated_case | simplifications is True:
            save_dir = harddrive_idfs + "/05_MaletPlaceEngineering_Project/BaseCase/"
            no_simplifications = 11
        if parallel_simulation is True:
            save_dir = harddrive_idfs + "/05_MaletPlaceEngineering_Project/IDFs/"

    elif building_abr == 'CH':
        no_variables = 300
        #                           Jan   Feb  Mar   Apr   May   jun,  jul,  aug,  sept, oct,  nov,  dec
        seasonal_occ_factor_week = [0.63, .67, 0.63, 0.66, 0.60, 0.53, 0.50, 0.48, 0.45, 0.44, 0.63, 0.61]
        seasonal_occ_factor_weekend = [0.63, .67, 0.63, 0.66, 0.60, 0.53, 0.50, 0.48, 0.45, 0.44, 0.63, 0.61]
        overtime_multiplier_equip = 65
        overtime_multiplier_light = 20
        multiplier_variation = 10
        run_periods = [[9, 1, 12, 31, '2016', 'Thursday', 'RunPeriod1'], [1, 1, 8, 31, '2017','Sunday', 'RunPeriod2']]  # first month, first day, last month, last day, 'Start Year', Start week, 'Name'

        building_name = 'CentralHouse_222'  # 'MalletPlace', 'CentralHouse_222' # building_name
        if base_case | calibrated_case | simplifications is True:
            save_dir = harddrive_idfs + "/01_CentralHouse_Project/BaseCase/"
            no_simplifications = 9
        if parallel_simulation is True:
            save_dir = harddrive_idfs + "/01_CentralHouse_Project/IDFs/"

    if calibrated_case is True:
        DataPath_model_real = 'D:\OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/'
        hours = 'hourly' if hourly == True else ''
        print(DataPath_model_real + 'best_individual' + time_step + str(end_uses) + hours + '.csv')
        df_calibrated = pd.read_csv(DataPath_model_real + 'best_individual' + time_step + str(end_uses) + hours + '.csv', index_col=0, header=0)

    if not os.path.exists(save_dir):  # check if folder exists
        os.makedirs(save_dir)  # create new folder

    print("{}".format(rootdir) + "/" + building_name + ".idf")
    idf1 = IDF("{}".format(rootdir) + "/" + building_name + ".idf")
    if print_statements is True: print(idf1)

    lhd = doe_lhs.lhs(no_variables, samples=n_samples)
    print(lhd.shape)

    run_lhs(idf1, lhd, building_name, building_abr, base_case, simplifications, no_simplifications, remove_sql, add_variables, run_periods, n_samples, from_samples, save_idfs, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation, seasonal_occ_factor_week, seasonal_occ_factor_weekend)

class DLine(): #use .dline[x] to access different columns in the csv files
    def __init__(self, dline):
        self.dline = dline
        self.name = dline[0]

class Material():
    def __init__(self, dline):  # initialised from line of csv file
        self.dline = dline
        self.name = dline[0]
        self.ep_type = dline[1]
        self.roughness = dline[2]
        self.conductivity = dline[3]
        self.density = dline[4]
        self.specific_heat = dline[5]
        self.thermal_abs = dline[6]
        self.solar_abs = dline[7]
        self.visible_abs = dline[8]
        self.thickness_abs = dline[58]
        self.thermal_res = dline[59]
        self.shgc = dline[60]
        self.ufactor = dline[61]
        ##heat and moisture properties
        self.porosity = dline[9]
        self.init_water = dline[10]
        self.n_iso = dline[11]
        self.sorb_iso = {}
        self.n_suc = dline[14]
        self.suc = {}
        self.n_red = dline[17]
        self.red = {}
        self.n_mu = dline[20]
        self.mu = {}
        self.n_therm = dline[23]
        self.therm = {}
        self.vis_ref = dline[44]

        ##solar properties
        if len(dline) > 31:
            if dline[31] != '':
                self.g_sol_trans = float(dline[31])
                self.g_F_sol_ref = float(dline[32])
                self.g_B_sol_ref = float(dline[33])
                self.g_vis_trans = float(dline[34])
                self.g_F_vis_ref = float(dline[35])
                self.g_B_vis_ref = float(dline[36])
                self.g_IR_trans = float(dline[37])
                self.g_F_IR_em = float(dline[38])
                self.g_B_IR_em = float(dline[39])

        if len(dline) > 39:
            self.ep_special = dline[40]
            if dline[41] != '':
                self.sol_trans = float(dline[41])
                self.sol_ref = float(dline[42])
                self.vis_trans = float(dline[43])
                # self.therm_hem_em = float(dline[45])
                # self.therm_trans = float(dline[46])
                # self.shade_to_glass_dist = float(dline[47])
                # self.top_opening_mult = float(dline[48])
                # self.bottom_opening_mult = float(dline[49])
                # self.left_opening_mult = float(dline[50])
                # self.right_opening_mult = float(dline[51])
                # self.air_perm = float(dline[52])

        if len(dline) > 53:
            self.ep_special = dline[40]
            if dline[53] != '':
                self.orient = dline[53]
                self.width = float(dline[54])
                self.separation = float(dline[55])
                self.angle = float(dline[56])
                self.blind_to_glass_dist = float(dline[57])

def read_sheet(rootdir, building_abr, base_case, simplifications, run_no,  fname=None):
    sheet = []

    if simplifications:
        file_ = "simplification_"+str(run_no)
    else:
        file_ = "base_case"

    if fname != None:
        if building_abr == 'CH':
            csvfile = "{}".format(rootdir)+"/data_files/CentralHouse/"+file_+"/{}".format(fname)
        elif building_abr == 'MPEB':
            csvfile = "{}".format(rootdir)+"/data_files/MalletPlace/"+file_+"/{}".format(fname)
        elif building_abr == '71':
            csvfile = "{}".format(rootdir)+"/data_files/BH71/"+file_+"/{}".format(fname)

        if print_statements is True: print(csvfile)
        with open(csvfile, 'r') as inf:
            for row in inf:
                row = row.rstrip()
                datas = row.split(',')
                sheet.append(datas)
    return sheet

def unpick_equipments(rootdir, building_abr, base_case, simplifications, run_no):
    sheet = read_sheet(rootdir, building_abr, base_case, simplifications, run_no, fname="equipment_props.csv")

    for nn, lline in enumerate(sheet):
        if len(lline) > 0 and 'Appliance' in lline[0]:
            d_start = nn + 1
            break

    equips = {}
    equip = None

    for dline in sheet[d_start:]:
        if len(dline) > 1:
            if dline[0] != '' and dline[1] != '':
                equip = DLine(dline)
            equips[dline[0]] = equip
    if print_statements is True: print(lineno(), [v for v in equips.keys()])

    return equips
def unpick_schedules(rootdir, building_abr, base_case, simplifications, run_no):
    sheet = read_sheet(rootdir, building_abr, base_case, simplifications, run_no,  fname="house_scheds.csv")

    for nn, lline in enumerate(sheet):
        if len(lline) > 0 and 'Name' in lline[0]:
            d_start = nn + 1
            break

    sched_groups = {}
    hgroup = None
    for dline in sheet[d_start:]:
        if len(dline) > 1:
            if dline[0] != '':

                #sname = dline[0]
                sched = DLine(dline)
                sched_groups[sched.name] = sched

                # if sname not in list(sched_groups.keys()):
                #     hgroup = sched.HGroup(dline)
                #     sched_groups[sname] = hgroup
                # hgroup = sched_groups[sname]
                #
                # # Loop her over zones so that can reduce the size of the house_sched csv
                # zones = dline[1].split(":")
                # if (len(zones) > 1):
                #     for zone in zones:
                #         zonesched = sched.ZoneSched(dline, zone)
                #         hgroup.zones_scheds.append(zonesched)
                # else:
                #     zonesched = sched.ZoneSched(dline, dline[1])
                #     hgroup.zones_scheds.append(zonesched)

    return sched_groups
def unpick_materials(rootdir, building_abr, base_case, simplifications, run_no):
    # open material properties file and puts everything into sheet
    sheet = read_sheet(rootdir, building_abr, base_case, simplifications,run_no,   fname="mat_props.csv")

    # find the material properties from csv file
    for nn, lline in enumerate(sheet):
        if len(lline) > 0 and 'Name' in lline[0]:
            d_start = nn + 1
            break

    mats = {}
    mat = None
    for dline in sheet[d_start:]:
        if len(dline) > 1:
            ##will have to read some stuff here
            if dline[0] != '' and dline[1] != '':
                mat = Material(dline)
                mats[mat.name] = mat
    return mats

def replace_schedules(run_file, lhd, input_values, input_names, occ_schedules, light_schedules, equip_schedules, var_num, run_no, building_abr, base_case, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation, seasonal_occ_factor_week, seasonal_occ_factor_weekend):
    # IN ORDER
    # 1. ScheduleTypeLimits
    # 2. Single
    # 3. NoChange
    # 4. TempScheds
    # 5. WaterHeaters
    # 6. Occupancy / Equip / Lights

    scheds = unpick_schedules(rootdir, building_abr, base_case, simplifications, run_no)
    scheddict = defaultdict(list)

    timeline = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + ":" + ("0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(30, 1470, 30)]
    timeline = timeline + timeline  # one for weekday and weekendday
    # if print_statements is True: print timeline

    #Randomly pick a schedule for zone set-point temperatures
    if building_abr in {'CH'}: #'MPEB'
        if parallel_simulation:
            #rand = 6
            #todo make this more efficient....
            if building_abr == 'CH':
                setpoints_strategy = [3, 4, 5, 10, 11, 12, 18, 19, 20, 26, 27, 28, 34, 35, 36, 37, 38]
                rand = random.choice(setpoints_strategy)
            else:
                rand = random.choice(range(36))

            if rand == 0:
                HeatingSched = scheds['h_sched_225']
                CoolingSched = scheds['c_sched_225']
                hp, db = 22, 0
            elif rand == 1:
                HeatingSched = scheds['h_sched_225']
                CoolingSched = scheds['c_sched_225']
                hp, db = 22.5, 0
            elif rand == 2:
                HeatingSched = scheds['h_sched_230']
                CoolingSched = scheds['c_sched_230']
                hp, db = 23, 0
            elif rand == 3:
                HeatingSched = scheds['h_sched_235']
                CoolingSched = scheds['c_sched_235']
                hp, db = 23.5, 0
            elif rand == 4:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_240']
                hp, db = 24, 0
            elif rand == 5:
                HeatingSched = scheds['h_sched_245']
                CoolingSched = scheds['c_sched_245']
                hp, db = 24.5, 0

            elif rand == 6:
                HeatingSched = scheds['h_sched_215']
                CoolingSched = scheds['c_sched_220']
                hp, db = 21.5, 0.5

            elif rand == 7:
                HeatingSched = scheds['h_sched_220']
                CoolingSched = scheds['c_sched_225']
                hp, db = 22, 0.5

            elif rand == 8:
                HeatingSched = scheds['h_sched_225']
                CoolingSched = scheds['c_sched_230']
                hp, db = 22.5, 0.5

            elif rand == 9:
                HeatingSched = scheds['h_sched_230']
                CoolingSched = scheds['c_sched_235']
                hp, db = 23, 0.5

            elif rand == 10:
                HeatingSched = scheds['h_sched_235']
                CoolingSched = scheds['c_sched_240']
                hp, db = 23.5, 0.5

            elif rand == 11:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_245']
                hp, db = 24, 0.5

            elif rand == 12:
                HeatingSched = scheds['h_sched_245']
                CoolingSched = scheds['c_sched_250']
                hp, db = 24.5, 0.5

            elif rand == 13:
                HeatingSched = scheds['h_sched_210']
                CoolingSched = scheds['c_sched_220']
                hp, db = 21, 1

            elif rand == 14:
                HeatingSched = scheds['h_sched_215']
                CoolingSched = scheds['c_sched_225']
                hp, db = 21.5, 1

            elif rand == 15:
                HeatingSched = scheds['h_sched_220']
                CoolingSched = scheds['c_sched_230']
                hp, db = 22, 1

            elif rand == 16:
                HeatingSched = scheds['h_sched_225']
                CoolingSched = scheds['c_sched_235']
                hp, db = 22.5, 1

            elif rand == 17:
                HeatingSched = scheds['h_sched_230']
                CoolingSched = scheds['c_sched_240']
                hp, db = 23, 1

            elif rand == 18:
                HeatingSched = scheds['h_sched_235']
                CoolingSched = scheds['c_sched_245']
                hp, db = 23.5, 1

            elif rand == 19:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_250']
                hp, db = 24, 1

            elif rand == 20:
                HeatingSched = scheds['h_sched_245']
                CoolingSched = scheds['c_sched_255']
                hp, db = 24.5, 1

            elif rand == 21:
                HeatingSched = scheds['h_sched_210']
                CoolingSched = scheds['c_sched_225']
                hp, db = 21, 1.5

            elif rand == 22:
                HeatingSched = scheds['h_sched_215']
                CoolingSched = scheds['c_sched_230']
                hp, db = 21.5, 1.5

            elif rand == 23:
                HeatingSched = scheds['h_sched_220']
                CoolingSched = scheds['c_sched_235']
                hp, db = 22, 1.5

            elif rand == 24:
                HeatingSched = scheds['h_sched_225']
                CoolingSched = scheds['c_sched_240']
                hp, db = 22.5, 1.5

            elif rand == 25:
                HeatingSched = scheds['h_sched_230']
                CoolingSched = scheds['c_sched_245']
                hp, db = 23, 1.5

            elif rand == 26:
                HeatingSched = scheds['h_sched_235']
                CoolingSched = scheds['c_sched_250']
                hp, db = 23.5, 1.5

            elif rand == 27:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_255']
                hp, db = 24, 1.5

            elif rand == 28:
                HeatingSched = scheds['h_sched_245']
                CoolingSched = scheds['c_sched_260']
                hp, db = 24.5, 1.5

            elif rand == 29:
                HeatingSched = scheds['h_sched_210']
                CoolingSched = scheds['c_sched_230']
                hp, db = 21, 2

            elif rand == 30:
                HeatingSched = scheds['h_sched_215']
                CoolingSched = scheds['c_sched_235']
                hp, db = 21.5, 2

            elif rand == 31:
                HeatingSched = scheds['h_sched_220']
                CoolingSched = scheds['c_sched_240']
                hp, db = 22, 2

            elif rand == 32:
                HeatingSched = scheds['h_sched_225']
                CoolingSched = scheds['c_sched_245']
                hp, db = 22.5, 2

            elif rand == 33:
                HeatingSched = scheds['h_sched_230']
                CoolingSched = scheds['c_sched_250']
                hp, db = 23, 2

            elif rand == 34:
                HeatingSched = scheds['h_sched_235']
                CoolingSched = scheds['c_sched_255']
                hp, db = 23.5, 2

            elif rand == 35:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_260']
                hp, db = 24, 2

            elif rand == 36:
                HeatingSched = scheds['h_sched_245']
                CoolingSched = scheds['c_sched_265']
                hp, db = 24.5, 2

            elif rand == 37:
                HeatingSched = scheds['h_sched_245']
                CoolingSched = scheds['c_sched_270']
                hp, db = 24.5, 2.5

            elif rand == 38:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_270']
                hp, db = 24, 3

            var_num += 1

        elif base_case and not simplifications:
            HeatingSched = scheds['h_sched_240']
            CoolingSched = scheds['c_sched_240']
            hp, db = 24, 0

        elif simplifications:
            if scheds['Office_Heating'].dline[5] in {'NCM'}:
                HeatingSched = scheds['Office_Heating']
                CoolingSched = scheds['Office_Cooling']
                hp, db = 24, 0
            else:
                HeatingSched = scheds['h_sched_240']
                CoolingSched = scheds['c_sched_240']
                hp, db = 24, 0

            print('heatingsched', HeatingSched.dline[8:])

        if calibrated_case:
            db = df_calibrated.iloc[0]['DeadBand']
            hp = df_calibrated.iloc[0]['Office_HeatingSP']

            HeatingSched = scheds['h_sched_'+str(hp).replace('.','')]
            CoolingSched = scheds['c_sched_'+str(hp+db).replace('.','')]

        if print_statements is True: print(rand, hp, db)

        input_names.append('DeadBand')
        input_values.append(db)

        input_names.append('Office_HeatingSP')
        input_values.append(hp)

    #as inf, open(run_file[:-4]+"s.idf", 'w') as outf
    if print_statements is True: print('run_file', run_file[:-4])
    with open(run_file[:-4]+".idf", 'a') as inf:
        # go through the schedule.csv and assign a sched to a list

        # STOCHASTIC HEATING AND COOLING TEMPERATURE SCHEDULES
        if base_case | parallel_simulation | simplifications:
            if building_abr == 'CH':
                office_c_scheds = ['LectureTheatre_Cooling', 'Meeting_Cooling', 'Office_Cooling', 'PrintRoom_Cooling',
                                   'Circulation_Cooling', 'Library_Cooling', 'Kitchen_Cooling', 'ComputerCluster_Cooling', 'Reception_Cooling']

                # elif building_abr == 'MPEB':
                #     office_c_scheds = ['Office_Cooling']

                temp_sched = [CoolingSched, HeatingSched]
                print('heating', HeatingSched.dline[8:])
                for sched in temp_sched:
                    hours = copy.copy(sched.dline[8:])
                    if '' in hours:
                        linx = hours.index('')
                        hours = np.array(copy.copy(hours[0:linx]))
                    else:
                        hours = np.array(copy.copy(hours))

                    if sched == CoolingSched:
                        for y in office_c_scheds:
                            SchedProperties = []
                            SchedProperties = ['Schedule:Compact', y, CoolingSched.dline[3]]
                            SchedProperties.append('Through: 12/31')
                            SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                            for i, v in enumerate(hours):
                                if i == 48:
                                    SchedProperties.append('For: Weekends Holiday')
                                SchedProperties.append('Until: ' + timeline[i])
                                SchedProperties.append(v)

                            scheddict[y].append(SchedProperties)
                            #if print_statements is True: print CoolingSched.dline[40], SchedProperties

                    elif sched == HeatingSched:
                        for y in office_c_scheds:
                            SchedProperties = []
                            SchedProperties = ['Schedule:Compact', y[:-8] + "_Heating", HeatingSched.dline[3]]
                            SchedProperties.append('Through: 12/31')
                            SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                            for i, v in enumerate(hours):
                                if i == 48:
                                    SchedProperties.append('For: Weekends Holiday')
                                SchedProperties.append('Until: ' + timeline[i])
                                SchedProperties.append(v)

                            scheddict[y[:-8] + "_Heating"].append(SchedProperties)
                            #if print_statements is True: print HeatingSched.dline[40], SchedProperties

        #HOT WATER HEATER SCHEDULES
        # todo it should automatically identify which schedules are the water heaters instead of manually picking up the names everytime.
        #todo should be easy enough to just select 'purpose' on dline 2
        if building_abr == 'CH':
            hwheaters = ["HWSchedule_Cleaner", "HWSchedule_Kitchenettes", "HWSchedule_Showers", "HWSchedule_Toilets"]
            hwheaters = []
        elif building_abr == 'MPEB':
            hwheaters = ["CWS_Sched", "HWS_Labs_Sched"]
        elif building_abr == '71':
            hwheaters = ["EWH_Fraction", "EWH_Kitchen_Fraction", "EWH_Shower_Fraction"]

        for heater in hwheaters:
            heater_profile = []
            hprofile = scheds[heater].dline[8:8 + 48+48]
            for i, v in enumerate(hprofile):
                mu = float(hprofile[i])

                if base_case is True:
                    sigma = 0
                    std_rand = 0

                elif parallel_simulation:
                    sigma = mu * 20 / 100
                    std_rand = lhd[run_no, var_num]

                elif calibrated_case:
                    std_rand = df_calibrated.iloc[0][heater + '_StdDev']

                lower, upper = mu - (2 * sigma), 1

                if sigma != 0:
                    hw_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(std_rand)
                else:
                    hw_sched = mu
                heater_profile.append(hw_sched)
            var_num += 1

            l = len(heater_profile)
            offset = [0, 1, 2]
            if base_case:
                offset_heater = 0
            elif parallel_simulation:
                #offset_heater = random.choice(offset)
                offset_heater = 0
            elif calibrated_case:
                #forgot to make it a discrete variable..
                if building_abr in {'CH', 'MPEB'}:
                    offset_heater = 0 #turned off offsets for heater
                else:
                    offset_heater = int(df_calibrated.iloc[0][heater+'_Offset'])

            heater_profile_off = []
            for x, val in enumerate(heater_profile):
                if x > 0 and x < (l - offset_heater):
                    if val < heater_profile[x-offset_heater]:
                        heater_profile_off.append(heater_profile[x-offset_heater])
                        continue
                    if val < heater_profile[x+offset_heater]:
                        heater_profile_off.append(heater_profile[x+offset_heater])
                        continue
                    else:
                        heater_profile_off.append(heater_profile[x])
                else:
                    heater_profile_off.append(heater_profile[x])

            var_num += 1

            if print_statements is True: print(lineno(), var_num, heater, offset_heater)
            input_names.append(heater+'_StdDev')
            input_values.append(std_rand)
            input_names.append(heater+'_Offset')
            input_values.append(offset_heater)

            SchedProperties = []
            if base_case is True:
                heater_profile_off = heater_profile # use standard profile?

            for sched, sname in enumerate(heater_profile_off):
                SchedProperties = ['Schedule:Compact', heater, scheds[heater].dline[3]]
                SchedProperties.append('Through: 12/31')
                SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                for i, v in enumerate(heater_profile_off):
                    if i == 48:
                        SchedProperties.append('For: Weekends Holiday')
                    SchedProperties.append('Until: ' + timeline[i])
                    SchedProperties.append(v)
            scheddict[heater].append(SchedProperties)

            def plot_schedules():
                heater_profile = pd.DataFrame(heater_profile_off, index=timeline, columns=[heater])
                ax = heater_profile.plot(drawstyle='steps')
                ax.set_ylim(0)
                plt.show()
            #plot_schedules()

        # FOR OCCUPANCY PROFILES TO INSERT STANDARD DEVIATION FROM THE SCHEDULES
        if building_abr in {'CH', 'MPEB'}:
            #overtime percentage
            if calibrated_case:
                overtime_multiplier_equip = df_calibrated.iloc[0]['LightOvertimeMultiplier']
            if parallel_simulation:
                sigma = overtime_multiplier_equip*(multiplier_variation/100) # * 20% variation
                lower, upper = overtime_multiplier_equip - (1 * sigma), overtime_multiplier_equip + (1 * sigma)
                overtime_multiplier_equip = stats.truncnorm((lower - overtime_multiplier_equip) / sigma, (upper - overtime_multiplier_equip) / sigma, loc=overtime_multiplier_equip, scale=sigma).ppf(lhd[run_no, var_num])
                var_num += 1

            input_names.append('EquipmentOvertimeMultiplier')
            input_values.append(overtime_multiplier_equip)

            if calibrated_case:
                overtime_multiplier_light = df_calibrated.iloc[0]['EquipmentOvertimeMultiplier']

            if parallel_simulation:
                sigma = overtime_multiplier_light*(multiplier_variation/100)
                lower, upper = overtime_multiplier_light - (1 * sigma), overtime_multiplier_light + (1 * sigma)
                overtime_multiplier_light = stats.truncnorm((lower - overtime_multiplier_light) / sigma, (upper - overtime_multiplier_light) / sigma, loc=overtime_multiplier_light, scale=sigma).ppf(lhd[run_no, var_num])
                var_num += 1

            input_names.append('LightOvertimeMultiplier')
            input_values.append(overtime_multiplier_light)

            occ_week = []
            equip_week = []
            light_week = []
            occ_profile_week = []
            week = ['Weekday', 'Weekend']

            if vertical_scaling is False:
                for day in week:
                    if day == 'Weekday':
                        occ_profile = scheds["WifiSumPlusStd"].dline[8:8 + 48]
                        occ_profile = [float(i) for i in occ_profile]
                    elif day == 'Weekend':
                        occ_profile = scheds["WifiSumPlusStd"].dline[8 + 48:8 + 48 + 48]
                        occ_profile = [float(i) for i in occ_profile]
                    if print_statements is True: print(lineno(), day, len(occ_profile), occ_profile)

                    """OFFSET"""
                    if building_abr == 'CH':
                        offset = [0, 1, 2, 3, 4]
                    else:
                        offset = [-1,0,1,2,3]
                    if base_case is True:
                        if building_abr =='CH':
                            offset_LP = 2
                        else:
                            offset_LP = 0
                    elif parallel_simulation:
                        offset_LP = random.choice(offset)  # use random choice instead of LHS generator
                        #offset_LP = 0
                        var_num += 1
                    elif calibrated_case:
                        offset_LP = int(df_calibrated.iloc[0][day + 'LandPsched_Offset'])

                    input_names.append(day + 'LandPsched_Offset')
                    input_values.append(offset_LP)

                    occ_profile_offset = []
                    for x, val in enumerate(occ_profile):
                        if offset_LP > 0: # positive offset
                            if x < (len(occ_profile) - abs(offset_LP)) and x > offset_LP:
                                if val < occ_profile[x - offset_LP]: #right-side bell curve
                                    occ_profile_offset.append(occ_profile[x - offset_LP])
                                elif val < occ_profile[x + offset_LP]: #left-side bell curve
                                    occ_profile_offset.append(occ_profile[x + offset_LP])
                                else:
                                    occ_profile_offset.append(occ_profile[x])
                            else:
                                occ_profile_offset.append(occ_profile[x])

                        elif offset_LP < 0 : # with negative offset
                            if x > abs(offset_LP) and x < (len(occ_profile) + offset_LP):
                                if val > occ_profile[x + offset_LP]:  # right-side bell curve
                                    occ_profile_offset.append(occ_profile[x + offset_LP])
                                elif val > occ_profile[x - offset_LP]:  # left-side bell curve
                                    occ_profile_offset.append(occ_profile[x - offset_LP])
                                else:
                                    occ_profile_offset.append(occ_profile[x])
                            else:
                                occ_profile_offset.append(occ_profile[x])

                        elif offset_LP == 0:
                            occ_profile_offset.append(occ_profile[x])

                    occ_profile_week.extend(occ_profile_offset)

                for day in week:
                    if day == 'Weekday':
                        occ_profile = occ_profile_week[:48]
                    elif day == 'Weekend':
                        occ_profile = occ_profile_week[48:48*2]

                    """Overtime for lighting and power"""
                    for k in range(3):
                        if k == 0:  # Lighting
                            end_use = 'lighting'
                            overtime = overtime_multiplier_light / 100
                            if day == 'Weekday':
                                L_week = [i + overtime_multiplier_light / 100 for i in occ_profile]  # add overtime to occupancy profile
                                L_week = [i / np.max(L_week) for i in L_week]  # scale to max during the week
                                #L_week = [i * np.max(occ_profile) for i in L_week]  # scale back to the variability of the occupancy
                                LP_profile = L_week
                                LP_profile = [overtime if i < overtime else i for i in LP_profile]

                            if day == 'Weekend':
                                L_weekend = [i + overtime_multiplier_light / 100 for i in occ_profile]
                                L_weekend = [i / np.max(L_week) for i in L_weekend]
                                #L_weekend = [i * np.max(occ_profile) for i in L_weekend]
                                L_weekend = [i - (np.min(L_weekend) - np.min(L_week)) for i in L_weekend]
                                LP_profile = L_weekend
                                LP_profile = [overtime if i < overtime else i for i in LP_profile]

                        elif k == 1:
                            end_use = 'power'
                            overtime = overtime_multiplier_equip / 100
                            if day == 'Weekday':
                                P_week = [i + overtime_multiplier_equip / 100 for i in occ_profile]  # add overtime to occupancy profile
                                P_week = [i / np.max(P_week) for i in P_week]  # scale to max during the week
                                #P_week = [i * np.max(occ_profile) for i in P_week]  # scale back to the variability of the occupancy
                                LP_profile = P_week
                                LP_profile = [overtime if i < overtime else i for i in LP_profile]
                            if day == 'Weekend':
                                P_weekend = [i + overtime_multiplier_equip / 100 for i in occ_profile]
                                P_weekend = [i / np.max(P_week) for i in P_weekend]
                                #P_weekend = [i * np.max(occ_profile) for i in P_weekend]
                                P_weekend = [i - (np.min(P_weekend) - np.min(P_week)) for i in P_weekend]
                                LP_profile = P_weekend
                                LP_profile = [overtime if i < overtime else i for i in LP_profile]

                        if k == 0:
                            light_week.extend(LP_profile)
                        elif k == 1:
                            equip_week.extend(LP_profile)
                        elif k == 2:
                            occ_week.extend(occ_profile)

            light_schedules.append(light_week)
            equip_schedules.append(equip_week)
            occ_schedules.append(occ_week)

            if print_statements is True: print('EquipmentOvertimeMultiplier:', overtime_multiplier_equip)
            if print_statements is True: print('LightOvertimeMultiplier:', overtime_multiplier_light)

            def plot_schedules():
                occ_plot = pd.DataFrame(occ_week, index=timeline, columns=['occupancy'])
                equip_plot = pd.DataFrame(equip_week, columns=['equipment'])
                light_plot = pd.DataFrame(light_week, columns=['lighting'])

                ax = occ_plot.plot(drawstyle='steps')
                equip_plot.plot(ax=ax, drawstyle='steps')
                light_plot.plot(ax=ax, drawstyle='steps')
                ax.set_ylim(0, 1)
                plt.show()
            #plot_schedules()

        if parallel_simulation:
            seasons = [seasonal_occ_factor_week, seasonal_occ_factor_weekend]
            for q, x in enumerate(seasons):
                seasonal_occ_factor_varied = []
                sigmas = [i * (10 / 100) for i in x] # set monthly variation to define sigma
                lower, upper = [x[v] - (2 * i) for v, i in enumerate(sigmas)], [x[v] + (2 * i) for v, i in enumerate(sigmas)]
                for i in range(len(sigmas)):
                    replace_seasonal = stats.truncnorm((lower[i]-x[i])/sigmas[i], (upper[i]-x[i])/sigmas[i], loc=x[i], scale=sigmas[i]).ppf(lhd[run_no, var_num])
                    var_num += 1

                    seasonal_occ_factor_varied.append(replace_seasonal)

                # seasonal_occ_factor_varied = [i if i < 1 else 1 for i in seasonal_occ_factor_varied]
                # seasonal_occ_factor_varied = [i if i > 0 else 0.1 for i in seasonal_occ_factor_varied]

                seasonal_occ_factor_varied = [i / np.max(seasonal_occ_factor_varied) for i in seasonal_occ_factor_varied]

                if q == 0:
                    seasonal_occ_factor_week = seasonal_occ_factor_varied
                if q == 1:
                    seasonal_occ_factor_weekend = seasonal_occ_factor_varied

        months_in_year = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_occ_factor_week_names = [str('SeasonWeekOccFactor_') + str(month) for month in months_in_year]
        seasonal_occ_factor_weekend_names = [str('SeasonWeekendOccFactor_') + str(month) for month in months_in_year]

        if calibrated_case:
            seasonal_occ_factor_week = [df_calibrated.iloc[0][i] for i in seasonal_occ_factor_week_names]
            seasonal_occ_factor_weekend = [df_calibrated.iloc[0][i] for i in seasonal_occ_factor_weekend_names]

        if print_statements is True: print(seasonal_occ_factor_week)
        if print_statements is True: print(seasonal_occ_factor_weekend)

        [input_values.append(i) for i in seasonal_occ_factor_week]
        [input_names.append(i) for i in seasonal_occ_factor_week_names]
        [input_values.append(i) for i in seasonal_occ_factor_weekend]
        [input_names.append(i) for i in seasonal_occ_factor_weekend_names]

        if building_abr in {'CH', 'MPEB'}:
            if simplifications: # will overwrite previous created scheduels if for simplifications
                if scheds['Office_OccSched'].dline[5] in {'NCM'}:
                    occ_week = scheds['Office_OccSched'].dline[8:8 + 48 + 48]
                    occ_week = [float(i) for i in occ_week]
                if scheds['Office_EquipSched'].dline[5] in {'NCM'}:
                    equip_week = scheds['Office_EquipSched'].dline[8:8 + 48 + 48]
                    equip_week = [float(i) for i in equip_week]
                if scheds['Office_LightSched'].dline[5] in {'NCM'}:
                    light_week = scheds['Office_LightSched'].dline[8:8 + 48 + 48]
                    light_week = [float(i) for i in light_week]

            print(occ_week)
            print(equip_week)

        if building_abr == 'CH':
            office_scheds_names = ['Office_OccSched', 'Office_EquipSched', 'Office_LightSched'] # has to align with previous profiles
            office_scheds = [occ_week, equip_week, light_week]

        elif building_abr == 'MPEB':
            office_scheds_names = ['Office_OccSched', 'Office_EquipSched', 'Office_LightSched']  # has to align with previous profiles
            office_scheds = [occ_week, equip_week, light_week]

        elif building_abr == '71':
            # For Office 71 only the seasonal factor is applied and nothing is done with regards to the overtime or offset of hours, so schedules are taken directly from those defined in the csv
            office_scheds_names = ['Office_LightSched_B', 'Office_LightSched_GF', 'Office_LightSched_1',\
                            'Office_LightSched_2','Office_LightSched_3', 'Office_EquipSched_B','Office_EquipSched_GF', \
                            'Office_EquipSched_1', 'Office_EquipSched_2', 'Office_EquipSched_3']
            office_scheds = []
            for i, v in enumerate(office_scheds_names):
                add_schedule = [float(c) for c in scheds[v].dline[8:8+48+48]]
                office_scheds.append(add_schedule)

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # typical 365 year
        if print_statements is True: print(lineno(), 'days in year', sum(days_in_month))

        for sched, sname in enumerate(office_scheds_names):
            # this to prevent the overtime factors to be overridden by the seasonal weekend occupancy changes?
            if building_abr in {'CH', 'MPEB'}:
                if sname in {'Office_EquipSched', 'Office_LightSched'}:
                    seasonal_occ_factor_weekend = seasonal_occ_factor_week # weekend overtime stays same as week (baseload)

            if simplifications:
                if scheds[sname].dline[4] in {'NoSeason'}:  # no seasonal variation for simplifications.
                    seasonal_occ_factor_week = [1 for i in range(1, 13)]  # TODO!!! This is from January to December!!!!
                    seasonal_occ_factor_weekend = [1 for i in range(1, 13)]  # TODO!!! This is from January to December!!!!

            print(seasonal_occ_factor_week)

            SchedProperties = ['Schedule:Compact', sname, 'Fraction']

            for t_month in range(0, 12): #include seasonal variability
                SchedProperties.append('Through: '+str(format(t_month+1, '02')) + '/'+str(days_in_month[t_month])) # where i is the month and then last day of the month
                SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                for i, v in enumerate(office_scheds[sched]): # run through a 96 element long list
                    if i == 48: #  if element 48 is reached, then
                        SchedProperties.append('For: Weekends Holiday')
                    SchedProperties.append('Until: ' + timeline[i])

                    if i < 48:
                        SchedProperties.append(v*seasonal_occ_factor_week[t_month]) # monthly seasonal factor
                    elif i >= 48: #  if element 48 is reached, then start appending the weekends and holidays profile
                        SchedProperties.append(v*seasonal_occ_factor_weekend[t_month])  # monthly seasonal factor

            #if print_statements is True: print(SchedProperties)
            scheddict[sname].append(SchedProperties)
            if print_statements is True: print(lineno(), 'Office', len(SchedProperties), SchedProperties)

        for key in scheds.keys():
            if scheds[key].dline[2] in {'Multiple'}:
                continue #skip heating cooling profiles already added

            elif key == 'ScheduleTypeLimits':
                SchedProperties = ['ScheduleTypeLimits', scheds[key].dline[1], scheds[key].dline[8], scheds[key].dline[9], scheds[key].dline[2], scheds[key].dline[3]]
                scheddict[key].append(SchedProperties)

            elif scheds[key].dline[2] in {'Single'}:
                # import schedules that are based on one value and their sigma
                hours = copy.copy(scheds[key].dline[8:])
                if '' in hours:
                    linx=hours.index('')
                    hours=np.array(copy.copy(hours[0:linx]))
                else:
                    hours=np.array(copy.copy(hours))

                new_hours = []
                #if print_statements is True: print(lineno(), scheds[key].dline[0], scheds[key].dline[4], type(scheds[key].dline[4]))
                if base_case | calibrated_case is True:
                    sigma = 0
                else:
                    sigma = abs(float(scheds[key].dline[4]))
                for i, v in enumerate(hours):
                    mu = abs(float(hours[i]))
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    if base_case | parallel_simulation is True:
                        if sigma != 0:
                            var_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            var_sched = mu
                    elif calibrated_case is True:
                        var_sched = df_calibrated.iloc[0][key+'_Value']
                    new_hours.append(var_sched)

                input_values.append(var_sched)
                input_names.append(key+"_Value")
                if print_statements is True: print(var_num, scheds[key].dline[0])
                var_num += 1

                SchedProperties = ['Schedule:Compact', scheds[key].name, scheds[key].dline[3]]
                SchedProperties.append('Through: 12/31')
                SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                for i, v in enumerate(new_hours):
                    if i == 48:
                        SchedProperties.append('For: Weekends Holiday')
                    SchedProperties.append('Until: ' + timeline[i])
                    if key == 'DeltaTemp_Sched':
                        SchedProperties.append(-v)
                    else:
                        SchedProperties.append(v)
                scheddict[key].append(SchedProperties)

            elif scheds[key].dline[2] in {'NoChange'}:
                if key in scheddict:
                    print(lineno(), 'key exists in scheddict - skipping')
                else:
                    hours = copy.copy(scheds[key].dline[8:])
                    #if print_statements is True: print scheds[key].name
                    if '' in hours:
                        linx=hours.index('')
                        hours=np.array(copy.copy(hours[0:linx]))
                    else:
                        hours=np.array(copy.copy(hours))

                    SchedProperties = ['Schedule:Compact', scheds[key].name, scheds[key].dline[3]]
                    SchedProperties.append('Through: 12/31')
                    SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                    for i, v in enumerate(hours):
                        if i == 48:
                            SchedProperties.append('For: Weekends Holiday')
                        SchedProperties.append('Until: ' + timeline[i])
                        SchedProperties.append(v)

                    scheddict[key].append(SchedProperties)


        if print_statements is True: print(scheddict.keys())
        for key in scheddict.keys(): # Write to idf file
            print(key)
            for i,v in enumerate(scheddict[key][0]):
                if i + 1 == len(scheddict[key][0]):  # if last element in list then put semicolon
                    inf.write('\n')
                    inf.write(str(v) + ';')
                    inf.write('\n\n')
                else:
                    inf.write('\n')
                    inf.write(str(v) + ',')

        #if print_statements is True: print scheddict.keys()
        #if print_statements is True: print scheddict['Office_OccSched'][0][1]

    return input_values, input_names, var_num

def remove_schedules(idf1, building_abr, base_case, simplifications, run_no): #todo should I have the option where I remove only the schedules that are going to be replaced?
    # remove existing schedules and make compact ones from scheds sheet
    scheds = unpick_schedules(rootdir, building_abr, base_case, simplifications, run_no)
    schedule_types = ["SCHEDULE:DAY:INTERVAL", "SCHEDULE:WEEK:DAILY", "SCHEDULE:YEAR"]

    for y in schedule_types:
        #if print_statements is True: print(len(idf1.idfobjects[y]), "existing schedule objects removed in ", y)
        no_of_objects = len(idf1.idfobjects[y])
        to_remove = []
        for i in range(0, no_of_objects):
            if idf1.idfobjects[y][i].Name in scheds.keys() and scheds[idf1.idfobjects[y][i].Name].dline[2] == 'OpenStudioSchedule':
                if print_statements is True: print('Did not remove schedule:', idf1.idfobjects[y][i].Name)  # scheds[v.Name].dline[0]
            else:
                to_remove.append(i)
        for i in to_remove[::-1]:
            if print_statements is True: print('remove', idf1.idfobjects[y][i].Name)
            idf1.popidfobject(y, i)

def remove_existing_outputs(idf1):
    output_types = ["OUTPUT:VARIABLE", "OUTPUT:METER:METERFILEONLY"]
    for y in output_types:
        existing_outputs = idf1.idfobjects[y]
        if print_statements is True: print(len(existing_outputs), "existing output objects removed in", y)
        if len(existing_outputs) == 0:
            if print_statements is True: print("no existing outputs to remove in", y)
        for i in range(0, len(existing_outputs)):
            idf1.popidfobject(y, 0)

def replace_materials_eppy(idf1, lhd, input_values, input_names, var_num, run_no, building_abr, base_case):
    mats = unpick_materials(rootdir, building_abr, base_case, simplifications, run_no)

    if print_statements is True: print(mats.keys())
    materials, material_types = [], []
    for key, val in mats.items():
        if key != '' and val != '':
            materials.append(key)
            material_types.append(val.dline[1].upper())
            material_types = list(set(material_types))  # make unique names
    if print_statements is True: print(materials)
    if print_statements is True: print(material_types)

    mats_types = ['MATERIAL', 'MATERIAL:NOMASS', 'MATERIAL:AIRGAP', 'WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM','WINDOWMATERIAL:BLIND','WINDOWPROPERTY:SHADINGCONTROL']

    for mats_type in material_types: # For every material type, create an object with its material then run through loop
        mats_idf = idf1.idfobjects[mats_type]

        for material in mats_idf: # For each material in object replace content with that defined in csv files
            if material.Name not in mats.keys():
                continue
            else:
                if base_case is True | calibrated_case is True:
                    sigma = 0
                elif parallel_simulation is True:
                    sigma = float(mats[material.Name].dline[62])

                lower, upper = float(mats[material.Name].dline[63]), float(mats[material.Name].dline[64])

                if mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material:NoMass':
                    material.Roughness = mats[material.Name].roughness
                    material.Thermal_Absorptance = mats[material.Name].thermal_abs
                    material.Solar_Absorptance = mats[material.Name].solar_abs
                    material.Visible_Absorptance = mats[material.Name].visible_abs
                    mu = float(mats[material.Name].thermal_res)
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]
                    material.Thermal_Resistance = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material:AirGap':
                    mu = float(mats[material.Name].thermal_res)
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.Thermal_Resistance = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material':
                    material.Roughness = mats[material.Name].roughness
                    material.Thickness = mats[material.Name].thickness_abs
                    material.Density = mats[material.Name].density
                    material.Specific_Heat = mats[material.Name].specific_heat
                    material.Thermal_Absorptance = mats[material.Name].thermal_abs
                    material.Solar_Absorptance = mats[material.Name].solar_abs
                    material.Visible_Absorptance = mats[material.Name].visible_abs
                    mu = float(mats[material.Name].conductivity)
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.Conductivity = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:SimpleGlazingSystem':
                    material.UFactor = mats[material.Name].dline[61] #ufactor
                    material.Solar_Heat_Gain_Coefficient = mats[material.Name].shgc
                    material.Visible_Transmittance = mats[material.Name].vis_ref

                    mu = float(mats[material.Name].dline[61]) #ufactor
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.UFactor = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Blind':
                    mu = float(mats[material.Name].conductivity)
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                                lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.Slat_Conductivity = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Shade':
                    mu = float(mats[material.Name].conductivity)
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                                lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.Conductivity = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Glazing':
                    mu = float(mats[material.Name].conductivity)
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                                lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.Conductivity = eq_mats

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowProperty:ShadingControl':
                    mu = float(mats[material.Name].conductivity) # is actually solar radiation set point control
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                                lhd[run_no, var_num])
                        else:
                            eq_mats = mu
                    elif calibrated_case:
                        eq_mats = df_calibrated.iloc[0][material.Name]

                    material.Setpoint = eq_mats

                if print_statements is True: print(lineno(), var_num, mats[material.Name].name, eq_mats)

                input_names.append(mats[material.Name].name)
                input_values.append(eq_mats)
                var_num +=1

    return input_values, input_names, var_num

def replace_equipment_eppy(idf1, lhd, input_values, input_names, var_num, run_no, building_abr, base_case, simplifications):
    equips = unpick_equipments(rootdir, building_abr, base_case, simplifications, run_no)

    # quick script to delete additional created load definitions (don't know where they come from).
    # If the name is not in the equips.csv then it will be removed from the idf file.
    # Those not removed are adjusted.
    load_types = ["People", "Lights", "ElectricEquipment", "ZoneInfiltration:DesignFlowRate"]
    for y in load_types:
        #if print_statements is True: print(len(idf1.idfobjects[y.upper()]), "existing schedule objects removed in ", y)
        for i in range(0, len(idf1.idfobjects[y.upper()])):
            to_replace = [v for v in equips.keys()]
            existing_loads = [v for v in idf1.idfobjects[y.upper()] if v.Name not in to_replace]

            for load in existing_loads:
                idf1.removeidfobject(load)

    appliances, app_purposes = [], []
    for key, val in equips.items():
        appliances.append(key)
        app_purposes.append(val.dline[1].upper())
    app_purposes = list(set(app_purposes)) #make unique names
    if print_statements is True: print(appliances)
    if print_statements is True: print(app_purposes)

    x = 0
    for equip_type in app_purposes: # For every type, create an object with its material then run through loop
        equip_idf = idf1.idfobjects[equip_type]
        fanzone_list, fanzone_list_two = [], []
        equip_list, equip_list_two = [], []
        infil_list, infil_list_two = [], []
        if print_statements is True: print(lineno(), equip_type)
        for equip in equip_idf: # For each instance of object replace content with that defined in csv files
            # for all ventilation objects, change to the same value
            if equip_type == 'ZONEVENTILATION:DESIGNFLOWRATE':
                if not equip_list:
                    equip_list.append('full')
                    var_num += 1 # todo, changing the var-num here before the actual value, because it is in contrast with other objects, it will use the same var-num for those after ventilaiton...

                len_zonevent = len(idf1.idfobjects[equip_type])
                object_name = "ZoneVentilation"
                if print_statements is True: print(lineno(), 'name', equips[object_name].name, )

                equip.Design_Flow_Rate_Calculation_Method = equips[object_name].dline[19]
                if base_case is True | calibrated_case is True:
                    sigma = 0
                else:
                    sigma = float(equips[object_name].dline[30])

                lower, upper = float(equips[object_name].dline[31]), float(equips[object_name].dline[32])
                mu = float(equips[object_name].dline[27])
                if base_case | parallel_simulation:
                    if sigma > 0:
                        eq_vent = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_vent = mu
                elif calibrated_case:
                    eq_vent = df_calibrated.iloc[0][object_name]
                equip.Flow_Rate_per_Person = eq_vent  # dline[19]

                if not equip_list_two:
                    equip_list_two.append('full')
                    input_values.append(eq_vent)
                    input_names.append(object_name)
                    if print_statements is True: print(eq_vent, var_num)

            elif equip_type == 'FAN:ZONEEXHAUST':
                if not fanzone_list:
                    fanzone_list.append('full')
                    var_num += 1 # give all fans the same randomness

                len_objects = len(idf1.idfobjects[equip_type])
                object_name = "ExhaustFans"

                if print_statements is True: print(lineno(), 'name', equips[object_name].name, )
                if base_case is True | calibrated_case is True:
                    sigma = 0
                else:
                    sigma = float(equips[object_name].dline[30])

                lower, upper = float(equips[object_name].dline[31]), float(equips[object_name].dline[32])
                mu = float(equips[object_name].dline[7])

                if base_case | parallel_simulation:
                    if sigma > 0:
                        eq_efficiency = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_efficiency = mu
                elif calibrated_case:
                    eq_efficiency = df_calibrated.iloc[0][object_name]
                equip.Fan_Total_Efficiency = eq_efficiency

                if not fanzone_list_two:
                    fanzone_list_two.append('full')
                    input_values.append(eq_efficiency)
                    input_names.append(object_name)
                    if print_statements is True: print(eq_efficiency, var_num)

            else:
                if equip_type == 'AIRCONDITIONER:VARIABLEREFRIGERANTFLOW':
                    equip_name = equip.Heat_Pump_Name
                else:
                    equip_name = equip.Name

                if calibrated_case is True:
                    # Read vars in .idf and see if they occur in calibrated file.
                    if equip_name in df_calibrated.columns:
                        print(equip_name)
                    else:
                        continue

                if print_statements is True: print(equip_name)
                if print_statements is True: print(lineno(), 'name', equips[equip_name].name)

                if equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'ElectricEquipment':
                    var_num += 1
                    equip.Design_Level_Calculation_Method = equips[equip_name].dline[19] #dline[19]

                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    else:
                        sigma = float(equips[equip_name].dline[30])

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[21])

                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_equip = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_equip = mu
                    elif calibrated_case:
                        eq_equip = df_calibrated.iloc[0][equip_name]

                    equip.Watts_per_Zone_Floor_Area = eq_equip
                    input_values.append(eq_equip)
                    input_names.append(equips[equip_name].name)

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'People':
                    var_num += 1
                    equip.Number_of_People_Calculation_Method = equips[equip_name].dline[19] #dline[19]

                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    else:
                        sigma = float(equips[equip_name].dline[30])

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[25])

                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_people = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_people = mu
                    elif calibrated_case:
                        eq_people = df_calibrated.iloc[0][equip_name]

                    equip.Zone_Floor_Area_per_Person = eq_people
                    #equip.People_per_Zone_Floor_Area = eq_people
                    input_values.append(eq_people)
                    input_names.append(equips[equip_name].name)


                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'Lights':
                    equip.Design_Level_Calculation_Method = equips[equip_name].dline[19]  # dline[19]
                    var_num += 1
                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    else:
                        sigma = float(equips[equip_name].dline[30])

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[21])
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_light = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_light = mu
                    elif calibrated_case:
                        eq_light = df_calibrated.iloc[0][equip_name]

                    equip.Watts_per_Zone_Floor_Area = eq_light
                    input_values.append(eq_light)
                    input_names.append(equips[equip_name].name)


                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'ZoneInfiltration:DesignFlowRate':

                    if not infil_list:
                        infil_list.append('full')
                        var_num += 1

                    equip.Design_Flow_Rate_Calculation_Method = equips[equip_name].dline[19]  # dline[19]
                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    else:
                        sigma = float(equips[equip_name].dline[30])

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[26])
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            eq_infil = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_infil = mu
                    elif calibrated_case:
                        if building_abr in {'CH', 'MPEB'}:
                            eq_infil = df_calibrated.iloc[0]['InfiltrationRate']
                        else:
                            eq_infil = df_calibrated.iloc[0][equip_name]

                    equip.Flow_per_Exterior_Surface_Area = eq_infil
                    if not infil_list_two:
                        infil_list_two.append('full')
                        input_values.append(eq_infil)
                        input_names.append('InfiltrationRate')

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'DesignSpecification:OutdoorAir':
                    equip.Outdoor_Air_Method = equips[equip_name].dline[19]  # dline[19]
                    var_num += 1
                    sigma = float(equips[equip_name].dline[30])
                    if base_case | calibrated_case | simplifications:
                        sigma = 0

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[28])
                    if base_case | parallel_simulation | simplifications :
                        if sigma > 0:
                            eq_oa = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            eq_oa = mu
                    elif calibrated_case:
                        eq_oa = df_calibrated.iloc[0][equip_name]

                    #todo is this right???
                    equip.Outdoor_Air_Flow_Air_Changes_per_Hour = eq_oa
                    #equip.Outdoor_Air_Flow_per_Zone = eq_oa
                    input_values.append(eq_oa)
                    input_names.append(equips[equip_name].name)


                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'AirConditioner:VariableRefrigerantFlow':
                    if print_statements is True: print(lineno(), equips[equip_name].name) # equips[equip_name].dline
                    var_num += 1
                    #ccop
                    mu = float(equips[equip_name].dline[5])
                    sigma = float(equips[equip_name].dline[33])/100*mu
                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    if print_statements is True: print('sigma in AC units', sigma)
                    if mu > 0:
                        lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    else:
                        lower, upper, sigma = 0, 0, 0
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            ccop = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                                lhd[run_no, var_num])
                        else:
                            ccop = mu
                    elif calibrated_case:
                        ccop = df_calibrated.iloc[0][equip_name+'ccop']

                    var_num += 1

                    #hcop
                    mu = float(equips[equip_name].dline[6])
                    sigma = float(equips[equip_name].dline[33]) / 100 * mu
                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)

                    if base_case | parallel_simulation:
                        if sigma > 0:
                            hcop = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            hcop = mu
                    elif calibrated_case:
                        hcop = df_calibrated.iloc[0][equip_name+'hcop']

                    equip.Gross_Rated_Cooling_COP = ccop
                    input_values.append(ccop)
                    input_names.append(equips[equip_name].name+"ccop")
                    equip.Gross_Rated_Heating_COP = hcop
                    input_values.append(hcop)
                    input_names.append(equips[equip_name].name+"hcop")

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'Boiler:HotWater':
                    sigma = float(equips[equip_name].dline[30])
                    var_num += 1

                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[7])
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            input = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            input = mu
                    elif calibrated_case:
                        input = df_calibrated.iloc[0][equip_name]

                    equip.Nominal_Thermal_Efficiency = input
                    input_values.append(input)
                    input_names.append(equips[equip_name].name)

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'WaterUse:Equipment':
                    sigma = float(equips[equip_name].dline[30])
                    var_num += 1

                    if base_case is True | calibrated_case is True:
                        sigma = 0
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[8])
                    if base_case | parallel_simulation:
                        if sigma > 0:
                            input = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                        else:
                            input = mu
                    elif calibrated_case:
                        print(equip_name)
                        input = df_calibrated.iloc[0][equip_name]

                    equip.Peak_Flow_Rate = input
                    input_values.append(input)
                    input_names.append(equips[equip_name].name)

                else:
                    continue

                if print_statements is True: print(lineno(), var_num, equip_name)

    var_num += 1
    return input_values, input_names, var_num

def add_groundtemps(idf1):
    out = idf1.newidfobject("Site:GroundTemperature:BuildingSurface".upper())
    out.January_Ground_Temperature = 8.3
    out.February_Ground_Temperature = 6.4
    out.March_Ground_Temperature = 5.8
    out.April_Ground_Temperature = 6.3
    out.May_Ground_Temperature = 8.9
    out.June_Ground_Temperature = 11.7
    out.July_Ground_Temperature = 14.4
    out.August_Ground_Temperature = 16.4
    out.September_Ground_Temperature = 17
    out.October_Ground_Temperature = 16.1
    out.November_Ground_Temperature = 13.8
    out.December_Ground_Temperature = 11

#From City of London weatherfile
#	3	0.5				6.65	6.05	7.12	8.8	13.19	16.58	18.87	19.56	18.38	15.75	12.23	8.96	2	8.88	7.8	8.02	8.91	11.85	14.51	16.63	17.77	17.53	16.04000	13.61000	11.05	4				10.72	9.61	9.37	9.71	11.37	13.16	14.8	15.94	16.21	15.56	14.14	12.4
#todo could even quickly run with different ground temps... and see the effect

def add_outputs(idf1, base_case, add_variables, building_abr): # add outputvariables and meters as new idf objects
    # add variables only when base case is simulated

    if base_case is True:
        output_variables = ["Site Outdoor Air Drybulb Temperature",
                            "Site Outdoor Air Relative Humidity",
                            "Site Diffuse Solar Radiation Rate per Area",
                            "Site Direct Solar Radiation Rate per Area"
                            ]

        if add_variables is True:
            extra_variables = [#"Zone Infiltration Mass Flow Rate",
                                "Zone Ventilation Mass Flow Rate"
                                "VRF Heat Pump Cooling Electric Energy",
                                "VRF Heat Pump Heating Electric Energy",
                                "Zone VRF Air Terminal Cooling Electric Energy",
                                "Zone VRF Air Terminal Heating Electric Energy",
                                "Zone Cooling Setpoint Not Met Time",
                                "Zone Heating Setpoint Not Met Time",
                                "Zone Air Temperature",
                                "Cooling Coil Total Cooling Energy",
                                "Fan Coil Total Cooling Energy",
                                "System Node Temperature",
                                "Zone Ventilation Air Change Rate",
                                "Zone Mechanical Ventilation Air Changes per Hour",
                                "System Node Mass Flow Rate",
                                "Air System Gas Energy",
                                "Air System Heating Coil Gas Energy",
                                ]
            output_variables.extend(extra_variables)

        # "Pump Electric Energy",
        # "Fan Electric Power",
        # "Zone Ventilation Mass Flow Rate",
        # "Zone Infiltration Mass Flow Rate",
        # "VRF Heat Pump Cooling Electric Energy",
        # "VRF Heat Pump Heating Electric Energy",
        # "VRF Heat Pump Cooling Electric Energy",
        # "VRF Heat Pump Heating Electric Energy",
        # "VRF Heat Pump Operating Mode",
        # "Zone VRF Air Terminal Cooling Electric Energy",
        # "Zone VRF Air Terminal Heating Electric Energy",
        # "Zone Cooling Setpoint Not Met Time",
        # "Zone Heating Setpoint Not Met Time",
        # "Zone Thermostat Heating Setpoint Temperature",
        # "Zone Thermostat Cooling Setpoint Temperature",
        # "Zone Air Temperature",

        output_diagnostics = []
        # "ReportDuringWarmup"
        # "DisplayAllWarnings",
        # "ReportDuringWarmup",
        # "ReportDuringWarmupConvergence",
        # "ReportDuringHVACSizingSimulation"

        if not output_diagnostics:
            if print_statements is True: print('No output diagnostics')
        else:
            for name in output_diagnostics:
                outvar = idf1.newidfobject("Output:Diagnostics".upper())
                outvar.Key_1 = name

        for name in output_variables:
            outvar = idf1.newidfobject("Output:Variable".upper())
            outvar.Key_Value = ''
            outvar.Variable_Name = name
            if base_case is True:
                outvar.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',
            else:
                outvar.Reporting_Frequency = 'hourly' #'timestep', 'hourly', 'detailed',

    # add the TYPICAL METERS
    output_meters = ["Fans:Electricity",
                     "InteriorLights:Electricity",
                     "InteriorEquipment:Electricity",
                     "WaterSystems:Electricity",
                     "Cooling:Electricity",
                     "Heating:Electricity",
                     "Gas:Facility",
                     "Pumps:Electricity",
                     "HeatRejection:Electricity",
                     "HeatRecovery:Electricity",
                     "DHW:Electricity",
                     "ExteriorLights:Electricity",
                     "Humidifier:Electricity",
                     "Cogeneration:Electricity",
                     "Refrigeration:Electricity",
                     "DistrictCooling:Facility",
                     "DistrictHeating:Facility"
                     ]

    # First create any CUSTOM METERS, these also need to be added to the MeterFileOnly object
    if building_abr == 'MPEB':
        custom_meter_names = ['Plant_Fans', 'AHU_Fans', 'Workshops_Power', 'Workshops_Lights', 'Chillers']
        custom_meters = [
             [["REF1", "Fan Electric Energy"],
              ["REF2", "Fan Electric Energy"],
              ["REF4A", "Fan Electric Energy"],
              ["REF4B", "Fan Electric Energy"],
              ["REF5", "Fan Electric Energy"]],

            [["AHU1_EXTRACT", "Fan Electric Energy"],
              ["AHU1_SUPPLY", "Fan Electric Energy"],
              ["AHU2_EXTRACT", "Fan Electric Energy"],
              ["AHU2_SUPPLY", "Fan Electric Energy"],
              ["AHU3_SUPPLY", "Fan Electric Energy"],
              ["AHU4_SUPPLY", "Fan Electric Energy"],
              ["AHU5_EXTRACT", "Fan Electric Energy"],
              ["AHU5_SUPPLY", "Fan Electric Energy"]],

            [['THERMAL ZONE: B13 CIRCULATION CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B70 CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B71 SERVICE CORRIDOR 1 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B72 SERVICE CORRIDOR 2 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B73 SERVICE CORRIDOR 3 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B75 CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B77 CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B77A_CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B82 CIRCULATION CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B84 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B13A TEA POINT KITCHEN_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B01 ENGINE DYNAMOMETER CELL 1 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B01A CONTROL ROOM CELL 1 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B02 ENGINE DYNAMOMETER CELL 2 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B02A CELL 2 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B03 ENGINE DYNAMOMETER CELL 3 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B03A CELL 3 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B04 ENGINE DYNAMOMETER CELL 4 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B04A CELL 4 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B05 EXPERIMENT ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B06 FSTF A LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B07 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B08 FSTF B LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B09 EXPERIMENT ROOM 2 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B09A EXPERIMENT ROOM 3 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15C MARINETECH LAB LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B17 WC LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B18 WC LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B19 TECHNICIANS REST ROOM LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B20 DIS WC LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15B OFFICE OFFICE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B08A PLANT ROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B81 PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B83 LIFT MOTOR ROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B87 SERVICES PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B95 LIFT MOTOR ROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B97 HYDRAULIC LIFT/ACCESS HATCH PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B80 CORE 1 STAIRS_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B91 CORE 3 STAIRS STAIRS_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B14 WELDING WORKSHOP WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15 WORKSHOP WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15A CNC OPER ROOM WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B22 LABORATORY WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy']],

            [['THERMAL ZONE: B13 CIRCULATION CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B70 CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B71 SERVICE CORRIDOR 1 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B72 SERVICE CORRIDOR 2 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B73 SERVICE CORRIDOR 3 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B75 CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B77 CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B77A_CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B82 CIRCULATION CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B84 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B13A TEA POINT KITCHEN_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B01 ENGINE DYNAMOMETER CELL 1 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B01A CONTROL ROOM CELL 1 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B02 ENGINE DYNAMOMETER CELL 2 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B02A CELL 2 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B03 ENGINE DYNAMOMETER CELL 3 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B03A CELL 3 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B04 ENGINE DYNAMOMETER CELL 4 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B04A CELL 4 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B05 EXPERIMENT ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B06 FSTF A LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B07 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B08 FSTF B LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B09 EXPERIMENT ROOM 2 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B09A EXPERIMENT ROOM 3 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15C MARINETECH LAB LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B17 WC LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B18 WC LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B19 TECHNICIANS REST ROOM LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B20 DIS WC LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15B OFFICE OFFICE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B08A PLANT ROOM PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B81 PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B83 LIFT MOTOR ROOM PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B87 SERVICES PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B95 LIFT MOTOR ROOM PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B97 HYDRAULIC LIFT/ACCESS HATCH PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B80 CORE 1 STAIRS_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B91 CORE 3 STAIRS STAIRS_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B14 WELDING WORKSHOP WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15 WORKSHOP WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15A CNC OPER ROOM WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B22 LABORATORY WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B10 MATERIAL STORE STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B11 ENGINE STORE RACK STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B12 ENGINE PARTS STORE STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B16 EQUIPSTORE STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B21 LV SWITCHROOM STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B71A GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B72A GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B73 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B88 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B89 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B90 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy']],

            [['CHILLER - AIR COOLED 1', 'Chiller Electric Energy'],
            ['CHILLER - AIR COOLED', 'Chiller Electric Energy']]
            ]
    elif building_abr == 'CH':
        custom_meters = []
    elif building_abr == '71':

        custom_meter_names = ['B_Lights', 'GF_Lights', '1st_Lights', '2nd_Lights', '3rd_Lights', 'B_Power', 'GF_Power','1st_Power', '2nd_Power', '3rd_Power', 'Plant_Fans']
        #custom_meters = []
        custom_meters = [
            [['B_ELEVATOR CIRCULATION_B_LIGHTS', 'Lights Electric Energy'],
             ['B_ELEVATORFRONT CIRCULATION_B_LIGHTS', 'Lights Electric Energy'],
             ['B_PLANTROOM PLANT_LIGHTS', 'Lights Electric Energy'],
             ['B_STORE1 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['B_STORE2 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['B_STORE3 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['B_TOILETSHOWERS TOILET_LIGHTS', 'Lights Electric Energy'],
             ['B_TOILETS TOILET_LIGHTS', 'Lights Electric Energy']],

            [['GF_CORRIDOR CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_CORRIDOREXITBACK CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_ELEVATORBACK CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_ELEVATORFRONT CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_ENTRANCE CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_BACKSPACE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['GF_BACKSPACESTORE2 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['GF_STORAGE1 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['GF_TOILET TOILET_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE1 MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE2 MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE3 MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_LOBBYPRESENTATION RECEPTION_LIGHTS', 'Lights Electric Energy']],

            [['1_ELEVATORFRONT CIRCULATION_1_LIGHTS', 'Lights Electric Energy'],  # meter 2
             ['1_ELEVATORBACK CIRCULATION_1_LIGHTS', 'Lights Electric Energy'],
             ['1_MEETING MEETING_1_LIGHTS', 'Lights Electric Energy'],
             ['1_OFFICE OFFICE_1_LIGHTS', 'Lights Electric Energy'],
             ['1_TOILETSTORAGE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['1_TOILETS TOILET_LIGHTS', 'Lights Electric Energy']],

            [['2_ELEVATORBACK CIRCULATION_2_LIGHTS', 'Lights Electric Energy'],
             ['2_ELEVATORFRONT CIRCULATION_2_LIGHTS', 'Lights Electric Energy'],
             ['2_TOILETSTORAGE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['2_OFFICE OFFICE_2_LIGHTS', 'Lights Electric Energy'],
             ['2_TOILETS TOILET_LIGHTS', 'Lights Electric Energy'],
             ['2_MEETING MEETING_2_LIGHTS', 'Lights Electric Energy']],

            [['3_ELEVATORBACK CIRCULATION_3_LIGHTS', 'Lights Electric Energy'],
             ['3_ELEVATORFRONT CIRCULATION_3_LIGHTS', 'Lights Electric Energy'],
             ['3_OFFICE OFFICE_3_LIGHTS', 'Lights Electric Energy'],
             ['3_TOILETSTORAGE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['3_TOILETS TOILET_LIGHTS', 'Lights Electric Energy']],

            [['B_PLANTROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_ELEVATOR CIRCULATION_B_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_ELEVATORFRONT CIRCULATION_B_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_TOILETSHOWERS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_STORE1 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_STORE2 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_STORE3 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_WATERHEATER_E','Water Heater Electric Energy'],
            ['B_WATERHEATER_E','Water Heater Off Cycle Parasitic Electric Energy'],
            ['B_WATERHEATER_E','Water Heater On Cycle Parasitic Electric Energy']],

            [['GF_CORRIDOR CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_CORRIDOREXITBACK CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_ELEVATORBACK CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_ELEVATORFRONT CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_ENTRANCE CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE1 MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE2 MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE3 MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_LOBBYPRESENTATION RECEPTION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_BACKSPACE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_BACKSPACESTORE2 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_STORAGE1 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_TOILET TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_WATERHEATER_C','Water Heater Electric Energy'],
            ['GF_WATERHEATER_C','Water Heater Off Cycle Parasitic Electric Energy'],
            ['GF_WATERHEATER_C','Water Heater On Cycle Parasitic Electric Energy']],

            [['1_ELEVATORBACK CIRCULATION_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_ELEVATORFRONT CIRCULATION_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_MEETING MEETING_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_OFFICE OFFICE_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_TOILETSTORAGE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_WATERHEATER_A','Water Heater Electric Energy'],
            ['1_WATERHEATER_A','Water Heater Off Cycle Parasitic Electric Energy'],
            ['1_WATERHEATER_A','Water Heater On Cycle Parasitic Electric Energy']],

            [['2_ELEVATORBACK CIRCULATION_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_ELEVATORFRONT CIRCULATION_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_MEETING MEETING_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_OFFICE OFFICE_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_TOILETSTORAGE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_WATERHEATER_A','Water Heater Electric Energy'],
            ['2_WATERHEATER_A','Water Heater Off Cycle Parasitic Electric Energy']],

            [['3_OFFICE OFFICE_3_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_ELEVATORBACK CIRCULATION_3_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_ELEVATORFRONT CIRCULATION_3_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_TOILETSTORAGE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
             ['3_WATERHEATER_B', 'Water Heater Electric Energy'],
             ['3_WATERHEATER_B', 'Water Heater Off Cycle Parasitic Electric Energy'],
             ['3_WATERHEATER_B', 'Water Heater On Cycle Parasitic Electric Energy']],

            [['AHU_ExtractFan', 'Fan Electric Energy'],
            ['AHU_SupplyFan', 'Fan Electric Energy'],
            ['1_TOILETEXHAUST', 'Fan Electric Energy'],
            ['2_TOILETEXHAUST', 'Fan Electric Energy']]
        ]

    if custom_meters: # check if any custom outputs in list
        # adding custom meter objects, make a list of list with inner lists containing other output meters and each separate list is a custom meter.
        # default max on number of meters  is 22, set by .idd file.

        for meter in range(len(custom_meters)):
            for output in range(len(custom_meters[meter])):
                new_meter = idf1.newidfobject("Output:Meter".upper())
                new_meter.Name = str(custom_meters[meter][output][0])+':'+str(custom_meters[meter][output][1])
                if base_case is True:
                    new_meter.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',
                else:
                    new_meter.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',

        for meter in range(len(custom_meters)):
            if print_statements is True: print(custom_meters[meter])
            if print_statements is True: print('meter', meter, custom_meter_names[meter])
            outmeter = idf1.newidfobject("Meter:Custom".upper())
            outmeter.Name = custom_meter_names[meter]
            outmeter.Fuel_Type = 'Electricity'
            if print_statements is True: print(len(custom_meters[meter]))
            for output in range(len(custom_meters[meter])):
                if print_statements is True: print(output, custom_meters[meter][output][0])
                if output == 0:
                    outmeter.Key_Name_1 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_1 = custom_meters[meter][output][1]
                if output == 1:
                    outmeter.Key_Name_2 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_2 = custom_meters[meter][output][1]
                if output == 2:
                    outmeter.Key_Name_3 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_3 = custom_meters[meter][output][1]
                if output == 3:
                    outmeter.Key_Name_4 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_4 = custom_meters[meter][output][1]
                if output == 4:
                    outmeter.Key_Name_5 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_5 = custom_meters[meter][output][1]
                if output == 5:
                    outmeter.Key_Name_6 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_6 = custom_meters[meter][output][1]
                if output == 6:
                    outmeter.Key_Name_7 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_7 = custom_meters[meter][output][1]
                if output == 7:
                    outmeter.Key_Name_8 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_8 = custom_meters[meter][output][1]
                if output == 8:
                    outmeter.Key_Name_9 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_9 = custom_meters[meter][output][1]
                if output == 9:
                    outmeter.Key_Name_10 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_10 = custom_meters[meter][output][1]
                if output == 10:
                    outmeter.Key_Name_11 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_11 = custom_meters[meter][output][1]
                if output == 11:
                    outmeter.Key_Name_12 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_12 = custom_meters[meter][output][1]
                if output == 12:
                    outmeter.Key_Name_13 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_13 = custom_meters[meter][output][1]
                if output == 13:
                    outmeter.Key_Name_14 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_14 = custom_meters[meter][output][1]
                if output == 14:
                    outmeter.Key_Name_15 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_15 = custom_meters[meter][output][1]
                if output == 15:
                    outmeter.Key_Name_16 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_16 = custom_meters[meter][output][1]
                if output == 16:
                    outmeter.Key_Name_17 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_17 = custom_meters[meter][output][1]
                if output == 17:
                    outmeter.Key_Name_18 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_18 = custom_meters[meter][output][1]
                if output == 18:
                    outmeter.Key_Name_19 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_19 = custom_meters[meter][output][1]
                if output == 19:
                    outmeter.Key_Name_20 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_20 = custom_meters[meter][output][1]
                if output == 20:
                    outmeter.Key_Name_21 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_21 = custom_meters[meter][output][1]
                if output == 21:
                    outmeter.Key_Name_22 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_22 = custom_meters[meter][output][1]
                if output == 22:
                    outmeter.Key_Name_23 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_23 = custom_meters[meter][output][1]
                if output == 23:
                    outmeter.Key_Name_24 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_24 = custom_meters[meter][output][1]
                if output == 24:
                    outmeter.Key_Name_25 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_25 = custom_meters[meter][output][1]
                if output == 25:
                    outmeter.Key_Name_26 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_26 = custom_meters[meter][output][1]
                if output == 26:
                    outmeter.Key_Name_27 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_27 = custom_meters[meter][output][1]
                if output == 27:
                    outmeter.Key_Name_28 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_28 = custom_meters[meter][output][1]
                if output == 28:
                    outmeter.Key_Name_29 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_29 = custom_meters[meter][output][1]
                if output == 29:
                    outmeter.Key_Name_30 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_30 = custom_meters[meter][output][1]
                if output == 30:
                    outmeter.Key_Name_31 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_31 = custom_meters[meter][output][1]
                if output == 31:
                    outmeter.Key_Name_32 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_32 = custom_meters[meter][output][1]
                if output == 32:
                    outmeter.Key_Name_33 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_33 = custom_meters[meter][output][1]
                if output == 33:
                    outmeter.Key_Name_34 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_34 = custom_meters[meter][output][1]
                if output == 34:
                    outmeter.Key_Name_35 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_35 = custom_meters[meter][output][1]
                if output == 35:
                    outmeter.Key_Name_36 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_36 = custom_meters[meter][output][1]
                if output == 36:
                    outmeter.Key_Name_37 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_37 = custom_meters[meter][output][1]
                if output == 37:
                    outmeter.Key_Name_38 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_38 = custom_meters[meter][output][1]
                if output == 38:
                    outmeter.Key_Name_39 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_39 = custom_meters[meter][output][1]
                if output == 39:
                    outmeter.Key_Name_40 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_40 = custom_meters[meter][output][1]
                if output == 40:
                    outmeter.Key_Name_41 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_41 = custom_meters[meter][output][1]
                if output == 41:
                    outmeter.Key_Name_42 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_42 = custom_meters[meter][output][1]
                if output == 42:
                    outmeter.Key_Name_43 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_43 = custom_meters[meter][output][1]
                if output == 43:
                    outmeter.Key_Name_44 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_44 = custom_meters[meter][output][1]
                if output == 44:
                    outmeter.Key_Name_45= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_45 = custom_meters[meter][output][1]
                if output == 45:
                    outmeter.Key_Name_46 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_46 = custom_meters[meter][output][1]
                if output == 46:
                    outmeter.Key_Name_47 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_47 = custom_meters[meter][output][1]
                if output == 47:
                    outmeter.Key_Name_48 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_48 = custom_meters[meter][output][1]
                if output == 48:
                    outmeter.Key_Name_49 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_49 = custom_meters[meter][output][1]
                if output == 49:
                    outmeter.Key_Name_50 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_50 = custom_meters[meter][output][1]
                if output == 50:
                    outmeter.Key_Name_51= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_51 = custom_meters[meter][output][1]
                if output == 51:
                    outmeter.Key_Name_52 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_52= custom_meters[meter][output][1]
                if output == 52:
                    outmeter.Key_Name_53 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_53= custom_meters[meter][output][1]
                if output == 53:
                    outmeter.Key_Name_54 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_54= custom_meters[meter][output][1]
                if output == 54:
                    outmeter.Key_Name_55 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_55= custom_meters[meter][output][1]
                if output == 55:
                    outmeter.Key_Name_56 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_56= custom_meters[meter][output][1]
                if output == 56:
                    outmeter.Key_Name_57 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_57= custom_meters[meter][output][1]
                if output == 57:
                    outmeter.Key_Name_58 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_58= custom_meters[meter][output][1]
                if output == 58:
                    outmeter.Key_Name_59 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_59= custom_meters[meter][output][1]
                if output == 59:
                    outmeter.Key_Name_60 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_60= custom_meters[meter][output][1]
                if output == 60:
                    outmeter.Key_Name_61= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_61= custom_meters[meter][output][1]
                if output == 61:
                    outmeter.Key_Name_62 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_62= custom_meters[meter][output][1]
                if output == 62:
                    outmeter.Key_Name_63 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_63= custom_meters[meter][output][1]
                if output == 63:
                    outmeter.Key_Name_64 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_64= custom_meters[meter][output][1]
                if output == 64:
                    outmeter.Key_Name_65 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_65= custom_meters[meter][output][1]
                if output == 65:
                    outmeter.Key_Name_66 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_66= custom_meters[meter][output][1]
                if output == 66:
                    outmeter.Key_Name_67 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_67= custom_meters[meter][output][1]
                if output == 67:
                    outmeter.Key_Name_68= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_68= custom_meters[meter][output][1]
                if output == 68:
                    outmeter.Key_Name_69 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_69= custom_meters[meter][output][1]
                if output == 69:
                    outmeter.Key_Name_70 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_70= custom_meters[meter][output][1]
                if output == 70:
                    outmeter.Key_Name_71 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_71= custom_meters[meter][output][1]
                if output == 71:
                    outmeter.Key_Name_72 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_72= custom_meters[meter][output][1]
                if output == 72:
                    outmeter.Key_Name_73 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_73= custom_meters[meter][output][1]
                if output == 73:
                    outmeter.Key_Name_74 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_74= custom_meters[meter][output][1]
                if output == 74:
                    outmeter.Key_Name_75 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_75= custom_meters[meter][output][1]
                if output == 75:
                    outmeter.Key_Name_76 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_76= custom_meters[meter][output][1]
                if output == 76:
                    outmeter.Key_Name_77 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_77= custom_meters[meter][output][1]
                if output == 77:
                    outmeter.Key_Name_78 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_78= custom_meters[meter][output][1]
                if output == 78:
                    outmeter.Key_Name_79 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_79= custom_meters[meter][output][1]
                if output == 79:
                    outmeter.Key_Name_80 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_80= custom_meters[meter][output][1]
                if output == 80:
                    outmeter.Key_Name_81= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_81= custom_meters[meter][output][1]
                if output == 81:
                    outmeter.Key_Name_82 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_82= custom_meters[meter][output][1]
                if output == 82:
                    outmeter.Key_Name_83= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_83= custom_meters[meter][output][1]
                if output == 83:
                    outmeter.Key_Name_84 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_84= custom_meters[meter][output][1]
                if output == 84:
                    outmeter.Key_Name_85 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_85= custom_meters[meter][output][1]
                if output == 85:
                    outmeter.Key_Name_86 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_86= custom_meters[meter][output][1]
                if output == 86:
                    outmeter.Key_Name_87 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_87= custom_meters[meter][output][1]
                if output == 87:
                    outmeter.Key_Name_88 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_88= custom_meters[meter][output][1]
                if output == 88:
                    outmeter.Key_Name_89= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_89= custom_meters[meter][output][1]
                if output == 89:
                    outmeter.Key_Name_90 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_90= custom_meters[meter][output][1]
                if output == 90:
                    outmeter.Key_Name_91 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_91= custom_meters[meter][output][1]
                if output == 91:
                    outmeter.Key_Name_92 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_92= custom_meters[meter][output][1]
                if output == 92:
                    outmeter.Key_Name_93 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_93= custom_meters[meter][output][1]
                if output == 93:
                    outmeter.Key_Name_94 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_94= custom_meters[meter][output][1]
                if output == 94:
                    outmeter.Key_Name_95 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_95= custom_meters[meter][output][1]
                if output == 95:
                    outmeter.Key_Name_96 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_96= custom_meters[meter][output][1]
                if output == 96:
                    outmeter.Key_Name_97 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_97= custom_meters[meter][output][1]
                if output == 97:
                    outmeter.Key_Name_98 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_98= custom_meters[meter][output][1]
                if output == 98:
                    outmeter.Key_Name_99 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_99= custom_meters[meter][output][1]
                if output == 99:
                    outmeter.Key_Name_100 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_100= custom_meters[meter][output][1]
                if output == 100:
                    outmeter.Key_Name_101 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_101= custom_meters[meter][output][1]
                if output == 101:
                    outmeter.Key_Name_102 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_102= custom_meters[meter][output][1]
                if output == 102:
                    outmeter.Key_Name_103 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_103= custom_meters[meter][output][1]
                if output == 103:
                    outmeter.Key_Name_104 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_104= custom_meters[meter][output][1]
                if output == 104:
                    outmeter.Key_Name_105 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_105= custom_meters[meter][output][1]
                if output == 105:
                    outmeter.Key_Name_106 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_106= custom_meters[meter][output][1]
                if output == 106:
                    outmeter.Key_Name_107 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_107= custom_meters[meter][output][1]
                if output == 107:
                    outmeter.Key_Name_108 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_108= custom_meters[meter][output][1]
                if output == 108:
                    outmeter.Key_Name_109 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_109= custom_meters[meter][output][1]
                if output == 109:
                    outmeter.Key_Name_110 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_110= custom_meters[meter][output][1]

    # extend any building specific meters
    if building_abr == 'MPEB':
        output_meters.extend(
            [
            "InteriorEquipment:Electricity:Zone:THERMAL ZONE: 409 MACHINE ROOM",
            "InteriorEquipment:Electricity:Zone:THERMAL ZONE: G01B MACHINE ROOM"
             ])

    if building_abr == '71':
        if print_statements is True: print('no extra meters')

    if custom_meters:
        output_meters.extend(custom_meter_names)  # add the custom meters to be added to the .mtr file!

    print(output_meters)

    for name in output_meters:
        outmeter = idf1.newidfobject("Output:Meter:MeterFileOnly".upper())
        outmeter.Name = name
        if base_case is True:
            outmeter.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',
        else:
            outmeter.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',

# Remove all comments from idf file so as to make them smaller.
def remove_comments(run_file):
    """
    removes comments from .idf file after writing.
    :param run_file: .idf file name.
    :return:
    """
    with open(run_file, 'r+') as f:
        data = f.readlines()
        for num, line in enumerate(data):
            data[num] = re.sub(re.compile("!-.*?\n"), "\n", data[num])  # remove all occurances singleline comments (!-COMMENT\n ) from string
            #if print_statements is True: print data[num]
        f.seek(0) #move to front of file
        f.writelines(data)
        f.truncate()

def set_runperiod(idf1, building_abr, run_periods):
    idf1.popidfobject('RunPeriod'.upper(), 0)

    for i in run_periods:
        obj = idf1.newidfobject('RunPeriod'.upper())
        obj.Begin_Month = i[0]
        obj.Begin_Day_of_Month = i[1]
        obj.End_Month = i[2]
        obj.End_Day_of_Month = i[3]
        obj.Start_Year = i[4]
        obj.Day_of_Week_for_Start_Day = i[5]
        obj.Name = i[6]

        obj.Use_Weather_File_Holidays_and_Special_Days = 'No'
        obj.Use_Weather_File_Daylight_Saving_Period = 'No'

def set_holidays(idf1, building_abr):
    if building_abr == 'CH': #2017
        holidays = [['Summer bank holiday', '8/28'], ['Spring bank holiday', '5/29'], ['Early may bank holiday', '5/1'],
                    ['Easter Monday', '4/17'], ['Easter Tue', '4/18'], ['Easter Wed', '4/19'], ['Good Friday', '4/14'], ['White Thursday', '4/13'],
                    ['New Years Day', '1/1'], ['New Years Day', '1/2'], ['Christmas', '12/25'],
                    ['Christmas2', '12/26'], ['Christmas3', '12/27'], ['Christmas4', '12/28'], ['Christmas5', '12/29']]  # 2017
    elif building_abr == 'MPEB': #2017 and 2016?
        holidays = [['Summer bank holiday', '8/28'], ['Spring bank holiday', '5/29'], ['Early may bank holiday', '5/1'],
                    ['Easter Monday', '4/17'], ['Easter2', '4/18'],['Easter3', '4/19'],['Good Friday', '4/14']]  # 2017
    elif building_abr == '71': # 2014
        holidays = [['Summer bank holiday', '8/25'], ['Spring bank holiday', '5/25'], ['Early may bank holiday', '5/5'],
                    ['Easter Monday', '4/21'], ['Good Friday', '4/18'], ['New Years Day', '1/1'], ['New Years Day', '1/2'], ['Christmas', '12/25'],
                    ['Christmas2', '12/26'], ['Christmas3', '12/27'], ['Christmas4', '12/28'], ['Christmas5', '12/29']]  # 2014

    for h in range(len(holidays)): # create new object for every holiday
        holiday = idf1.newidfobject("RunPeriodControl:SpecialDays".upper())
        holiday.Name = holidays[h][0]
        holiday.Start_Date = str(holidays[h][1])
        holiday.Duration = 1
        holiday.Special_Day_Type = 'Holiday'

def run_lhs(idf1, lhd, building_name, building_abr, base_case, simplifications, no_simplifications,  remove_sql, add_variables, run_periods, n_samples, from_samples, save_idfs, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation, seasonal_occ_factor_week, seasonal_occ_factor_weekend):
    """
    # Explanation
    # 1. replace_materials_eppy uses mat_props.csv, replace_equipment_eppy uses equipment_props.csv, replace_schedules uses house_scheds.csv
    # 2. adds ground temperatures and output variables
    # 3. define number of idf files to be created and base model file name
    # 4. the functions sample the inputs using lhs, inputs are in the csv files to be manually copied
    # 5. then create separate idf files

    # NOTE:
    # var_num +=1 implies a new random number for another variable generated with LHS
    :param idf1:
    :param lhd:
    :param building_name:
    :param building_abr:
    :param base_case:
    :param simplifications:
    :param no_simplifications:
    :param remove_sql:
    :param add_variables:
    :param run_periods:
    :param n_samples:
    :param from_samples:
    :param save_idfs:
    :param overtime_multiplier_equip:
    :param overtime_multiplier_light:
    :param multiplier_variation:
    :param seasonal_occ_factor_week:
    :param seasonal_occ_factor_weekend:
    :return:
    """

    if base_case is False | remove_sql is True:
        idf1.popidfobject('Output:SQLite'.upper(), 0) # remove sql output, have all the outputs in the .eso and meter data in .mtr
    idf1.popidfobject('Output:VariableDictionary'.upper(), 0)
    idf1.popidfobject('Output:Table:SummaryReports'.upper(), 0)
    idf1.popidfobject('OutputControl:Table:Style'.upper(), 0)

    collect_inputs = []
    if base_case is True | calibrated_case is True: # search script for base_case to adjust internal base_case values
        n_samples = 1
        from_samples = 0
    if simplifications:
        from_samples = 0
        n_samples = no_simplifications

    # change the base idf first by adding ground temps, and removing existing objects before adding new ones.
    add_groundtemps(idf1)
    set_runperiod(idf1, building_abr, run_periods)
    set_holidays(idf1, building_abr)
    remove_existing_outputs(idf1)

    add_outputs(idf1, base_case, add_variables, building_abr)


    occ_schedules = []
    light_schedules = []
    equip_schedules = []

    if base_case is True:
        csv_outfile = save_dir + "/inputs_basecase_" + building_name + strftime("_%d_%m_%H_%M", gmtime()) + ".csv"
    elif base_case is not True:
        csv_outfile = save_dir + "/inputs_" + building_name + strftime("_%d_%m_%H_%M", gmtime()) + ".csv"

    for run_no in range(from_samples, n_samples):

        remove_schedules(idf1, building_abr, base_case, simplifications, run_no)

        print(run_no)
        var_num = 0
        input_names = []
        input_values = []

        if base_case is True:
            run_file = save_dir + building_name + "_basecase.idf"  # use new folder for save location, zero pad to 4 numbers
        if simplifications is True:
            run_file = save_dir + "/" + building_name + "simplification_" + str(run_no) + ".idf"  # use new folder for save location, zero pad to 4 numbers
        if parallel_simulation is True:
            run_file = save_dir + "/" + building_name + "_" + str(format(run_no, '04')) + ".idf"  # use new folder for save location, zero pad to 4 numbers
        elif calibrated_case:
            run_file = save_dir + building_name + time_step + str(end_uses) + hours + "_calibrated.idf"  # use new folder for save location, zero pad to 4 numbers

        input_values, input_names, var_num = replace_equipment_eppy(idf1, lhd, input_values, input_names, var_num, run_no, building_abr, base_case, simplifications)

        var_equip = var_num
        if print_statements is True: print("number of variables changed for equipment ", var_equip)
        #input_values, input_names, var_num = replace_materials_eppy(idf1, lhd, input_values,input_names,var_num,run_no,building_abr, base_case)
        #var_mats = var_num-var_equip
        var_mats = 0
        if print_statements is True: print("number of variables changed for materials ", var_mats)

        if save_idfs is True:
            idf1.saveas(run_file)
            remove_comments(run_file)

        #replace_schedules append to file instead of using eppy, it will open the written idf file
        var_scheds = var_num-var_mats-var_equip
        input_values, input_names, var_num = replace_schedules(run_file, lhd, input_values,input_names, occ_schedules, light_schedules, equip_schedules,var_num,run_no,building_abr, base_case, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation, seasonal_occ_factor_week, seasonal_occ_factor_weekend) #create_schedules = the heating and coolign profiles. Which are not to be created for MPEB (however, the remove_schedules shouldn't delete existing ones. Or create them...

        if print_statements is True: print("number of variables changed for schedules", var_scheds)

        if len(input_names) != len(input_values):
            raise ValueError('Length of input names and input values not equal.')

        input_names.append('run_no')
        input_values.append(run_no)
        collect_inputs.append(input_values)

        if print_statements is True: print("total number of variables ", var_num)
        if print_statements is True: print(input_names)
        if print_statements is True: print(input_values)
        print('file saved here', run_file)
        if print_statements is True: print('file used', "{}".format(rootdir) + "/" + building_name + ".idf")

        if from_samples != 0: #todo instead I should append to existing file (but no time)
            if run_no % 20 == 0:
                df_inputs = pd.DataFrame(collect_inputs, columns=input_names)
                df_inputs = df_inputs.reindex(sorted(df_inputs.columns), axis=1)  # sort columns alphabetically
                df_inputs.to_csv(csv_outfile, index=False)
        else:
            if run_no % 20 == 0:
                df_inputs = pd.DataFrame(collect_inputs, columns=input_names)
                df_inputs = df_inputs.reindex(sorted(df_inputs.columns), axis=1)  # sort columns alphabetically
                df_inputs.to_csv(csv_outfile, index=False)



    #Write inputs to csv file
    if print_statements is True: print(len(input_values), len(input_names))


    df_inputs = pd.DataFrame(collect_inputs, columns=input_names)
    df_inputs = df_inputs.reindex(sorted(df_inputs.columns), axis=1) # sort columns alphabetically

    pd.DataFrame(occ_schedules).to_csv('D:\OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/' + 'occ_schedules.csv', index=False)
    pd.DataFrame(light_schedules).to_csv('D:\OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/' + 'light_schedules.csv', index=False)
    pd.DataFrame(equip_schedules).to_csv('D:\OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/' + 'equip_schedules.csv', index=False)

    df_inputs.to_csv(csv_outfile[:-4]+'final.csv', index=False)



if __name__ == '__main__':
    def start__main__():
        print('start')
    start__main__()

    main()