import pandas as pd
import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm #color map
import getpass
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, sem

# import statsmodels.api as sm

def ReadRuns(parallel_runs_harddisk, time_step, NO_ITERATIONS):
    """
    Read eplusmtr files in a folder and combine them in one dataframe, formatted based on time_step.
    :param parallel_runs_harddisk: location of eplusmtr.csv files.
    :param time_step: resample to 'month' or 'year'.
    :param NO_ITERATIONS: number of files to include in new dataframe.
    :return:
    """
    def strip_columns(df):
        """
        Rename columns of loaded eplusmtr file
        - get rid off name "Thermal Zone"
        - delete any unwanted symbols
        - strip blank spaces
        - split by ':'
        - delete word "Interior"
        """
        cols = df.columns.tolist()
        for i, v in enumerate(cols):  # rename columns
            if 'THERMAL ZONE' in cols[i]:
                rest = cols[i].split(':', 1)[1]
                rest = ''.join(cols[i].partition('ZONE:')[-1:])
                rest = re.sub("([(\[]).*?([)\]])", "\g<1>\g<2>", rest)  # remove text within symbols
                rest = re.sub("[(\[].*?[)\]]", "", rest)  # remove symbols
                rest = rest.strip()  # remove leading and trailing spaces
            elif ':' not in cols[i]:
                rest = cols[i]
                rest = re.sub("([(\[]).*?([)\]])", "\g<1>\g<2>", rest)  # remove text within symbols
                rest = re.sub("[(\[].*?[)\]]", "", rest)  # remove symbols
                rest = rest.strip()  # remove leading and trailing spaces
            else:
                rest = cols[i].split(':', 1)[0]  # removes all characters after ':'
                rest = rest.replace('Interior', '')
            df.rename(columns={cols[i]: rest}, inplace=True)
        return df

    def read_eplusout(df):
        """
        Read an eplusmtr.csv file and manipulate index and content
        - delete design days
        - set index
        - remove last row
        - displace index to align with measured data time range.
        """
        if df.shape[0] < 10000:  # hourly
            df = df[48:]  # exclude 2 design days
            rng = pd.date_range(start='09/01/2016 01:00:00', end='09/01/2017', freq='H')
            df = df.set_index(rng)
            df = df[:-1]  # remove the last row cause it is in the next year.
        else:
            df = df[48 * 4:]
            rng = pd.date_range(start='09/01/2016 00:15:00', end='09/01/2017', freq='15Min')
            df = df.set_index(rng)
            df.index = df.index - pd.DateOffset(hours=-.75)
            df = df[:-1]

        df = strip_columns(df)
        df = df.div(3600 * 1000)

        return df

    runs = pd.DataFrame()
    not_run = []
    run_numbers = []

    for i in range(NO_ITERATIONS):
        file_name = 'eplusmtr' + str(format(i, '04')) + '.csv'
        if os.path.isfile(os.path.join(parallel_runs_harddisk, file_name)):
            run_numbers.append(i)
        else:
            not_run.append(str(format(i, '04')))

    print('run_numbers', len(run_numbers), run_numbers)
    print('not run', len(not_run), not_run)
    print('timestep set to', time_step)

    for i in run_numbers:
        if i % 50 == 0:
            print(i)

        file_name = 'eplusmtr'+str(format(i, '04'))+'.csv'

        if os.path.isfile(os.path.join(parallel_runs_harddisk, file_name)):
            df = pd.read_csv(os.path.join(parallel_runs_harddisk, file_name), header=0, index_col=0)
            df_runs = read_eplusout(df)

        if time_step == 'year': # with a full year of simulation
            df = df_runs.resample('A').sum()
            #df_pred = df_pred.sum(axis=0)

            df = df.sum(axis=0)
            df = pd.DataFrame(df)
            runs = pd.concat([runs, df.T], axis=0)

            if i == run_numbers[-1]:
                runs.reset_index(drop=True, inplace=True)  # throw out the index (years)

        elif time_step == 'month':
            df = df_runs.resample('M').sum()
            #df_pred = df_pred.sum(axis=0)

            df.index = [i.strftime('%b %y') for i in df.index]

            dict_in = {i: df[i].T for i in df.columns.tolist()} # create dictionary of col names and dfs
            df_multi = pd.concat(dict_in.values(), axis=0, keys=dict_in.keys()) #create multiindex
            runs = pd.concat([runs, df_multi], axis=1)

            if i == run_numbers[-1]:
                runs = runs.T

    return runs

def ImportOutputs(DataPath_model_real, inputs):
    not_run = []
    run_numbers = []

    for i in range(NO_ITERATIONS):
        file_name = 'eplusmtr' + str(format(i, '04')) + '.csv'
        if os.path.isfile(os.path.join(parallel_runs_harddisk, file_name)):
            run_numbers.append(i)
        else:
            not_run.append(str(format(i, '04')))

    df = pd.read_csv(DataPath_model_real + inputs, header=0)
    cols_inputs = df.columns.tolist()
    df_in = df.copy()

    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)

    X_real = df.ix[run_numbers]
    X_real = X_real.drop(df.columns[[0]], axis=1)  # drop run_no column
    X_real = X_real.as_matrix()

    return X_real, cols_inputs

def Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs):
    cols_outputs.append('Total')
    Y_real = pd.DataFrame(pd.concat([pd.DataFrame(Y_real), pd.DataFrame(Y_real.sum(axis=1), columns=['Total'])], axis=1))
    Y_real = Y_real.as_matrix()
    print('no. variables', len(cols_inputs))
    print('cols_outputs', cols_outputs)
    label = ['Standardized', 'Spearman', 'Pearson']

    for p, q in enumerate(label):
        df_corr = pd.DataFrame(cols_inputs)
        df_corr.columns = [q]
        for j in range(Y_real.shape[1]):
            coef_list = []
            for i in range(X_real.shape[1]):
                if p == 0:
                    # coef_list.append(sm.OLS(zscore(X_real[:, i]), zscore(Y_real.iloc[:, j])).fit().params[0])
                    #print('install statsmodels')
                    continue
                elif p == 1:
                    coef_list.append(spearmanr(X_real[:, i], Y_real[:, j])[0])
                elif p == 2:
                    coef_list.append(pearsonr(X_real[:, i], Y_real[:, j])[0])
            df_corr[cols_outputs[j]] = pd.Series(coef_list)  # append list to df
        df_corr.set_index(q, inplace=True)
        print(df_corr.head())

        if p == 0:
            df_corr_stnd = df_corr
        elif p == 1:
            df_corr_spearman = df_corr
        elif p == 2:
            df_corr_pearson = df_corr

    return df_corr_stnd, df_corr_spearman, df_corr_pearson


def HeatmapCorrelations(df, runs):
    df_perc_total = [i / runs.mean(axis=0).sum() for i in runs.mean(axis=0)]
    df_perc_total.append(1) # for the total, which is 100%

    runs = pd.DataFrame(pd.concat([runs, pd.DataFrame(runs.sum(axis=1), columns=['Total'])], axis=1))
    cols_outputs = runs.columns.tolist()

    df_standard = df.multiply(df_perc_total)
    df_standard = (df_standard-df_standard.mean().mean()) / (df_standard.max().max() - df_standard.min().min())

    df_index = df[abs(df[abs(df) > .25].count(axis=1) > 0.25)]
    df_index_standard = df_standard[abs(df_standard[abs(df_standard) > .25].count(axis=1) > 0.25)]

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    df_index = f7(df_index.index.tolist()+df_index_standard.index.tolist())
    df = df.loc[df_index]
    df_standard = df_standard.loc[df_index]

    cols_outputs_add = [v+' ['+str(round(df_perc_total[i]*100,1))+'%]' for i,v in enumerate(cols_outputs)]

    heatmap = df.as_matrix(columns=cols_outputs)
    fig, ax = plt.subplots(figsize=(10 / 2.54, 12 / 2.54))

    use_sns = False
    if use_sns is True:
        ax = sns.heatmap(heatmap, linewidths=.8, annot=True,  cmap='RdBu_r', annot_kws={"size": 6}, fmt='.2f', vmin=-1, vmax=1) #cmap=cm.Spectral_r,
        ax.set_yticklabels(df.index, rotation=0)  # set y labels ('variables') from index
        ax.set_xticklabels(cols_outputs_add, rotation=90)  # set y labels ('variables') from index
        ax.xaxis.tick_top()

    else:
        im = ax.matshow(heatmap, cmap='RdBu_r', interpolation='none')
        cbar = plt.colorbar(im, fraction=0.04555, pad=0.04)
        cbar.ax.tick_params()

        ind_x = np.arange(df.shape[1])
        ind_y = np.arange(df.shape[0])

        ax.set_aspect('equal')
        ax.set_yticks(ind_y)  # set positions for y-labels, .5 to put the labels in the middle
        ax.set_yticklabels(df.index, rotation = 0)  # set y labels ('variables') from index
        ax.set_yticks(ind_y + .5, minor=True)
        ax.set_xticklabels('')
        ax.set_xticks(ind_x)  # set positions for y-labels, .5 to put the labels in the middle
        ax.set_xticklabels(cols_outputs_add, rotation=90)  # set y labels ('variables') from index
        ax.set_xticks(ind_x + .5, minor=True)

        ax.grid(which='minor', linewidth=1, color='white')
        ax.grid(False, which='major')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # annotating the data inside the heatmap
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                plt.text(x, y, '%.2f' % heatmap[y][x],horizontalalignment='center',verticalalignment='center',fontsize=6)

    plt.savefig(DataPathImages +  '_HeatMapCorrelations.png', dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    def start():
        print('start')
    start()

    UserName = getpass.getuser()
    if UserName == 'cvdronke':
        start_path = 'C:/Users/' + UserName + '/'
    else:
        start_path = 'D:/'

    BuildingList = ['05_MaletPlaceEngineering', '01_CentralHouse', '02_BuroHappold_17', '03_BuroHappold_71']  # Location of DATA
    BuildingHardDisk = ['05_MaletPlaceEngineering_Project', '01_CentralHouse_Project', '02_BuroHappold_17','03_BuroHappold_71']
    DataFilePaths = ['MPEB', 'CentralHouse', '17', '71']  # Location of STM data and file naming
    BuildingLabels = ['MPEB', 'CH', 'Office 17', 'Office 71']
    BuildingAbbreviations = ['MPEB', 'CH', '17', '71', 'Nothing']
    InputVariables = ['inputs_MaletPlace_FINAL.csv', 'inputs_CentralHouse_222_29_11_15_02_2870.csv', 'Inputs.csv',
                      'inputs_BH71_27_09_13_46.csv']
    FloorAreas = [9579, 5876, 1924, 1691]
    NO_ITERATIONS = [3000, 2870, 100, 100]

    building_num = 1  # 0 = MPEB, 1 = CH, 2 = 17, 3 = 71
    time_step = 'year' # month or year

    building_abr = BuildingAbbreviations[building_num]
    datafile = DataFilePaths[building_num]
    building_label = BuildingLabels[building_num]
    floor_area = FloorAreas[building_num]
    building_harddisk = BuildingHardDisk[building_num]
    NO_ITERATIONS = NO_ITERATIONS[building_num]
    inputs = InputVariables[building_num]

    parallel_runs_harddisk = start_path + 'OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/Eplusmtr/'
    DataPath_model_real = start_path + 'OneDrive - BuroHappold\EngD_hardrive backup/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/'
    DataPathImages = start_path + 'OneDrive - BuroHappold/01 - EngD/01 - Thesis/02_Images/'

    to_hdf = False
    if to_hdf:
        runs = ReadRuns(parallel_runs_harddisk, time_step=time_step, NO_ITERATIONS=NO_ITERATIONS)
        print(runs.head())
        def filter_enduses(df):
            if building_abr in {'CH'}:
                df_systems = df[['Fans', 'Cooling', 'Heating', 'Pumps']]
                df_landp = df[['Lights', 'Equipment', 'WaterSystems']]
                df_gas = df[['Gas']]

                df = pd.concat([df_systems, df_landp, df_gas], axis=1)

                df_runs_systems = pd.DataFrame(df[['Heating', 'Pumps', 'Cooling', 'Fans']].sum(axis=1), columns=['Systems'])
                df_runs_landp = pd.DataFrame(df[['Lights', 'Equipment', 'WaterSystems']].sum(axis=1), columns=['L&P'])

                df_runs_gas = df[['Gas']]
                df_runs = pd.concat([df_runs_landp, df_runs_systems, df_runs_gas], axis=1)

                df_enduses = df

            elif building_abr in {'MPEB'}:
                df_pred = df

                # Separate PLant and FCU fans
                df_pred_plant_fans = pd.DataFrame(df[['PLANT_FANS','AHU_FANS']].sum(axis=1), columns=['Plant Fans'])
                df_pred_fans = pd.concat([df_pred_plant_fans, df_pred[['Fans']]], axis=1)
                df_pred_fans['FCU Fans'] = df_pred_fans['Fans'] - df_pred_fans['Plant Fans']

                df_workshop = pd.DataFrame(df[['WORKSHOPS_POWER', 'WORKSHOPS_LIGHTS']].sum(axis=1), columns=['Workshops'])

                # Separate Servers and Equipment
                df_pred_servers = pd.DataFrame(df_pred[['G01B MACHINE ROOM', '409 MACHINE ROOM']].sum(axis=1), columns=['Servers'])
                df_pred_equipment = pd.concat([df_pred['Equipment'], df_pred_servers, df_pred['WORKSHOPS_POWER']], axis=1)
                df_pred_equipment['Equipment'] = df_pred['Equipment'] - df_pred_equipment['Servers'] - df_pred_equipment['WORKSHOPS_POWER']

                df_pred_lights = pd.concat([df_pred['Lights'], df_pred['WORKSHOPS_LIGHTS']], axis=1)
                df_pred_lights['Lights'] = df['Lights'] - df['WORKSHOPS_LIGHTS']

                df.rename(columns={'CHILLERS': 'Chillers'}, inplace=True)
                df_pred['Lights'] = df_pred['Lights'] - df_pred['WORKSHOPS_LIGHTS']
                df_cooling = df['Cooling'] - df['Chillers'] #separate chillers from total cooling
                df_runs_systems = pd.DataFrame(pd.concat([df_pred[['Heating', 'Pumps']], df_pred_plant_fans, df_cooling], axis=1).sum(axis=1), columns=['Systems'])
                df_runs_landp = pd.DataFrame(pd.concat([df_pred[['Lights', 'WaterSystems']], df_pred_fans[['FCU Fans']], df_pred_equipment[['Equipment']]], axis=1).sum(axis=1), columns=['L&P'])

                df_runs = pd.concat([df_runs_landp, df['Chillers']/3, df_runs_systems, df_pred_servers, df_workshop], axis=1)

                df_enduses = pd.concat([df[['Fans', 'WaterSystems', 'DistrictHeating']], df_pred_equipment[['Equipment']],
                                        df[['Chillers']], df_pred_lights[['Lights']], df[['Heating', 'Pumps']],
                                        df_workshop, df_pred_servers], axis=1)
        runs.to_hdf(DataPath_model_real + building_abr+ '_' + str(NO_ITERATIONS) +'_RUNS_' + time_step + '.hdf', 'runs', mode='w')

    runs = pd.read_hdf(DataPath_model_real + building_abr+ '_' + str(NO_ITERATIONS) +'_RUNS_' + time_step + '.hdf', 'runs')
    cols_outputs = runs.columns.tolist()

    Y_real = runs.as_matrix()
    X_real, cols_inputs = ImportOutputs(DataPath_model_real, inputs)

    # X_real_data = np.vstack((cols_inputs, X_real))
    # Y_real_data = np.vstack((cols_outputs, Y_real))
    #input_outputs = np.hstack((X_real_data, Y_real_data))
    #pd.DataFrame(input_outputs).to_csv(DataPath_model_real + 'input_outputs_' + time_step + '.csv', header=None)

    df_corr_stnd, df_corr_spearman, df_corr_pearson = Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs)
    print(df_corr_pearson.head())

    HeatmapCorrelations(df_corr_pearson, runs)
    plt.show()

