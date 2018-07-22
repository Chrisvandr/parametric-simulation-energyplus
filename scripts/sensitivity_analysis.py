import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm #color map
import getpass
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, sem
import itertools
import matplotlib.ticker as ticker
# import statsmodels.api as sm

def main():
    BuildingList = ['05_MaletPlaceEngineering', '01_CentralHouse', '02_BuroHappold_17', '03_BuroHappold_71']  # Location of DATA
    BuildingHardDisk = ['05_MaletPlaceEngineering_Project', '01_CentralHouse_Project', '02_BuroHappold_17','03_BuroHappold_71']
    DataFilePaths = ['MPEB', 'CentralHouse', '17', '71']  # Location of STM data and file naming
    BuildingLabels = ['MPEB', 'CH', 'Office 17', 'Office 71']
    BuildingAbbreviations = ['MPEB', 'CH', '17', '71', 'Nothing']
    InputVariables = ['inputs_MaletPlace_FINAL.csv', 'inputs_CentralHouse_222_29_11_15_02_2870.csv', 'Inputs.csv', 'inputs_BH71_27_09_13_46.csv']
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
        runs_outputs = readRuns(parallel_runs_harddisk, time_step=time_step, NO_ITERATIONS=NO_ITERATIONS)
        print(runs_outputs.head())
        runs_outputs.to_hdf(DataPath_model_real + building_abr+ '_' + str(NO_ITERATIONS) +'_RUNS_' + time_step + '.hdf', 'runs', mode='w')

    runs_outputs = pd.read_hdf(DataPath_model_real + building_abr+ '_' + str(NO_ITERATIONS) +'_RUNS_' + time_step + '.hdf', 'runs')
    Y_real = runs_outputs.as_matrix()
    cols_outputs = runs_outputs.columns.tolist()

    runs_inputs = readInputs(DataPath_model_real, parallel_runs_harddisk, inputs, NO_ITERATIONS)
    X_real = runs_inputs.as_matrix()
    cols_inputs = runs_inputs.columns.tolist()

    # X_real_data = np.vstack((cols_inputs, X_real))
    # Y_real_data = np.vstack((cols_outputs, Y_real))
    #input_outputs = np.hstack((X_real_data, Y_real_data))
    #pd.DataFrame(input_outputs).to_csv(DataPath_model_real + 'input_outputs_' + time_step + '.csv', header=None)

    df_corr_stnd, df_corr_spearman, df_corr_pearson = calculateCorrelations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs)

    #heatmapCorrelations(df_corr_pearson, runs_outputs, DataPathImages)
    #scatterCorrelation(runs_inputs[['WeekdayLandPsched_Offset']], runs_outputs[['Cooling']]/floor_area, input_label='L&P profile offset (per 30Min)', output_label='Cooling $\mathregular{(kWh/m^{2}a)}$', DataPathImages)
    boxplotPredictions(runs_outputs/floor_area, time_step, DataPathImages)

    plt.show()



def calculateCorrelations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs):
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

        if p == 0:
            df_corr_stnd = df_corr
        elif p == 1:
            df_corr_spearman = df_corr
        elif p == 2:
            df_corr_pearson = df_corr

    return df_corr_stnd, df_corr_spearman, df_corr_pearson

def scatterCorrelation(df_inputs, df_outputs, input_label, output_label, DataPathImages):

    df_in = df_inputs.columns.tolist()
    df_out = df_outputs.columns.tolist()
    for i, v in enumerate(range(len(df_in))):

        input = df_inputs[df_in[i]]
        output = df_outputs[df_out[i]]

        fig = plt.figure(figsize=(6/ 2.54, 6 / 2.54))
        ax = fig.add_subplot(111)

        reorder = sorted(range(len(input)), key = lambda ii: input[ii])
        xd = [input[ii] for ii in reorder]
        yd = [output[ii] for ii in reorder]
        par = np.polyfit(xd, yd, 1, full=True)

        slope=par[0][0]
        intercept=par[0][1]
        xl = [min(xd), max(xd)]
        yl = [slope*xx + intercept  for xx in xl]

        # coefficient of determination, plot text
        variance = np.var(yd)
        residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
        Rsqr = np.round(1-residuals/variance, decimals=2)

        # error bounds
        yerr = [abs(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)]
        par = np.polyfit(xd, yerr, 2, full=True)

        yerrUpper = [(xx*slope+intercept)+(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(xd,yd)]
        yerrLower = [(xx*slope+intercept)-(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(xd,yd)]

        ax.plot(xl, yl, '-', color=colors[1])
        ax.plot(xd, yerrLower, '--', color=colors[1])
        ax.plot(xd, yerrUpper, '--', color=colors[1])

        max_dots = 500
        ax.scatter(df_inputs[df_in[i]][:max_dots], df_outputs[df_out[i]][:max_dots], alpha=.8)
        #ax.plot(x, m*x + b, '-')
        #ax.set_xlim(0, ),

        ax.set_xlabel(input_label)
        ax.set_ylabel(output_label)
        ax.set_title('$R^2 = %0.2f$'% Rsqr, fontsize=9)
    plt.savefig(DataPathImages + '_ScatterSingleVariable.png', dpi=300, bbox_inches='tight')

def boxplotPredictions(runs, time_step, DataPathImages): # for multiple runs
    """
    :param runs: Pandas DataFrame of predictions (i.e. combined eplusmtr results)
    :param time_step: 'month' or 'year'
    :return:
    """

    if time_step == 'year':
        no_end_uses = len(runs.columns)
        fig = plt.figure(figsize=(18 / 2.54, 8 / 2.54)) #width and height
        ax2 = plt.subplot2grid((1, no_end_uses+1), (0, 0))
        ax = plt.subplot2grid((1, no_end_uses+1), (0, 1), colspan=no_end_uses)

        #plot end-uses boxplots
        x = np.arange(1, len(runs.columns) + 1)
        bplot = runs.iloc[:, :].plot.box(ax=ax,  widths=.85, showfliers=False, patch_artist=True, return_type='dict') #showmeans=True
        colors_repeat = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in colors))

        for y in range(runs.shape[0]):
            if y < 250: # otherwise it gets too crowded
                q = np.random.normal(0, 0.06, size=runs.shape[1])
                ax.scatter(x+q, runs.iloc[y, :], edgecolors='r', alpha=0.05, zorder=5, facecolors='none',)

        #plot total boxplot
        bplot_ax2 = pd.DataFrame(runs.sum(axis=1), columns=['Total']).plot.box(ax=ax2, widths=.85, showfliers=False, patch_artist=True, return_type='dict', )

        for y in range(pd.DataFrame(runs.sum(axis=1)).shape[0]):
            if y < 500:
                q = np.random.normal(0, 0.06)
                ax2.scatter(1+q, pd.DataFrame(runs.sum(axis=1)).iloc[y, :], edgecolors='r', alpha=0.1, zorder=5, facecolors='none', )

        bplots = [bplot, bplot_ax2]
        for bplot in bplots:
            [i.set(color=colors[0], linewidth=1.5) for i in bplot['boxes']]
            [i.set(facecolor='white') for i in bplot['boxes']]
            for key in ['whiskers', 'caps', 'medians']:
                for y, box in enumerate(bplot[key]): #set colour of boxes
                        box.set(color=colors[0], linewidth=1.5)
            [i.set(color='black') for i in bplot['medians']]

        fig.subplots_adjust(wspace=1)

        ax2.set_ylabel('Energy $\mathregular{(kWh/m^{2}a)}$')
        ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
        ax2.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
        ax.set_axisbelow(True)
        ax2.set_axisbelow(True)

    if time_step == 'month':
        cols = runs.columns.tolist()

        runs_total = runs.sum(axis=1, level=[1]) # sum the months for each end-use
        runs_total.columns = pd.MultiIndex.from_product([['Total'], runs_total.columns]) # add new level total and columns
        runs = pd.concat([ runs, runs_total], axis=1) #add total to multiindex

        print(runs.head())

        end_uses = runs.columns.levels[0].tolist()
        print(runs[end_uses[0]].columns)
        month_list = runs[end_uses[0]].columns.tolist()

        ticklabels=month_list

        fig, axes = plt.subplots(nrows=len(end_uses), ncols=1, sharey=False, figsize=(18 / 2.54, len(end_uses)*3.5 / 2.54))

        end_uses.remove('Total')
        end_uses.append('Total') #total to end

        for x, y in enumerate(end_uses):
            ax = axes[x]

            props = dict(boxes=colors[0], whiskers=colors[0], medians='black', caps=colors[0])
            runs.xs(y, axis=1).plot.box(ax=ax, color=props, patch_artist=True, showfliers=False)  # access highlevel multiindex

            #hide month labels for all but last plot
            ax.set_ylabel(y)
            if x != len(end_uses)-1:
                for index, label in enumerate(ax.get_xaxis().get_ticklabels()):
                    label.set_visible(False)

            ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
            ax.set_axisbelow(True)

        axes[0].set_title('Energy $\mathregular{(kWh/m^{2}a)}$', fontsize=9)
        axes[len(end_uses)-1].xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

        plt.savefig(DataPathImages + time_step + '_boxplot.png', dpi=300, bbox_inches='tight')

def heatmapCorrelations(df, runs, DataPathImages):
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

    use_sns = True
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
    def start__main__():
        print('start')




    start__main__()

    from read_predictions import readRuns, readInputs

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    UserName = getpass.getuser()
    if UserName == 'cvdronke':
        start_path = 'C:/Users/' + UserName + '/'
    else:
        start_path = 'D:/'

    main()