import re
import os
import pandas as pd
import numpy as np

def readRuns(parallel_runs_harddisk, time_step, NO_ITERATIONS):
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

def readInputs(DataPath_model_real, parallel_runs_harddisk, inputs, NO_ITERATIONS):
    """
    Read .csv file with inputs for each run created by create_idfs.py
    Return df of input parameter values and names per run.
    :param DataPath_model_real: path to .csv file
    :param parallel_runs_harddisk: path to dir containing eplusmtr.csv files.
    :param inputs: name of .csv file
    :param NO_ITERATIONS: no. iterations to take from file.
    :return: return parameter values per run as a DataFrame, header contains the input parameter names.
    """
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

    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)

    df = df.ix[run_numbers]
    df = df.drop(df.columns[[0]], axis=1)  # drop run_no column

    return df