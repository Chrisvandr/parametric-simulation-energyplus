import re
import os
import pandas as pd
import numpy as np

def readGas(DataPath, building, building_num, write_data, datafile, floor_area):
    dateparse = lambda x: pd.datetime.strptime(x, '%d-%b-%y')
    print('importing gas data from:', DataPath + building + '/Data/' + datafile + '_SubmeteringData.csv')

    if building_num == 1:  # Central House
        df = pd.read_csv(DataPath + building + '/Data/' + datafile + '_GasData.csv', date_parser=dateparse,
                         header=0, index_col=0)

        df = df.loc['2013-01-01':'2016-10-01']  # ['2015-09-31':'2016-10-01']  ['2012-01-24':'2016-10-01']
        df = df.groupby(df.index.month).mean()  # get the monthly mean over multiple years
        df = pd.concat([df[9:], df[:9]])  # reorder the months to align with the submetered data...
        rng = pd.date_range(start='09/2016', end='09/2017', freq='M')
        df = df.set_index(rng)  # set new index to align mean monthly gas data with metered electricity
        df.rename(columns={df.columns[0]: 'Gas'}, inplace=True)

    return df

def readSTM(DataPathSTM, building, building_num, write_data, datafile, floor_area):
    """ Short Term Monitoring """
    if building_num in {0}:
        dateparseSTM = lambda x: pd.datetime.strptime(x, '%d-%m-%y %H:%M')
    elif building_num in {1}:
        dateparseSTM = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')

    if building_num in {0,1}:
        df_stm = pd.read_csv(DataPathSTM + datafile + '/' + datafile + '_combined.csv', date_parser=dateparseSTM, header=0,index_col=0)
    else:
        df_stm = pd.DataFrame()


    cols = df_stm.columns.tolist()

    if building_num == 0:  # MaletPlaceEngineering
        cols_new = ['Server GF [GP3]', 'Lighting 2nd', 'Power 2nd', 'Lighting 3rd', 'Power 3rd', 'DB1', 'Lighting 6th',
                    'Power 6th', 'Power 7th', 'Lighting 7th']

        for i, v in enumerate(cols):
            df_stm.rename(columns={cols[i]: cols_new[i]}, inplace=True)

    if building_num == 1:  # CentralHouse
        cols_new = ['MCP01', 'B2', 'PV', '3A', '3D', 'B41']

        for i, v in enumerate(cols):
            df_stm.rename(columns={cols[i]: cols_new[i]}, inplace=True)

    """ Manipulate """
    # Created average and standard deviation profiles for the weekday and weekendday for short term monitoring. Interpolated the values to half-hour based on 2hour metering data.
    df_stm = df_stm.divide(8)  # because it's kWh each value needs to be divided by 8 if we go from 2h to 15min frequency
    df_stm = df_stm[~df_stm.index.duplicated(keep='first')]
    df_stm = df_stm.reindex(pd.date_range(start=df_stm.index.min(), end=df_stm.index.max(), freq='15Min'))
    df_stm = df_stm.interpolate(method='linear')

    return df_stm

def readSubmetering(DataPath, building, building_num, building_abr, write_data, datafile, df_stm, floor_area):
    print(building_abr)
    print('importing submetering data from:', DataPath + building + '/Data/' + datafile + '_SubmeteringData.csv')
    if building_abr == 'MPEB':  # Malet Place Engineering
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%b-%d %H:%M:%S.000')
        df = pd.read_csv(DataPath + building + '/Data/' + datafile + '_SubmeteringData.csv', date_parser=dateparse,
                         header=0, index_col=0)
        # Check if there are duplicate index values (of which there are in CH data) and remove them...
        df = df[~df.index.duplicated(keep='first')]
        # There are missing indices in the data, reindex the missing indices of which there are only a few and backfill them
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='15Min'), method='backfill')
        df_realweather = pd.DataFrame()
        cols = df.columns.tolist()

        df = df.loc['2016-09-01 00:15:00':'2017-08-31']  # ['2016-09-01':'2017-04-30']
        cols_new = ['B10', 'B11', 'B12', 'B14', 'B15', 'B16', 'B17', 'B8', 'B9', 'BB4', 'BB3',
                    'CH1', 'CH2', 'DB5MS', 'GP2', 'Dynamo', 'Fire Lift', 'Lift Panel',
                    'Lift P1', 'LV1', 'LV2', 'LV3', 'MCCB03', 'MCCB01', 'BB2', 'BB1']

        # print(pd.DataFrame([df.columns, cols_new]).T)
        # stm cols [server excluded...,'2L','2P','3L','3P','DB1','6L','6P','7P','7L']
        for i, v in enumerate(cols):
            df.rename(columns={cols[i]: cols_new[i]}, inplace=True)

        df_stm = pd.concat([df_stm], axis=1, join_axes=[df.index])  # set the short term monitoring to the same axis
        df = pd.concat([df, df_stm], axis=1)

        # df = df.convert_objects(convert_numeric=True).fillna(0)
        df[df < 0] = 0  # set negative values (LV3) to 0

        df_MPEB = pd.concat([df[['LV1', 'LV2']]], axis=1)
        df_MPEB = df_MPEB.sum(axis=1)
        df_MPEB = df_MPEB.sum(axis=0)
        print('df_MPEB total kWh/m2a:', df_MPEB / floor_area)

        # real LV metered
        df_LV1_real = df['LV1']
        df_LV2_real = df['LV2']
        df_mains = pd.DataFrame(pd.concat([df_LV1_real, df_LV2_real], axis=1).sum(axis=1), columns=['Mains'])

        df_workshops = pd.DataFrame(pd.concat([df[['BB4', 'BB3', 'LV3', 'GP2']]], axis=1).sum(axis=1),
                                    columns=['Workshops'])
        df_lifts = pd.DataFrame(pd.concat([df[['Fire Lift', 'Lift Panel']]], axis=1).sum(axis=1), columns=['Lifts'])
        df_mech = pd.DataFrame(pd.concat([df[['MCCB03', 'Dynamo', 'MCCB01']]], axis=1).sum(axis=1), columns=['Systems'])
        df_chillers = pd.DataFrame(pd.concat([df[['CH1', 'CH2']]]).sum(axis=1), columns=['Chillers'])

        # Lighting and Power
        df_BB2 = pd.DataFrame(pd.concat([df[['Lighting 6th', 'Power 6th', 'Lighting 7th', 'Power 7th']].sum(axis=1),
                                         pd.DataFrame(df[['Lighting 7th', 'Power 6th']].sum(axis=1) * 3)], axis=1).sum(
            axis=1), columns=['BB2 L&P'])

        df_BB1 = pd.DataFrame(pd.concat([df[['Lighting 2nd', 'Power 2nd', 'Lighting 3rd', 'Power 3rd']].sum(axis=1),
                                         pd.DataFrame(df[['Lighting 6th', 'Power 6th']].sum(axis=1) * 2)], axis=1).sum(
            axis=1), columns=['BB1 L&P'])

        df_BB1_surplus = df['BB1'] - df['DB1'] - df['Server GF [GP3]'] - df_BB1['BB1 L&P']
        df_BB2_surplus = df['BB2'] - df_BB2['BB2 L&P']
        print('Busbar 1')
        print('BB1', df['BB1'].sum(axis=0) / floor_area)
        print('DB1', df['DB1'].sum(axis=0) / floor_area)
        print('GP3', df['Server GF [GP3]'].sum(axis=0) / floor_area)
        print('BB1 L&P', df_BB1['BB1 L&P'].sum(axis=0) / floor_area)
        print('BB1remaining', df_BB1_surplus.sum(axis=0) / floor_area)

        print('LP on 6th', pd.DataFrame(df[['Lighting 6th', 'Power 6th']]).sum(axis=1).sum(axis=0) / floor_area)
        print('LP on 2 and 3rd',
              df[['Lighting 2nd', 'Power 2nd', 'Lighting 3rd', 'Power 3rd']].sum(axis=1).sum(axis=0) / floor_area)

        print('Busbar 2')
        print('BB2', df['BB2'].sum(axis=0) / floor_area)
        print('BB2 L&P', df_BB2['BB2 L&P'].sum(axis=0) / floor_area)
        print('BB2remaining', df_BB2_surplus.sum(axis=0) / floor_area)

        print(((df_BB1_surplus.sum(axis=0) / floor_area) + (df_BB2_surplus.sum(axis=0) / floor_area)) / (
        df['DB1'].sum(axis=0) / floor_area))
        print(((df_BB1_surplus.sum(axis=0) / floor_area) + (df_BB2_surplus.sum(axis=0) / floor_area)) / df['DB1'].sum(
            axis=0) / floor_area)
        df_lp = pd.DataFrame(pd.concat([df_BB1['BB1 L&P'], df_BB2['BB2 L&P']], axis=1).sum(axis=1),
                             columns=['floors L&P'])
        surplus_basedonDB1 = df['DB1'] * ((((df_BB1_surplus.sum(axis=0) / floor_area) + (
        df_BB2_surplus.sum(axis=0) / floor_area)) / (df['DB1'].sum(axis=0) / floor_area)) / 10)
        # keep within 20% of the mean.
        surplus_basedonDB1[
            surplus_basedonDB1 < surplus_basedonDB1.mean() - 0.2 * surplus_basedonDB1.mean()] = surplus_basedonDB1.mean()  # remove negative values..
        surplus_basedonDB1[
            surplus_basedonDB1 > surplus_basedonDB1.mean() + 0.2 * surplus_basedonDB1.mean()] = surplus_basedonDB1.mean()  # remove negative values..
        df_BB1and2 = pd.DataFrame(
            df[['BB1', 'BB2']].sum(axis=1) - surplus_basedonDB1 - df['Server GF [GP3]'] - df['DB1'], columns=['L&P'])

        # scaled_daily(df_BB1and2.resample('30Min').sum(), building_label='MPEB', building_abr='MPEB', day_type='three', scale=False, time_interval='30Min')

        surplus = pd.concat([df_BB1_surplus + df_BB2_surplus], axis=1)

        # determine server based on difference between LV2 and dissaggregated LV2.
        df_LV2_aggregate = pd.concat([df[['BB1', 'BB2', 'CH2', 'MCCB01', 'GP2']]],
                                     axis=1)  # LV2, missing Fire alam and DB409 (big server)
        df_LV2_aggregate = df_LV2_aggregate.sum(axis=1)
        df_bigserver = pd.DataFrame(df_LV2_real - df_LV2_aggregate, columns=[
            'DB409'])  # difference between LV2 and LV2 dissaggregated is the difference, which should be the server.
        df_bigserver[df_bigserver < 0] = 0  # remove negative values...
        df_bigserver = pd.DataFrame(
            pd.concat([df_bigserver, surplus_basedonDB1, df['Server GF [GP3]'], df['DB1']], axis=1).sum(axis=1),
            columns=['DB409'])

        print(df_bigserver.sum(axis=0) / floor_area, 'kWh/m2a')

        df_floorsLP = pd.DataFrame(pd.concat([df[['BB1', 'BB2']]], axis=1).sum(axis=1), columns=['L&P'])
        df_floorsLP['L&P'] = df_floorsLP['L&P'] - df['Server GF [GP3]']

        df_floorsLP = pd.DataFrame(pd.concat([df_BB1, df_BB2], axis=1).sum(axis=1), columns=['L&P'])
        df_servers = pd.DataFrame(pd.concat([df_bigserver, df[['Server GF [GP3]']]], axis=1).sum(axis=1),
                                  columns=['Servers'])
        print("Average kWh per day for the server DB409 = " + str(df_bigserver.mean()))

        df_LVL1 = pd.concat([df_BB1and2, df_chillers, df_mech, df_servers, df_workshops],
                            axis=1)  # LV1, missing LV1A, PF LV1

        print('Workshops', df_workshops.values.sum() / floor_area, 'servers', df_servers.values.sum() / floor_area,
              'Lifts', df_lifts.values.sum() / floor_area)

        print('lift', df['Lift P1'].values.sum() / floor_area)
        print('GP2', df['GP2'].values.sum() / floor_area)
        print('DB5MS', df['DB5MS'].values.sum() / floor_area)

        # diff between BB3 aggregated and separate
        df_BB3 = df[['B9', 'B10', 'B14', 'B15', 'B8']]  # these combined form Busbar-2 (BB3)
        df_BB4 = df[['B12', 'B16', 'B17', 'B11']]  # these combined form Busbar-1 (BB4) # excludes B13
        df_BB3and4 = pd.concat([df_BB3, df_BB4], axis=1)
        df_BB3and4 = df_BB3and4.sum(axis=1)
        df_BB3and4real = pd.concat([df['BB2'], df['BB1']], axis=1)

        df = pd.concat([df, df_bigserver], axis=1)

    if building_abr == 'CH':  # CentralHouse
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%b-%d %H:%M:%S.000')
        df = pd.read_csv(DataPath + building + '/Data/' + datafile + '_SubmeteringData.csv', date_parser=dateparse,
                         header=0, index_col=0)
        # Check if there are duplicate index values (of which there are in CH data) and remove them...
        df = df[~df.index.duplicated(keep='first')]

        # There are missing indices in the data, reindex the missing indices of which there are only a few and backfill them
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='15Min'), method='backfill')
        df_realweather = pd.DataFrame()

        df = df.loc['2016-09-01':'2017-08-31']

        # Naming
        # cols_new = ['BB2', 'LIFT1', 'LIFT2', 'LIFT3', 'B1', 'B4', 'BB1', 'LIFT4', 'DB21', 'R1', 'Server [B5]', 'R2',
        #             'G2', 'G1', 'B31', 'B32', 'B44', 'B52', 'B54', 'B53',
        #             'B62', 'B64', 'B11', 'B12', 'B21', 'B22', 'B43', 'B42', 'B51',
        #             'B61', 'MP1']
        cols = df.columns.tolist()
        # STM ['MCP01', 'B2', 'PV', '3A', '3D', 'B41']
        cols_new = ['BB2', 'LIFT1', 'LIFT2', 'LIFT3', 'B1', 'B4', 'BB1', 'LIFT4', 'DB21', 'R1', 'Server', 'R2',
                    'G2', 'G1', 'B31', 'B32', 'B44', 'B52', 'B54', 'B53', 'B62', 'B64', 'B11', 'B12', 'B21', 'B22',
                    'B43', 'B42', 'B51',
                    'B61', 'MP1']

        for i, v in enumerate(cols):
            df.rename(columns={cols[i]: cols_new[i]}, inplace=True)

        df_m = df.resample('M').sum()
        df_m_sum = df_m.mean(axis=0)

        # combine B1 and B2 and B4 (boiler house L&P) as Basement L&P
        df_basementLP = pd.concat([df[['B1', 'B4']], df_stm[['B2']]], axis=1, join_axes=[df.index])
        df_basementLP = pd.DataFrame(df_basementLP.sum(axis=1), columns=['L&P Basement'])
        # combine L&P per floor
        df_groundLP = df[['G1', 'G2']]
        df_groundLP = pd.DataFrame(df_groundLP.sum(axis=1), columns=['L&P Ground floor'])
        # first floor lighting and power
        df_firstLP = df[['B12', 'B11']]
        df_firstLP = pd.DataFrame(df_firstLP.sum(axis=1), columns=['L&P 1st floor'])
        # second floor lighting and power
        df_secondLP = df[['B21', 'B22']]
        df_secondLP = pd.DataFrame(df_secondLP.sum(axis=1), columns=['L&P 2nd floor'])
        # third floor lighting and power
        df_thirdLP = pd.concat([df[['B31', 'B32']], df_stm[['3A', '3D']]], axis=1, join_axes=[df.index])
        df_thirdLP = pd.DataFrame(df_thirdLP.sum(axis=1), columns=['L&P 3rd floor'])
        # fourth floor lighting and power
        df_fourthLP = pd.concat([df[['B42', 'B43', 'B44']], df_stm[['B41']]], axis=1, join_axes=[df.index])
        df_fourthLP = pd.DataFrame(df_fourthLP.sum(axis=1), columns=['L&P 4th floor'])  # [B41, B42]
        # fifth floor lighting and power
        df_fifthLP = df[['B51', 'B53', 'B54']]
        df_fifthLP = pd.DataFrame(df_fifthLP.sum(axis=1), columns=['L&P 5th floor'])
        # sixth floor lighting and power
        df_sixthLP = df[['B61', 'B62']]
        df_sixthLP = pd.DataFrame(df_sixthLP.sum(axis=1), columns=['L&P 6th floor'])

        # combine Lifts 1-4
        df_lifts = pd.DataFrame(df[['LIFT1', 'LIFT2', 'LIFT3', 'LIFT4']].sum(axis=1), columns=['Lifts'])

        # combine R1, R2 and MCP01 as systems
        df_mech = pd.concat([df[['R1', 'R2']], df_stm[['MCP01']]], axis=1, join_axes=[df.index])
        df_mech = pd.DataFrame(df_mech.sum(axis=1), columns=['Systems'])

        df_BBs = pd.concat([df[['BB1', 'BB2']], df_basementLP], axis=1, join_axes=[df.index])
        df_BBs = df_BBs.sum(axis=1)
        df_BBs = pd.DataFrame(df_BBs)
        df_BBs.rename(columns={df_BBs.columns[0]: 'L&P'}, inplace=True)  # R1, R2', MCP01

        df_BB1 = df[['G1', 'B11', 'B21', 'B61', 'B42']]
        df_BB2 = pd.concat([df[['G2', 'B12', 'B22', 'B51', 'B62']], df_stm[['B41']]], axis=1, join_axes=[df.index])

        df_lighting = pd.concat([df[['B31', 'B62']], df_stm[['B41']]], axis=1, join_axes=[df.index])  # is this correct?

        df_MP1_real = df['MP1']

        df_floorsLP = pd.concat(
            [df_basementLP, df_groundLP, df_firstLP, df_secondLP, df_thirdLP, df_fourthLP, df_fifthLP, df_sixthLP],
            axis=1)  # B3 is not measured... (should be small)
        df_floorsLP_sum = pd.concat([df_floorsLP, df[['Server']], df_lifts], axis=1)
        df_floorsLP_sum = pd.DataFrame(df_floorsLP_sum.sum(axis=1), columns=['L&P'])

        df_LVL1 = pd.concat([df_floorsLP_sum, df_mech], axis=1,
                            join_axes=[df.index])  # B3 is not measured... (should be small)
        df_stm = pd.concat([df_stm], axis=1, join_axes=[df.index])
        df_mains = pd.DataFrame(
            pd.concat([df[['MP1']], df[['B31', 'B32']], df_stm[['3A', '3D']]], axis=1, join_axes=[df.index]).sum(
                axis=1), columns=['Mains'])

    if building_abr == '17':  # 17
        dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%y %H:%M')
        # for pilot study
        # df = pd.read_csv(DataPath + building + '/Data/17_actual_clean.csv', date_parser=dateparse, header=0, index_col=0)

        df = pd.read_csv(DataPath + building + '/Data/17_SubmeteringData.csv', date_parser=dateparse, header=0,
                         index_col=0)

        # Check if there are duplicate index values (of which there are in CH data) and remove them...
        df = df[~df.index.duplicated(keep='first')]
        # There are missing indices in the data, reindex the missing indices of which there are only a few and backfill them
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='30Min'), method='backfill')

        df_realweather = pd.DataFrame()

        df = df[:-1]
        cols = df.columns.tolist()

        print(df.columns)
        cols_new = ['Gas', 'B_Power', 'B_Lights', 'B_AC', 'Print', 'Canteen', 'GF_Lights', 'Servers', 'GF_Power',
                    '1st_Lights', 'GF_AC', 'Lift', '1st_Power', '2nd_Lights', '2nd_Power']

        for i, v in enumerate(cols):
            df.rename(columns={cols[i]: cols_new[i]}, inplace=True)

        # ['Gas', 'B_Power', 'B_Lights', 'B_AC', 'Print', 'Canteen', 'Server', 'GF_Power', 'GF_Lights', 'GF_AC', 'Lift', '1st_Power', '1st_Lights', '2nd_Power', '2nd_Lights']
        df_lights = pd.concat([df[['B_Lights', 'GF_Lights', '1st_Lights', '2nd_Lights']]], axis=1)
        df_lights = pd.DataFrame(df_lights.sum(axis=1), columns=['Lights'])

        df_mech = pd.concat([df[['B_AC', 'GF_AC']]], axis=1)
        df_mech = pd.DataFrame(df_mech.sum(axis=1), columns=['AC'])

        df_power = pd.concat([df[['B_Power', 'GF_Power', '1st_Power', '2nd_Power']]], axis=1)
        df_power = pd.DataFrame(df_power.sum(axis=1), columns=['Power'])

        # L&P
        df_floorsLP = pd.concat([df[['B_Power', 'B_Lights', 'GF_Power', 'GF_Lights', '1st_Power', '1st_Lights',
                                     '2nd_Power', '2nd_Lights']]], axis=1)  # B3 is not measured... (should be small)
        df_LVL1 = pd.concat([df_lights, df_power, df[['Gas', 'Servers', 'Canteen', 'Print']]], axis=1)

        df_mains = pd.DataFrame(
            pd.concat([df_lights, df_power, df[['Servers', 'Canteen', 'Print']]], axis=1).sum(axis=1),
            columns=['Mains'])

    if building_abr == '71':  # 71
        dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%y %H:%M')
        df = pd.read_csv(DataPath + building + '/Data/' + datafile + '_SubmeteringData.csv', date_parser=dateparse,
                         header=0, index_col=0)
        # Check if there are duplicate index values (of which there are in CH data) and remove them...
        df = df[~df.index.duplicated(keep='first')]
        # There are missing indices in the data, reindex the missing indices of which there are only a few and backfill them
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='30Min'), method='backfill')
        df_realweather = pd.DataFrame()

        df = df[:-1]
        df = df.loc['2012-04-01':'2017-10-31']

        cols = df.columns.tolist()

        # df = df.loc['2014-01-01':'2014-12-31']  # ['2016-09-01':'2017-04-30']
        cols_new = ['Gas', 'B_Power', 'Canteen', 'Lifts', 'GF_Power', 'GF_Lights', 'GF_AC', '1st_Power', '1st_Lights',
                    '1st_AC', '2nd_Power', '2nd_Lights', '2nd_AC', '3rd_Power', '3rd_Lights', '3rd_AC']
        # print(pd.DataFrame([df.columns, cols_new]).T)

        for i, v in enumerate(cols):
            df.rename(columns={cols[i]: cols_new[i]}, inplace=True)

        """Multiply gas by 100!"""
        df = pd.concat([df.iloc[:, 0].mul(100), df.iloc[:, 1:]], axis=1)

        df_lights = pd.concat([df[['GF_Lights', '1st_Lights', '2nd_Lights', '3rd_Lights']]], axis=1)
        df_lights = pd.DataFrame(df_lights.sum(axis=1), columns=['Lights'])

        df_mech = pd.concat([df[['GF_AC', '1st_AC', '2nd_AC', '3rd_AC']]], axis=1)
        df_mech = pd.DataFrame(df_mech.sum(axis=1), columns=['AC'])

        df_power = pd.concat([df[['B_Power', 'GF_Power', '1st_Power', '2nd_Power', '3rd_Power']], df_mech], axis=1)
        df_power = pd.DataFrame(df_power.sum(axis=1), columns=['Power'])

        # todo I have excluded the Canteen here because the model seems to not predict the zone for some reason (both the Canteen and the Kitchen within it). It broke at some point... (no time to remodel it)
        df_b_power = pd.DataFrame(pd.concat([df[['B_Power']]], axis=1).sum(axis=1), columns=['B_Power'])
        gf_power = pd.DataFrame(pd.concat([df[['GF_Power']]], axis=1).sum(axis=1), columns=['GF_Power'])
        first_power = pd.DataFrame(pd.concat([df[['1st_Power']]], axis=1).sum(axis=1), columns=['1st_Power'])
        second_power = pd.DataFrame(pd.concat([df[['2nd_Power']]], axis=1).sum(axis=1), columns=['2nd_Power'])
        third_power = pd.DataFrame(pd.concat([df[['3rd_Power']]], axis=1).sum(axis=1), columns=['3rd_Power'])

        df_LVL1 = pd.concat([df_power, df_lights, df[['Gas']]], axis=1)
        df_mains = pd.DataFrame(pd.concat([df_lights, df_power], axis=1).sum(axis=1), columns=['Mains'])
        # df_LVL1 = pd.concat([df[['GF_Lights', '1st_Lights', '2nd_Lights', '3rd_Lights']], gf_power, first_power, second_power, third_power, df_b_power, df[['Gas']]], axis=1)

        # L&P
        df_floorsLP = pd.concat([df[['B_Power', 'GF_Power', 'GF_Lights', '1st_Power', '1st_Lights', '2nd_Power',
                                     '2nd_Lights', '3rd_Power', '3rd_Lights']]],
                                axis=1)  # B3 is not measured... (should be small)

    if write_data == 'True':
        timeframe = ['A', 'M', 'd']
        for t, v in enumerate(timeframe):
            df_resample = df.resample(v).sum()  # resample year
            df_resample.to_csv(DataPath + building + '/Data/' + datafile + '_Submetering' + '_' + v + '.csv')
    else:
        print('Writing data set to:', write_data)

    return df, df_mains, df_LVL1, df_floorsLP, df_mech, df_stm, df_realweather
