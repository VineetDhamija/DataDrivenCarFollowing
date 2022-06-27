import numpy as np
import pandas as pd
from pathlib import Path


class Transformation():
    p = Path().cwd()
    stringpath = str(p)[0:str(p).rfind('\\')] + '\\data'

    def __init__(self):
        self.initialize()

    def readInput(self, fileName):
        '''

        '''
        print(f"original File path: {self.p}")
        print(f"Data File path: { self.stringpath}")
        ngsimfile = self.stringpath + '/' + fileName + '.csv'
        df = pd.read_csv(ngsimfile, low_memory=False)
        return df

    def trial_func(self):
        '''
        Function to verofy that the class works
        '''
        print(f"Transformation:Class called")  # duplicate values have been removed")

        # return True

    def convertFeetToMetre(self, df):
        '''
        Convert the input variables which exist in Feet to Metres or Feet/Second to Metre/Second and so on. 
        '''
        df['v_length'] = df['v_length']*0.3048
        df['Space_Headway'] = df['Space_Headway']*0.3048
        df['v_Vel'] = df['v_Vel']*0.3048
        df['v_Acc'] = df['v_Acc']*0.3048
        return df

    def dropNotRequiredColumns(self, df):
        '''
        Drop the columns that are not required for our Model 
        '''
        df = df.drop(columns=['Movement', 'Direction', 'Section_ID', 'Int_ID',
                     'D_Zone', 'O_Zone', 'Following', 'v_Width', 'Total_Frames'])
        return df

    def createColumnPlaceholders(self, df):
        '''
        Create PLaceholder for columns that will be generated and populated to use in the Model 
        '''
        df['Preceding_Vehicle_Class'] = np.NaN
        df['Rear_to_Front_Space_Headway'] = np.NaN
        df['Front_To_Rear_Time_Headway'] = np.NaN
        df['Velocity Difference_Following-Preceding'] = np.NaN
        df['Acceleration Difference_Following-Preceding'] = np.NaN
        return df

    def bifurcateHighways(self, df):
        '''
        filter out US-101 and I-80 into separate HIghway dataframes
        '''

        filtered_U = df[((df['Location'] == 'us-101'))]

        filtered_I = df[(df['Location'] == 'i-80')]

        return filtered_U, filtered_I

    def classVehicleSets(self, df):
        '''
        Create the set for the vehicle classes, Motorcycle, Car and Heavy Vehicle
        Input:Dataframe
        output: Vehicle ID set in sequence: 1. Motorcycle , 2. Car, 3. Heavy Vehicle
        '''

        filtered_vClass = df[['Vehicle_ID', 'v_Class',
                              'v_length']].drop_duplicates().sort_values('v_Class')
        v_Class_M = set(
            filtered_vClass[(filtered_vClass['v_Class'] == 1)]['Vehicle_ID'])
        v_Class_C = set(
            filtered_vClass[(filtered_vClass['v_Class'] == 2)]['Vehicle_ID'])
        v_Class_HV = set(
            filtered_vClass[(filtered_vClass['v_Class'] == 3)]['Vehicle_ID'])

        return v_Class_M, v_Class_C, v_Class_HV

    def precedingVehicleClass(self, df, v_Class_M, v_Class_C, v_Class_HV):
        '''
        Find and populate the preceding Vehicle Class Name into Preceding_Vehicle_Class column, and populate the Vehicle_combination as well. 
        Input: df,the three deciding Class sets of Motorcycle and Heavy Vehicle. 
        '''
        result = []

        for i in df.index:
            if df['Preceding'][i] in v_Class_C:
                result.append('Car')
            elif df['Preceding'][i] in v_Class_HV:
                result.append('Heavy Vehicle')
            elif df['Preceding'][i] in v_Class_M:
                result.append('Motorcycle')
            else:
                result.append('Free Flow')
        df['Preceding_Vehicle_Class'] = result
        df['Vehicle_combination'] = df['Preceding_Vehicle_Class'] + \
            '-' + df['v_Class_Name']
        return df

    def precedingVehicleLength(self, df):
        '''
        Find and populate the preceding Vehicle Length Name into preceding_vehicle_length column. If there is no Preceding vehicle, it will be populated with 0 in case of Vehicle ID 0
        Input: df
        Ouptut: df
        '''
        vehicle_lengths = df[['Vehicle_ID', 'v_length']]
        # print(f"{vehicle_lengths.duplicated().sum()} duplicate values have been removed")
        vehicle_lengths.drop_duplicates(inplace=True)
        # print(vehicle_lengths.shape)
        x = vehicle_lengths.groupby(['Vehicle_ID']).mean()
        dict = x.to_dict()['v_length']
        df["preceding_vehicle_length"] = df["Preceding"].map(dict)
        df["preceding_vehicle_length"] = df["preceding_vehicle_length"].fillna(
            0)
        return df

    def frontToFrontBumperDetailsChangedToRearToFrontBumperDetails(self, df):
        '''
        Change the details from the Front to Front Bumper to Rear of Lead to Front Bumper of Subject Vehicle.
        1. Space Headway
        2. Time Headway
        Input: df
        Ouptut: df
        '''
        df['Rear_to_Front_Space_Headway'] = df['Space_Headway'] - \
            df['preceding_vehicle_length']
        df['Front_To_Rear_Time_Headway'] = df['Rear_to_Front_Space_Headway'] / df['v_Vel']
        return df

    def mapPreviousVehicleDetails(self, df):
        '''
        Change the details from the Front to Front Bumper to Rear of Lead to Front Bumper of Subject Vehicle.
            1. Space Headway
            2. Time Headway
        Input: 
            df
        Ouptut: 
            df
        '''
        lane_verify = df[['Vehicle_ID', 'Lane_ID']]
        # print(f"{lane_verify.duplicated().sum()} duplicate values have been removed")
        lane_verify.drop_duplicates(inplace=True)
        lane_verify = lane_verify.groupby(
            ['Vehicle_ID'], as_index=False).count()
        # print(lane_verify.shape)
        lane_change_vehicles = set(
            lane_verify[lane_verify["Lane_ID"] > 1]['Vehicle_ID'])
        df['lane_changes'] = df['Vehicle_ID'].isin(
            lane_change_vehicles)
        print(
            f"{df['Location']}::{df[(df['lane_changes'] == False) ]['Vehicle_ID'].unique().size} cars dont change lanes")
        print(
            f"{df['Location']}::{df[(df['lane_changes'] == True) ]['Vehicle_ID'].unique().size} cars Change lanes")
        right_df = df[['Preceding', 'Relative_Time',
                       'v_Vel', 'v_Acc', 'lane_changes']]
        right_df.rename(columns={'Preceding': 'Prec_Vehicle_ID', 'v_Vel': 'previous_Vehicle_Velocity',
                        'v_Acc': 'previous_Vehicle_Acceleration', "lane_changes": "previous_car_lane_changes"}, inplace=True)
        df['Prec_Vehicle_ID'] = df['Vehicle_ID']
        df = df.merge(right=right_df, how='left', on=(
            'Prec_Vehicle_ID', 'Relative_Time'))
        df['previous_Vehicle_Velocity'] = df['previous_Vehicle_Velocity'].fillna(
            0)
        df['previous_Vehicle_Acceleration'] = df['previous_Vehicle_Acceleration'].fillna(
            0)
        df['previous_car_lane_changes'] = df['previous_car_lane_changes'].fillna(
            False)
        return df

    def mapPairs(self, df):
        '''
        Map the Pairs for the Preceding and lead vehicle. 
        Input: 
            df
        Ouptut: 
            df
        '''
        df['Velocity Difference_Following-Preceding'] = df['v_Vel'] - \
            df['previous_Vehicle_Velocity']
        df['Acceleration Difference_Following-Preceding'] = df['v_Acc'] - \
            df['previous_Vehicle_Acceleration']
        df = df.sort_values(by=['Relative_Time'],
                            ascending=True, ignore_index=True)
        df['pair_Time_Duration'] = (df.groupby(
            ['L-F_Pair'], as_index=False).cumcount()*0.1)
        x = (df[['L-F_Pair', 'pair_Time_Duration']].groupby(['L-F_Pair'],
                                                            as_index=False).max(['pair_Time_Duration']))
        dict_lenght = dict(zip(x['L-F_Pair'], x['pair_Time_Duration']))
        df["total_pair_duration"] = df["L-F_Pair"].map(dict_lenght)
        print(df["total_pair_duration"].dtype)
        return df

    def filterRecordsForModel(self, df):
        '''
        Map the Pairs for the Preceding and lead vehicle. 
        Input: 
            df
        Ouptut: 
            df
        '''
        total_duration_less_than_minute = df[(
            df['total_pair_duration'] >= 6)]
        total_duration_less_than_minute.index
        both_lane_change = df[(df['previous_car_lane_changes'] == True) & (df['lane_changes'] == True) & (
            (df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        lead_change = df[(df['previous_car_lane_changes'] == True) & (
            df['lane_changes'] == False) & ((df['pair_Time_Duration'] <= 5))]
        subject_change = df[(df['previous_car_lane_changes'] == False) & (
            df['lane_changes'] == True) & (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5))]
        total_duration_less_than_minute = df[(
            df['total_pair_duration'] < 60)]
        before = df.shape[0]
        print(df.shape)
        print(both_lane_change.shape)
        print(lead_change.shape)
        print(subject_change.shape)
        remove = pd.concat([both_lane_change, lead_change,
                           subject_change, total_duration_less_than_minute])

        df.drop(labels=remove.index, inplace=True)
        after = df.shape[0]
        removed_row_count = after - before
        print(f"{removed_row_count} rows removed for the first and last 5 seconds of the cars that changed lanes")
        df['Vehicle_ID'].isna().sum()

        return df


def mergeFiles(self, df1, df2):
    '''
    Map the Pairs for the Preceding and lead vehicle. 
    Input: 
        df
    Ouptut: 
        df
    '''
    df = pd.concat([df1, df2])
    print(df1.shape[0])
    print(df2.shape[0])
    print(df.shape[0])
    return df


def exportFile(self, df, fileName):
    '''
    Map the Pairs for the Preceding and lead vehicle. 
    Input: 
        df
    Ouptut: 
        df
    '''
    stringpath = str(p)[0:str(p).rfind('\\')] + '\\data'
    ngsimfilteredfile = stringpath + '\\' + fileName + '.csv'
    df.to_csv(ngsimfilteredfile, index=False)
