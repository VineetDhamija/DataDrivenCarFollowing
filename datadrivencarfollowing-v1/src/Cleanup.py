import numpy as np
import pandas as pd


class Cleanup():
    # def __init__(self):
    #    self.initialize()

    def trial_func(self):
        '''
        Function Removes Duplicates from the Dataframe
        '''
        print(f"Class called")  # duplicate values have been removed")
        return True

    def remove_dups(self, df):
        '''
        Remove Duplicates.
        Input: 
            df
        Output:
            df
        '''
        print(f"{df.duplicated().sum()} duplicate values have been removed")
        df.drop_duplicates(inplace=True)
        return df

    def filter_records_for_model(self, df):
        '''
        Map the Pairs for the Preceding and lead vehicle. 
        Input: 
            df
        Ouptut: 
            df
        '''
        v_Class_verify = df[['Vehicle_ID', 'v_Class']]
        v_Class_verify.drop_duplicates(inplace=True)
        v_Class_verify = v_Class_verify.groupby(
            ['Vehicle_ID'], as_index=False).count()
        remove_bad_data_vehicles = set(
            v_Class_verify[v_Class_verify["v_Class"] > 1]['Vehicle_ID'])
        bad_v_Class_length = df[(
            df['Vehicle_ID'].isin(remove_bad_data_vehicles))]

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
        print(f"dataset before Row Removal{df.shape}")
        print(f"{both_lane_change.shape} Lead and Subject both change lane, so first and last 5 seconds of trajectory removed")
        print(f"{lead_change.shape} Lead Vehicle changes Lane, so first 5 seconds of car following Removed. ")
        print(
            f"{bad_v_Class_length} have multiple lengths and classes for same Vehicle ID")
        print(f"{subject_change.shape} subject vehicles change lanes so last 5 seconds of vehicle trajectory removed")
        remove = pd.concat([both_lane_change, lead_change,
                           subject_change, total_duration_less_than_minute, bad_v_Class_length])

        df.drop(labels=remove.index, inplace=True)
        after = df.shape[0]
        removed_row_count = before - after
        print(f"{removed_row_count} rows removed for the first and last 5 seconds of the cars that changed lanes")
        df['Vehicle_ID'].isna().sum()

        return df
