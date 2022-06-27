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

        # return True

    def remove_dups(df):
        '''
        Define the function to perform duplicate removal
        '''
        print(f"{df.duplicated().sum()} duplicate values have been removed")
        df.drop_duplicates(inplace=True)
        # return True


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


clean = Cleanup()
clean = ['abc']
clean.trial_func()
