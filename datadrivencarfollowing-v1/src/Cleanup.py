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
        loc = df['Location'][0]
        v_Class_verify = df[['Vehicle_ID', 'v_Class']]
        v_Class_verify.drop_duplicates(inplace=True)
        v_Class_verify = v_Class_verify.groupby(
            ['Vehicle_ID'], as_index=False).count()
        remove_bad_data_vehicles = set(
            v_Class_verify[v_Class_verify["v_Class"] > 1]['Vehicle_ID'])
        bad_v_Class_length = df[(
            df['Vehicle_ID'].isin(remove_bad_data_vehicles))]
        lf_pair_remove_first_last_5_seconds_lane1, lf_pair_remove_first_last_5_seconds_lane2, lf_pair_remove_first_last_5_seconds_lane3, lf_pair_remove_first_last_5_seconds_lane4, lf_pair_remove_first_last_5_seconds_lane5, lf_pair_remove_first_last_5_seconds_lane6, lf_pair_remove_first_last_5_seconds_lane7, lf_pair_remove_first_last_5_seconds_lane8 = self.lane_change_info(
            df)

        both_lane_change_1 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 1) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_2 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 2) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_3 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 3) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_4 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 4) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_5 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 5) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_6 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 6) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_7 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 7) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        both_lane_change_8 = df[(df['L-F_Pair'].isin(lf_pair_remove_first_last_5_seconds_lane1)) & (df['Lane_ID'] == 8) &
                                ((df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        #time_headway_less_than5 = df[(df['Time_Headway'] <= 5)]
        time_headway_less_than5 = df[(df['Time_Headway'] <= 1)]
        remove_ramp_data = df[(df['Lane_ID'] == 7) | (df['Lane_ID'] == 8)]
        '''
        both_lane_change = df[(df['preceding_car_lane_changes'] == True) & (df['lane_changes'] == True) & (
            (df['pair_Time_Duration'] <= 5) | (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5)))]
        lead_change = df[(df['preceding_car_lane_changes'] == True) & (
            df['lane_changes'] == False) & ((df['pair_Time_Duration'] <= 5))]
        subject_change = df[(df['preceding_car_lane_changes'] == False) & (
            df['lane_changes'] == True) & (df['pair_Time_Duration'] >= (df['total_pair_duration'] - 5))]
        total_duration_less_than_minute = df[(
            df['total_pair_duration'] < 60)]
        '''

        total_duration_less_than_minute = df[(
            df['total_pair_duration'] < 30)]

        before = df.shape[0]
        print(f"dataset before Row Removal{df.shape}")

        print(
            f"{loc}: {both_lane_change_1.shape[0]} vehicles first and Last 5 seconds removed from Lane 1 due to lane changing")
        print(
            f"{loc}: {both_lane_change_2.shape[0]} vehicles first and Last 5 seconds removed from Lane 2 due to lane changing")
        print(
            f"{loc}: {both_lane_change_3.shape[0]} vehicles first and Last 5 seconds removed from Lane 3 due to lane changing")
        print(
            f"{loc}: {both_lane_change_4.shape[0]} vehicles first and Last 5 seconds removed from Lane 4 due to lane changing")
        print(
            f"{loc}: {both_lane_change_5.shape[0]} vehicles first and Last 5 seconds removed from Lane 5 due to lane changing")
        print(
            f"{loc}: {both_lane_change_6.shape[0]} vehicles first and Last 5 seconds removed from Lane 6 due to lane changing")
        print(
            f"{loc}: {both_lane_change_7.shape[0]} vehicles first and Last 5 seconds removed from Lane 7 due to lane changing")
        print(
            f"{loc}: {both_lane_change_8.shape[0]} vehicles first and Last 5 seconds removed from Lane 8 due to lane changing")
        print(
            f"{loc}: {bad_v_Class_length.shape[0]} have multiple lengths and classes for same Vehicle ID")
        print(
            f"{loc}: {total_duration_less_than_minute.shape[0]} have total car following less than 30 seconds")
        print(
            f"{loc}: {time_headway_less_than5.shape[0]} have time Headway less than 5 seconds")
        print(
            f"{loc}: {remove_ramp_data.shape[0]} have Lane ID as 7 or 8 which are the Ramps")

        # remove = pd.concat([both_lane_change, lead_change,
        #                    subject_change, total_duration_less_than_minute, bad_v_Class_length])
        remove = pd.concat(
            [time_headway_less_than5, total_duration_less_than_minute, bad_v_Class_length, both_lane_change_1, both_lane_change_2, both_lane_change_3, both_lane_change_4, both_lane_change_5, both_lane_change_6, both_lane_change_7, both_lane_change_8, remove_ramp_data])
        df.drop(labels=remove.index, inplace=True)
        after = df.shape[0]
        removed_row_count = before - after
        print(f"{loc}: {removed_row_count} rows removed using above criterias")
        # df['Vehicle_ID'].isna().sum()

        return df

    def following_lane_change_data(self, df):
        lane_verify = df[['Vehicle_ID', 'Lane_ID', 'Following']]
        #print(f"{lane_verify.duplicated().sum()} duplicate values have been removed")
        lane_verify.drop_duplicates(inplace=True)
        lane_verify.sort_values(['Vehicle_ID', 'Lane_ID'])
        lane_verify_1 = lane_verify[((lane_verify['Lane_ID'] == 1))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_2 = lane_verify[((lane_verify['Lane_ID'] == 2))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_3 = lane_verify[((lane_verify['Lane_ID'] == 3))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_4 = lane_verify[((lane_verify['Lane_ID'] == 4))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_5 = lane_verify[((lane_verify['Lane_ID'] == 5))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_6 = lane_verify[((lane_verify['Lane_ID'] == 6))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_7 = lane_verify[((lane_verify['Lane_ID'] == 7))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_8 = lane_verify[((lane_verify['Lane_ID'] == 8))].drop(
            columns='Lane_ID', axis=1)

        lf_pair_remove_first_5_seconds_lane1 = self.find_following_lane_change_pairs(
            lane_verify_1)
        lf_pair_remove_first_5_seconds_lane2 = self.find_following_lane_change_pairs(
            lane_verify_2)
        lf_pair_remove_first_5_seconds_lane3 = self.find_following_lane_change_pairs(
            lane_verify_3)
        lf_pair_remove_first_5_seconds_lane4 = self.find_following_lane_change_pairs(
            lane_verify_4)
        lf_pair_remove_first_5_seconds_lane5 = self.find_following_lane_change_pairs(
            lane_verify_5)
        lf_pair_remove_first_5_seconds_lane6 = self.find_following_lane_change_pairs(
            lane_verify_6)
        lf_pair_remove_first_5_seconds_lane7 = self.find_following_lane_change_pairs(
            lane_verify_7)
        lf_pair_remove_first_5_seconds_lane8 = self.find_following_lane_change_pairs(
            lane_verify_8)
        return lf_pair_remove_first_5_seconds_lane1,    lf_pair_remove_first_5_seconds_lane2,    lf_pair_remove_first_5_seconds_lane3,    lf_pair_remove_first_5_seconds_lane4,    lf_pair_remove_first_5_seconds_lane5,    lf_pair_remove_first_5_seconds_lane6,    lf_pair_remove_first_5_seconds_lane7,    lf_pair_remove_first_5_seconds_lane8

    def find_following_lane_change_pairs(self, lane_verify_df):
        lane_change_vehicles = lane_verify_df.sort_values(
            ['Vehicle_ID']).groupby('Vehicle_ID').count()

        vehicles_with_preceding_change_lane = lane_change_vehicles[
            lane_change_vehicles['Following'] > 1].index.values

        df_reqd = lane_verify_df[lane_verify_df['Vehicle_ID'].isin(
            vehicles_with_preceding_change_lane)]
        df_reqd['lf_pair'] = df_reqd['Vehicle_ID'].astype(
            str) + '-' + df_reqd['Following'].astype(str)
        lf_pair_remove_first_5_seconds_lane = df_reqd['lf_pair']
        return lf_pair_remove_first_5_seconds_lane

    def preceding_lane_change_data(self, df):
        lane_verify = df[['Vehicle_ID', 'Lane_ID', 'Preceding']]
        #print(f"{lane_verify.duplicated().sum()} duplicate values have been removed")
        lane_verify.drop_duplicates(inplace=True)
        lane_verify.sort_values(['Vehicle_ID', 'Lane_ID'])
        lane_verify_1 = lane_verify[((lane_verify['Lane_ID'] == 1))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_2 = lane_verify[((lane_verify['Lane_ID'] == 2))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_3 = lane_verify[((lane_verify['Lane_ID'] == 3))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_4 = lane_verify[((lane_verify['Lane_ID'] == 4))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_5 = lane_verify[((lane_verify['Lane_ID'] == 5))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_6 = lane_verify[((lane_verify['Lane_ID'] == 6))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_7 = lane_verify[((lane_verify['Lane_ID'] == 7))].drop(
            columns='Lane_ID', axis=1)
        lane_verify_8 = lane_verify[((lane_verify['Lane_ID'] == 8))].drop(
            columns='Lane_ID', axis=1)

        lf_pair_remove_first_5_seconds_lane1 = self.find_preceding_lane_change_pairs(
            lane_verify_1)
        lf_pair_remove_first_5_seconds_lane2 = self.find_preceding_lane_change_pairs(
            lane_verify_2)
        lf_pair_remove_first_5_seconds_lane3 = self.find_preceding_lane_change_pairs(
            lane_verify_3)
        lf_pair_remove_first_5_seconds_lane4 = self.find_preceding_lane_change_pairs(
            lane_verify_4)
        lf_pair_remove_first_5_seconds_lane5 = self.find_preceding_lane_change_pairs(
            lane_verify_5)
        lf_pair_remove_first_5_seconds_lane6 = self.find_preceding_lane_change_pairs(
            lane_verify_6)
        lf_pair_remove_first_5_seconds_lane7 = self.find_preceding_lane_change_pairs(
            lane_verify_7)
        lf_pair_remove_first_5_seconds_lane8 = self.find_preceding_lane_change_pairs(
            lane_verify_8)
        return lf_pair_remove_first_5_seconds_lane1,    lf_pair_remove_first_5_seconds_lane2,    lf_pair_remove_first_5_seconds_lane3,    lf_pair_remove_first_5_seconds_lane4,    lf_pair_remove_first_5_seconds_lane5,    lf_pair_remove_first_5_seconds_lane6,    lf_pair_remove_first_5_seconds_lane7,    lf_pair_remove_first_5_seconds_lane8

    def find_preceding_lane_change_pairs(self, lane_verify_df):
        lane_change_vehicles = lane_verify_df.sort_values(
            ['Vehicle_ID']).groupby('Vehicle_ID').count()

        vehicles_with_preceding_change_lane = lane_change_vehicles[
            lane_change_vehicles['Preceding'] > 1].index.values

        df_reqd = lane_verify_df[lane_verify_df['Vehicle_ID'].isin(
            vehicles_with_preceding_change_lane)]
        df_reqd['lf_pair'] = df_reqd['Preceding'].astype(
            str) + '-' + df_reqd['Vehicle_ID'].astype(str)
        lf_pair_remove_first_5_seconds_lane = df_reqd['lf_pair']
        return lf_pair_remove_first_5_seconds_lane

    def lane_change_info(self, df):
        lf_pair_remove_first_5_seconds_lane1,    lf_pair_remove_first_5_seconds_lane2,    lf_pair_remove_first_5_seconds_lane3,    lf_pair_remove_first_5_seconds_lane4,    lf_pair_remove_first_5_seconds_lane5,    lf_pair_remove_first_5_seconds_lane6,    lf_pair_remove_first_5_seconds_lane7,    lf_pair_remove_first_5_seconds_lane8 = self.preceding_lane_change_data(
            df)
        lf_pair_remove_first_5_seconds_lane1_f,    lf_pair_remove_first_5_seconds_lane2_f,    lf_pair_remove_first_5_seconds_lane3_f,    lf_pair_remove_first_5_seconds_lane4_f,    lf_pair_remove_first_5_seconds_lane5_f,    lf_pair_remove_first_5_seconds_lane6_f,    lf_pair_remove_first_5_seconds_lane7_f,    lf_pair_remove_first_5_seconds_lane8_f = self.following_lane_change_data(
            df)

        lf_pair_remove_first_last_5_seconds_lane1 = lf_pair_remove_first_5_seconds_lane1.append(
            lf_pair_remove_first_5_seconds_lane1_f)
        lf_pair_remove_first_last_5_seconds_lane2 = lf_pair_remove_first_5_seconds_lane2.append(
            lf_pair_remove_first_5_seconds_lane2_f)
        lf_pair_remove_first_last_5_seconds_lane3 = lf_pair_remove_first_5_seconds_lane3.append(
            lf_pair_remove_first_5_seconds_lane3_f)
        lf_pair_remove_first_last_5_seconds_lane4 = lf_pair_remove_first_5_seconds_lane4.append(
            lf_pair_remove_first_5_seconds_lane4_f)
        lf_pair_remove_first_last_5_seconds_lane5 = lf_pair_remove_first_5_seconds_lane5.append(
            lf_pair_remove_first_5_seconds_lane5_f)
        lf_pair_remove_first_last_5_seconds_lane6 = lf_pair_remove_first_5_seconds_lane6.append(
            lf_pair_remove_first_5_seconds_lane6_f)
        lf_pair_remove_first_last_5_seconds_lane7 = lf_pair_remove_first_5_seconds_lane7.append(
            lf_pair_remove_first_5_seconds_lane7_f)
        lf_pair_remove_first_last_5_seconds_lane8 = lf_pair_remove_first_5_seconds_lane8.append(
            lf_pair_remove_first_5_seconds_lane8_f)

        return lf_pair_remove_first_last_5_seconds_lane1,    lf_pair_remove_first_last_5_seconds_lane2,    lf_pair_remove_first_last_5_seconds_lane3,    lf_pair_remove_first_last_5_seconds_lane4,    lf_pair_remove_first_last_5_seconds_lane5,    lf_pair_remove_first_last_5_seconds_lane6,    lf_pair_remove_first_last_5_seconds_lane7,    lf_pair_remove_first_last_5_seconds_lane8

    def remove_dup_pairs(self, df):
        before = df.shape[0]

        verify_pairs = df[['L-F_Pair', 'Location']]
        verify_pairs.drop_duplicates(inplace=True)
        verify_pairs = verify_pairs.groupby(
            ['L-F_Pair'], as_index=False).count()
        duplicate_pairs = set(
            verify_pairs[verify_pairs["Location"] > 1]['L-F_Pair'])
        remove_dup_pairs = df[df['L-F_Pair'].isin(
            duplicate_pairs) & df['Location'] == 'i-80']
        df.drop(labels=remove_dup_pairs.index, inplace=True)
        after = df.shape[0]
        removed_row_count = before - after
        print(f"{removed_row_count} rows removed for Dups. Count Before removal: {before}, Count After Removal:{after}")
        return df
