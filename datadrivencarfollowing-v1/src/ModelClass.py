import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
#from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
#from tensorflow.keras.models import Sequential
from tensorflow import keras
#import tensorflow
from tensorflow.keras import layers


class ModelClass():

    '''
    '''

    def fit_neural_network(input):

        input = keras.Input
        # Fully connected layers
        x = layers.Dense(128, activation='relu')(input)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        # Softmax used for classifier problems
        outputs = layers.Dense(1, activation="softmax")(x)

        model = keras.Model(inputs=input, outputs=outputs)
        model.compile(optimizer='rmsprop',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        model.summary()
        return model

    def create_prediction_pair(self, df, n):
        '''
        create the prediction pair by shifting the actual data up by the mentioned number(0.1*n seconds) to create the timeseries info
        '''

        df["nextframeAcc"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Acc"].shift(-1*n)
        df["nextframesvel"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Vel"].shift(-1*n)
        df["nextframeposition"] = df.groupby(
            ["L-F_Pair"], as_index=False)["Local_Y"].shift(-1*n)
        df['nextframeposition'] = df['nextframeposition'].fillna(0)
        df['nextframesvel'] = df['nextframesvel'].fillna(0)
        df['nextframeAcc'] = df['nextframeAcc'].fillna(0)

        return df

    def test_train_pairs(self, df, split, seed):
        '''
        Read the input file into a dataframe.
        Input: File name for the file present in Data folder.
        Output: Dataframe name.
        '''
        random.seed(seed)
        total_pairs = df["L-F_Pair"].unique()
        total_pairs = total_pairs.tolist()
        test_split_cnt = round(len(total_pairs)*split)
        test_split_pairs = random.sample(total_pairs, test_split_cnt)
        train_df = df[df['L-F_Pair'].isin(test_split_pairs)]
        test_df = df[~df['L-F_Pair'].isin(test_split_pairs)]
        return train_df, test_df

    def prediction_pairs(self, df, pair_from, pair_to):
        unique_pairs_values = df['LF_pairs'].unique()
        unique_pairs_list = unique_pairs_values.tolist()
        unique_pairs_df = unique_pairs_list[pair_from:pair_to]
        return unique_pairs_df

    def fit_random_forest_model(self, train_df, test_df, regressors):
        X_train = train_df[['Rear_to_Front_Space_Headway', 'Vehicle_combination_cat',
                            'Local_Y', 'Velocity Difference_Following-Preceding', 'v_Vel']]
        y_train = train_df['nextframeAcc']
        X_test = test_df[['Rear_to_Front_Space_Headway', 'Vehicle_combination_cat',
                          'Local_Y', 'Velocity Difference_Following-Preceding', 'v_Vel']]
        y_test = test_df['nextframeAcc']
        rf = RandomForestRegressor(n_estimators=regressors, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf, X_train, y_train, X_test, y_test

    def prediction(test, b, target_variable, rf):
        F_df = []
        r = []
        Q = pd.DataFrame()
        # b is the test range
        for i in b:
            # Q this is the input data frame
            Q = test[test['L-F_Pair'] == i]
            vel = np.zeros(Q.shape[0])
            Vehicle_combination = np.zeros(Q.shape[0])
            Local_Y = np.zeros(Q.shape[0])
            spacing = np.zeros(Q.shape[0])
            dv = np.zeros(Q.shape[0])

            pred_acc = np.zeros(Q.shape[0])

            # adding first value of the vehicle
            vel[0] = Q.iloc[0]['v_Vel']
            spacing[0] = Q.iloc[0]['Rear_to_Front_Space_Headway']
            Vehicle_combination[0] = Q.iloc[0]['Vehicle_combination_cat']
            Local_Y[0] = Q.iloc[0]['Local_Y']
            dv[0] = Q.iloc[0]['Velocity Difference_Following-Preceding']

            pred_acc[0] = Q.iloc[1][target_variable]

        #       #predicting first value of acceleration
            # check here
            pred_acc[1] = rf.predict(np.array(
                [vel[0], Vehicle_combination[0], Local_Y[0], dv[0], spacing[0]]).reshape(1, -1))

        #     #calculating speed from the predicted acceleration.
            # check here

            for j in range(2, len(Q)):

                vel[j] = vel[j-1]+(pred_acc[j]*0.1)
                dv[j] = vel[j] - Q.iloc[j]['previous_Vehicle_Velocity']
                # spacing[j] = spacing[j-1]+((vel[j-1]*0.1)+ (0.5*pred_acc[j]*pow(0.1,2)))
                spacing[j] = (vel[j-1]*0.1) + (0.5*pred_acc[j]*pow(0.1, 2))
                Local_Y[j] = Local_Y[j-1]
                Vehicle_combination[j] = Vehicle_combination[j-1]

                if j == len(Q)-1:
                    break
                pred_acc[j+1] = rf.predict(np.array(
                    [Vehicle_combination[j], Local_Y[j], vel[j], dv[j], spacing[j]]).reshape(1, -1))

        Q['pacc'] = pred_acc
        Q['pvel'] = vel

        F_df.append(Q)
        result = pd.concat(F_df)
        return result

    def predict_cnn(test, b, target_variable, rf):
        F_df = []
        r = []
        Q = pd.DataFrame()
        # b is the test range
        for i in b:
            # Q this is the input data frame
            Q = test[test['L-F_Pair'] == i]
            vel = np.zeros(Q.shape[0])
            Vehicle_combination = np.zeros(Q.shape[0])
            Local_Y = np.zeros(Q.shape[0])
            spacing = np.zeros(Q.shape[0])
            dv = np.zeros(Q.shape[0])

            pred_acc = np.zeros(Q.shape[0])

            # adding first value of the vehicle
            vel[0] = Q.iloc[0]['v_Vel']
            spacing[0] = Q.iloc[0]['Rear_to_Front_Space_Headway']
            Vehicle_combination[0] = Q.iloc[0]['Vehicle_combination_cat']
            Local_Y[0] = Q.iloc[0]['Local_Y']
            dv[0] = Q.iloc[0]['Velocity Difference_Following-Preceding']

            pred_acc[0] = Q.iloc[1][target_variable]

        #       #predicting first value of acceleration
            # check here
            pred_acc[1] = rf.predict(np.array(
                [vel[0], Vehicle_combination[0], Local_Y[0], dv[0], spacing[0]]).reshape(1, -1))

        #     #calculating speed from the predicted acceleration.
            # check here

            for j in range(2, len(Q)):

                vel[j] = vel[j-1]+(pred_acc[j]*0.1)
                dv[j] = vel[j] - Q.iloc[j]['previous_Vehicle_Velocity']
                # spacing[j] = spacing[j-1]+((vel[j-1]*0.1)+ (0.5*pred_acc[j]*pow(0.1,2)))
                spacing[j] = (vel[j-1]*0.1) + (0.5*pred_acc[j]*pow(0.1, 2))
                Local_Y[j] = Local_Y[j-1]
                Vehicle_combination[j] = Vehicle_combination[j-1]

                if j == len(Q)-1:
                    break
                pred_acc[j+1] = rf.predict(np.array(
                    [Vehicle_combination[j], Local_Y[j], vel[j], dv[j], spacing[j]]).reshape(1, -1))

        Q['pacc'] = pred_acc
        Q['pvel'] = vel

        F_df.append(Q)
        result = pd.concat(F_df)
        return result

    def accuracy(self, F):
        mae_score = mean_absolute_error(F['V_Acc'], F['pacc'])
        r2_scores = r2_score(F['V_Acc'], F['pacc'])
        return mae_score, r2_score
