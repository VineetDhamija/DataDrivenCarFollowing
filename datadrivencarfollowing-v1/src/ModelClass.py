from asyncio.windows_events import NULL
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
import tensorflow
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import FileProcessing

import warnings
warnings.filterwarnings("ignore")


class ModelClass():

    '''

    '''
    saved_pairs = []

    def preprocessing(self, input_df, time_frame):
        input_df = self.create_prediction_pair(input_df, time_frame)
        train_df, val_df, test_df = self.test_train_pairs(input_df, 0.9)
        X_train, y_train, X_val, y_val, X_test, y_test = self.feature_selection(
            train_df, val_df, test_df)
        return input_df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test

    def fit_and_run_neural(self, df, time_frame):
        shift_instance = time_frame*10
        df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
            df, shift_instance)
        model = self.define_neural_network(X_train)
        model = self.fit_neural_network(
            model, X_train, y_train, X_val, y_val, time_frame)
        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
        predict_on_pair[0]
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'

        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, time_frame)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]

        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', time_frame)
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_velocity', 'v_Vel', 'Velocity', time_frame)
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_spacing', 'Rear_to_Front_Space_Headway', 'Spacing', time_frame)

        return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model

    def fit_and_run_random_forest(self, df, time_frame):
        shift_instance = time_frame*10
        df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
            df, shift_instance)
        # define_fit_random_forest_model(self, train_df, test_df, regressors):
        regressors = 25
        rf = self.define_fit_random_forest_model(
            regressors, train_df,  X_train, y_train, X_val, y_val)
        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)

        predict_on_pair[0]
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'

        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, time_frame)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]

        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', time_frame)
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_velocity', 'v_Vel', 'Velocity', time_frame)
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_spacing', 'Rear_to_Front_Space_Headway', 'Spacing', time_frame)

        return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model

    def plot_prediction(self, df, col_x, predicted_y, actual_y, name, time_frame):
        plt.figure(figsize=(11, 9))
        label1 = "Actual" + str(name) + "Value"
        label2 = "Predicted" + str(name) + "Value"
        title_value = str(name) + \
            " : Actual vs Fitted Values for Reaction Time: " + str(time_frame)
        ax = sns.lineplot(x=df[col_x], y=df[actual_y], color="r", label=label1)
        sns.lineplot(x=df[col_x], y=df[predicted_y], color="b", label=label2)
        plt.title(title_value)
        plt.show()
        plt.close()
        return None

    def define_neural_network(self, input_df):

        # input = keras.Input(shape=(18,))
        input_df = tensorflow.expand_dims(input_df, axis=-1)

        input = keras.Input(shape=(input_df.shape[1], 1))

        x = layers.Conv1D(filters=16, kernel_size=(
            2), padding='same', activation="sigmoid", name='Block1_Conv1')(input)
        x = layers.Conv1D(filters=16, kernel_size=(
            2), padding='same', activation="sigmoid", name='Block1_Conv2')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2, name='Block1_Pool')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="tanh", name='Block2_Conv1')(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="tanh", name='Block2_Conv2')(x)
        x = layers.MaxPool1D(pool_size=2, strides=2, name='Block2_Pool')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.05)(x)
        # prework for fully connected layer.
        x = layers.Flatten()(x)
        # Fully connected layers
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(64, activation='sigmoid')(x)
        x = layers.Dense(16, activation='tanh')(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=input, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        model.summary()

        return model

    def define_fit_random_forest_model(self, regressors, X_train, y_train, X_val, y_val):
        '''
        train/fit the model on train dataset. Also assign inputs variables to X_train, X_test, y_train, and y_test
        X_train = train_df[['Rear_to_Front_Space_Headway', 'Vehicle_combination_cat',
                            'Local_Y', 'Velocity Difference_Following-Preceding', 'v_Vel']]
        y_train = train_df['nextframeAcc']
        X_test = test_df[['Rear_to_Front_Space_Headway', 'Vehicle_combination_cat',
                          'Local_Y', 'Velocity Difference_Following-Preceding', 'v_Vel']]
        y_test = test_df['nextframeAcc']
        '''
        rf = RandomForestRegressor(n_estimators=regressors, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf

    def fit_neural_network(self, model, X_train, y_train, X_val, y_val, reaction_time):
        modelName = "neural_network_model" + str(reaction_time) + ".keras"
        save_callback = keras.callbacks.ModelCheckpoint(
            modelName, save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', verbose=1, patience=7)
        history = model.fit(X_train, y_train, epochs=10, batch_size=16,
                            verbose=1, validation_data=(X_val, y_val), callbacks=[save_callback, early_stopping])
        # convertingt the accuracy of the model to a graph.
        # the dictionary that has the information on loss and accuracy per epoch
        history_dict = history.history

        loss_values = history_dict['loss']   # training loss
        val_loss_values = history_dict['val_loss']  # validation loss

        # creates list of integers to match the number of epochs of training
        epochs = range(1, len(loss_values)+1)

        # code to plot the results
        plt.plot(epochs, loss_values, 'b', label="Training Loss")
        plt.plot(epochs, val_loss_values, 'r', label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.xticks(epochs)
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        # As above, but this time we want to visualize the training and validation accuracy
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']

        plt.plot(epochs, acc_values, 'b', label="Training Accuracy")
        plt.plot(epochs, val_acc_values, 'r', label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(epochs)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        return model

    def create_prediction_pair(self, df, n):
        '''
        create the prediction pair by shifting the actual data up by the mentioned number(0.1*n seconds) to create the timeseries info
        '''
        df = self.tranformations_for_models(df)
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

    def tranformations_for_models(self, df):
        '''
        Read the input file into a dataframe.
        Input: File name for the file present in Data folder.
        Output: Dataframe name.
        '''
#        df["Vehicle_combination_cat"] = df["Vehicle_combination"].astype("category").cat.codes
        df["Location_cat"] = df["Location"].astype("category").cat.codes

        drop_columns_list = ['Vehicle_ID', 'Frame_ID', 'Global_Time', 'Local_X', 'Global_X', 'Global_Y', 'v_length', 'Lane_ID', 'Preceding', 'Space_Headway',
                             'Time_Headway', 'v_Class_Name', 'lane_changes', 'preceding_car_lane_changes', 'Prec_Vehicle_ID', 'Vehicle_combination', 'Preceding_Vehicle_Class', 'Relative_Time']
        df["Front_To_Rear_Time_Headway"] = df["Front_To_Rear_Time_Headway"].replace(
            np.inf, 999)
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(
            np.float64).astype(np.float32)
        df = df.drop(drop_columns_list, axis=1, errors='ignore')
        return df

    def test_train_pairs(self, df, split, seed=0):
        '''
        Read the input file into a dataframe.
        Input: File name for the file present in Data folder.
        Output: Dataframe name.
        '''

        if seed > 0:
            random.seed(seed)
        # df = df.applymap(lambda x: float(round(x, 4))if isinstance(x, (int, float)) else x)

        total_pairs = df["L-F_Pair"].unique()
        total_pairs = total_pairs.tolist()
        test_split_cnt = round(len(total_pairs)*split)
        test_split_pairs = random.sample(total_pairs, test_split_cnt)
        train_df = df[df['L-F_Pair'].isin(test_split_pairs)]
        test_df = df[~df['L-F_Pair'].isin(test_split_pairs)]

        validation_split_cnt = round(test_split_cnt*0.2)
        validation_split_pairs = random.sample(
            test_split_pairs, validation_split_cnt)
        val_df = df[df['L-F_Pair'].isin(validation_split_pairs)]
        train_df = df[~df['L-F_Pair'].isin(validation_split_pairs)]

        return train_df, val_df, test_df

    def prediction_test_pairs(self, df, pair_from, pair_to):
        unique_pairs_values = df['L-F_Pair'].unique()
        unique_pairs_list = unique_pairs_values.tolist()
        unique_pairs_df = unique_pairs_list[pair_from:pair_to]
        return unique_pairs_df

    def feature_selection(self, train_df, val_df, test_df):
        features = ['Rear_to_Front_Space_Headway', 'preceding_v_Class', "v_Class",
                    'Velocity Difference_Following-Preceding', 'v_Vel', 'Location_cat']
        X_train = train_df[features]
        X_val = val_df[features]
        X_test = test_df[features]

        y_train = train_df['nextframeAcc']
        y_val = val_df['nextframeAcc']
        y_test = test_df['nextframeAcc']

        # scaler = StandardScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test
    '''features:
    'Rear_to_Front_Space_Headway',
    'preceding_v_Class',
    "v_Class",
    'Velocity Difference_Following-Preceding',
    'v_Vel',
    'Location_cat']]
    predicted_data = prediction(
        test_df, predict_on_pair, target_variable, model,0.1)
    '''

    def prediction(self, test_df, test_range, target_variable, model, time_frame):
        predicted_df = []
        # this loop runs for each pair required predictions.
        for current_pair in test_range:
            # Assign shape of the predictions
            input_df = []
            input_df = test_df[test_df['L-F_Pair'] == current_pair]
            spacing = np.zeros(input_df.shape[0])
            local_y_subject = np.zeros(input_df.shape[0])
            local_y_preceding = np.zeros(input_df.shape[0])
            dv = np.zeros(input_df.shape[0])
            vel = np.zeros(input_df.shape[0])
            pred_acc = np.zeros(input_df.shape[0])

            # updating the values for first Predictions
            vel[0] = input_df.iloc[0]['v_Vel']
            spacing[0] = input_df.iloc[0]['Rear_to_Front_Space_Headway']
            dv[0] = input_df.iloc[0]['Velocity Difference_Following-Preceding']

            pred_acc[0] = input_df.iloc[0]['v_Acc']
            local_y_subject[0] = input_df.iloc[0]['Local_Y']
            local_y_preceding[0] = input_df.iloc[0]['preceding_Local_Y']
            preceding_vehicle_class = input_df.iloc[0]['preceding_v_Class']
            vehicle_class = input_df.iloc[0]['v_Class']
            # vehicle_combination= input_df.iloc[0]['Vehicle_combination_cat']
            length_preceding_vehicle = input_df.iloc[0]['preceding_vehicle_length']
            location = input_df.iloc[0]['Location_cat']

            # predicting first value of acceleration

            predict_for_input = np.array(
                [spacing[0], preceding_vehicle_class, vehicle_class, dv[0], vel[0], location]).reshape(1, -1)
            pred_acc[1] = model.predict(predict_for_input)
            # calculating vel,frspacing,local.y,dv from the predicted acceleration.

            for j in range(1, len(input_df)):
                # v= u + at
                vel[j] = vel[j-1]+(pred_acc[j-1]*time_frame)

            # dv=current velocity of subject - velocity of Lead/Preceding
                dv[j] = vel[j] - input_df.iloc[j]['preceding_Vehicle_Velocity']

            # distance travelled by vehicle. :s
            # s = ut + 0.5*a*t^2
                s_subject = ((vel[j-1]*time_frame) +
                             (0.5*pred_acc[j-1]*pow(time_frame, 2)))

                # s_preceding = ((input_df.iloc[j-1]['preceding_Vehicle_Velocity']*time_frame) + (
                #    0.5*input_df.iloc[j-1]['preceding_Vehicle_Acceleration']*pow(time_frame, 2)))
                #spacing_calc = spacing[j-1] + s_preceding - s_subject
                local_y_subject[j] = local_y_subject[j-1] + s_subject
                # spacing[j] = spacing[j-1]+ s_lead- s_subject
                local_y_preceding[j] = input_df.iloc[j-1]['preceding_Local_Y']
                spacing_calc = local_y_preceding[j] - \
                    local_y_subject[j] - length_preceding_vehicle
                # print(f"s_subject: {s_subject},local_y_subject:{local_y_subject[j]},local_y_preceding: {local_y_preceding[j]},spacing[j]:{spacing[j]}")
                if spacing_calc < 0:
                    spacing[j] = 0
                else:
                    spacing[j] = spacing_calc

                # print(
                #    f"s_subject: {s_subject},s_preceding:{s_preceding},previous spacing: {spacing[j-1]},spacing[j]:{spacing[j]}")

                # as we are predicting the next values, we should not predict for the last one.
                if j == len(input_df)-1:
                    break
                # if j == 5: # this is temporary
                #    break

                predict_for_input = np.array(
                    [spacing[j], preceding_vehicle_class, vehicle_class, dv[j], vel[j], location]).reshape(1, -1)
                # pred_acc[j+1] = model.predict(np.array([spacing[j],vehicle_combination,local_y[j],dv[j],vel[j]]))
                pred_acc[j+1] = model.predict(predict_for_input)
                print(
                    f"j: {j},predict_for_input:{predict_for_input},pred_acc: {pred_acc[j+1]}")

                ########
                # print(pred_acc)
                ########

            print(f"input_df shape: {input_df.shape}")
            print(f"pred_acc shape: {pred_acc.shape}")
            input_df['predicted_acceleration'] = pred_acc
            input_df['predicted_velocity'] = vel
            input_df['predicted_spacing'] = spacing

            predicted_df.append(input_df)
            result = pd.concat(predicted_df)
        return result
