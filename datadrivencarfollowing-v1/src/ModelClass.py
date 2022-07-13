from tensorflow.keras import layers
from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling2D
import tensorflow
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import joblib
import FileProcessing
import ModelClass

import warnings
warnings.filterwarnings("ignore")


class ModelClass():

    '''

    '''

    def fit_and_run_neural(self, df, time_frame):
        shift_instance = time_frame*10
        df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
            df, shift_instance, True)
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
        self.display_prediction_plots(prediction_1, time_frame, 'CNN ')

        '''
        #        self.display_prediction_plots(prediction_1, delta_time, 
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', time_frame, )
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_velocity', 'nextframesvel', 'Velocity', time_frame, 'CNN')
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_spacing', 'nextFrameSpacing', 'Spacing', time_frame, 'CNN')
        '''
        return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model

    def preprocessing(self, input_df, time_frame, neural=False):
        input_df = self.create_prediction_columns(input_df, time_frame)
        if neural:
            train_df, val_df, test_df = self.test_train_pairs(input_df, 0.9)
        else:
            train_df, val_df, test_df = self.test_train_pairs(input_df, 0.7)
        X_train, y_train, X_val, y_val, X_test, y_test = self.feature_selection(
            train_df, val_df, test_df)
        return input_df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test

    def test_train_pairs(self, df, split, neural=False, seed=0):
        '''
        Read the input file into a dataframe.
        Input: File name for the file present in Data folder.
        Output: Dataframe name.
        '''

        if seed > 0:
            random.seed(seed)

        total_pairs = df["L-F_Pair"].unique()
        total_pairs = total_pairs.tolist()
        test_split_cnt = round(len(total_pairs)*split)
        test_split_pairs = random.sample(total_pairs, test_split_cnt)
        train_df = df[df['L-F_Pair'].isin(test_split_pairs)]
        test_df = df[~df['L-F_Pair'].isin(test_split_pairs)]
        if neural:
            validation_split_cnt = round(test_split_cnt*0.2)
        else:
            validation_split_cnt = round(test_split_cnt*0.0)

        validation_split_pairs = random.sample(
            test_split_pairs, validation_split_cnt)
        val_df = df[df['L-F_Pair'].isin(validation_split_pairs)]
        train_df = df[~df['L-F_Pair'].isin(validation_split_pairs)]

        return train_df, val_df, test_df

    def feature_selection(self, train_df, val_df, test_df):
        features = ['Rear_to_Front_Space_Headway', 'preceding_v_Class', "v_Class",
                    'Velocity Difference_Following-Preceding', 'v_Vel']
        X_train = train_df[features]
        X_val = val_df[features]
        X_test = test_df[features]

        y_train = train_df['nextframeAcc']
        y_val = val_df['nextframeAcc']
        y_test = test_df['nextframeAcc']

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_prediction_columns(self, df, n):
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
        df["nextFrameSpacing"] = df.groupby(
            ["L-F_Pair"], as_index=False)["Rear_to_Front_Space_Headway"].shift(-1*n)
        df["precnextframeposition"] = df.groupby(
            ["L-F_Pair"], as_index=False)["preceding_Local_Y"].shift(-1*n)
        df["precnextframesvel"] = df.groupby(
            ["L-F_Pair"], as_index=False)["preceding_Vehicle_Velocity"].shift(-1*n)
        df = df[df['nextframeposition'].notna()]
        df = df[df['nextframesvel'].notna()]
        df = df[df['nextframeAcc'].notna()]
        df = df[df['nextFrameSpacing'].notna()]

        return df

    def tranformations_for_models(self, df):
        '''
        Read the input file into a dataframe.
        Input: File name for the file present in Data folder.
        Output: Dataframe name.
        '''
        drop_columns_list = ['Vehicle_ID', 'Frame_ID', 'Global_Time', 'Local_X', 'Global_X', 'Global_Y', 'v_length', 'Lane_ID', 'Preceding', 'Space_Headway',
                             'Time_Headway', 'v_Class_Name', 'lane_changes', 'preceding_car_lane_changes', 'Prec_Vehicle_ID', 'Preceding_Vehicle_Class', 'Relative_Time', 'total_pair_duration', 'total_pair_dur', 'diference', 'Front_To_Rear_Time_Headway']
        df["Front_To_Rear_Time_Headway"] = df["Front_To_Rear_Time_Headway"].replace(
            np.inf, 999)
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(
            np.float64).astype(np.float32)
        df = df.drop(drop_columns_list, axis=1, errors='ignore')
        return df

    def prediction_test_pairs(self, df, pair_from, pair_to, vehicle_combination=''):
        if vehicle_combination > '':
            df = df[(df['Vehicle_combination'] == vehicle_combination)]
        unique_pairs_values = df['L-F_Pair'].unique()
        unique_pairs_list = unique_pairs_values.tolist()
        if pair_to == 9999:
            unique_pairs_df = unique_pairs_list
        else:
            unique_pairs_df = unique_pairs_list[pair_from:pair_to]

        return unique_pairs_df

    def define_neural_network(self, input_df):

        input_df = tensorflow.expand_dims(input_df, axis=-1)

        input = keras.Input(shape=(input_df.shape[1], 1))

        x = layers.Conv1D(filters=16, kernel_size=(
            2), padding='same', activation="sigmoid", name='Block1_Conv1')(input)
        x = layers.Conv1D(filters=16, kernel_size=(
            2), padding='same', activation="sigmoid", name='Block1_Conv2')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="elu", name='Block2_Conv1')(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="elu", name='Block2_Conv2')(x)

        x = layers.Dropout(0.05)(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="tanh", name='Block3_Conv1')(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="tanh", name='Block3_Conv2')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.05)(x)
        # prework for fully connected layer.
        x = layers.Flatten()(x)
        # Fully connected layers
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(64, activation='sigmoid')(x)
        x = layers.Dense(16, activation='tanh')(x)
        outputs = layers.Dense(1, activation="elu")(x)

        model = keras.Model(inputs=input, outputs=outputs)

        model.compile(optimizer="adam",
                      loss="mean_squared_error",
                      metrics=["accuracy"])
        model.summary()

        return model

    def fit_neural_network(self, model, X_train, y_train, X_val, y_val, reaction_time):
        modelName = "neural_network_model" + str(reaction_time) + ".keras"
        save_callback = keras.callbacks.ModelCheckpoint(
            modelName, save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', verbose=1, patience=7)
        history = model.fit(X_train, y_train, epochs=10, batch_size=16,
                            verbose=1, validation_data=(X_val, y_val), callbacks=[save_callback, early_stopping])

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

    def plot_prediction(self, df, col_x, predicted_y, actual_y, name, time_frame, modelname):
        plt.figure(figsize=(11, 9))
        label1 = "Actual" + str(name) + "Value"
        label2 = "Predicted" + str(name) + "Value"
        title_value = str(modelname) + str(name) + \
            " : Actual vs Fitted Values for Reaction Time: " + str(time_frame)
        ax = sns.lineplot(x=df[col_x], y=df[actual_y], color="r", label=label1)
        sns.lineplot(x=df[col_x], y=df[predicted_y],
                     color="b", label=label2, ci=None)
        plt.title(title_value)
        plt.show()
        plt.close()
        return None

    def fit_and_run_KNN(self, df, delta_time):
        shift_instance = delta_time*10
        df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
            df, shift_instance)
        model = self.define_fit_KNN(X_train, y_train)
        model_name = 'knn' + str(delta_time) + '.pkg'
        pickle.dump(model, open(model_name, 'wb'))

        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
        predict_on_pair
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'
        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, delta_time)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        self.display_prediction_plots(prediction_1, delta_time, 'KNN ')
        '''
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', delta_time, 'KNN')
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_velocity', 'nextframesvel', 'Velocity', delta_time, 'KNN')
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_spacing', 'nextFrameSpacing', 'Spacing', delta_time, 'KNN')
        '''
        return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model

    def define_fit_KNN(self, X_train, y_train):
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)

        return model

    def fit_and_run_Random_Forest(self, df, delta_time, number_of_estimators):
        shift_instance = delta_time*10
        df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
            df, shift_instance)
        model = self.define_fit_RF(X_train, y_train, number_of_estimators)
        model_name = 'randomForest' + str(delta_time) + '.pkg'
        pickle.dump(model, open(model_name, 'wb'))
        model_name = './randomForest' + str(delta_time) + '.joblib'
        joblib.dump(model, model_name)

        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
        # predict_on_pair
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'
        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, delta_time)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        self.display_prediction_plots(
            prediction_1, delta_time, 'Random Forest ')
        '''
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', delta_time, 'Random Forest')
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_velocity', 'nextframesvel', 'Velocity', delta_time, 'Random Forest')
        self.plot_prediction(prediction_1, 'pair_Time_Duration',
                             'predicted_spacing', 'nextFrameSpacing', 'Spacing', delta_time, 'Random Forest')
        '''
        return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model

    def define_fit_RF(self, X_train, y_train, number_of_estimators):
        model = RandomForestRegressor(
            n_estimators=number_of_estimators, n_jobs=-1)
        model.fit(X_train, y_train)

        return model

    def prediction(self, test_df, test_range, target_variable, model, time_frame):

        delta_time = 0.1
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
            s_subject = np.zeros(input_df.shape[0])
            # updating the values for first Predictions
            vel[0] = input_df.iloc[0]['v_Vel']
            spacing[0] = input_df.iloc[0]['Rear_to_Front_Space_Headway']
            dv[0] = input_df.iloc[0]['Velocity Difference_Following-Preceding']

            local_y_subject[0] = input_df.iloc[0]['Local_Y']

            local_y_preceding[0] = input_df.iloc[0]['preceding_Local_Y']
            preceding_vehicle_class = input_df.iloc[0]['preceding_v_Class']
            vehicle_class = input_df.iloc[0]['v_Class']
            length_preceding_vehicle = input_df.iloc[0]['preceding_vehicle_length']

            predict_for_input = np.array(
                [spacing[0], preceding_vehicle_class, vehicle_class, dv[0], vel[0]]).reshape(1, -1)
            pred_acc[0] = model.predict(predict_for_input)
            # print(
            #    f"j: {0} input:{predict_for_input},subject localy:{local_y_subject[0]},preceding_local_y:{local_y_preceding[0]},spacing:{spacing[0]} pred_acc: {pred_acc[0]}")
            vel[1] = vel[0]+(pred_acc[0] * delta_time)
            if vel[1] < 0:
                vel[1] = 0

            dv[1] = vel[1] - input_df.iloc[1]['preceding_Vehicle_Velocity']

            s_subject[0] = ((vel[0] * delta_time) +
                            (0.5*pred_acc[0]*pow(delta_time, 2)))

            #print(f"row 0=s_subject:{s_subject[0]}")
            local_y_subject[1] = local_y_subject[0] + s_subject[0]
            local_y_preceding[1] = input_df.iloc[1]['preceding_Local_Y']

            spacing[1] = local_y_preceding[1] - \
                local_y_subject[1] - length_preceding_vehicle

            for j in range(1, len(input_df)):
                predict_for_input = np.array(
                    [spacing[j], preceding_vehicle_class, vehicle_class, dv[j], vel[j]]).reshape(1, -1)

                pred_acc[j] = model.predict(predict_for_input)
                if j == len(input_df)-1:
                    break

                vel[j+1] = vel[j]+(pred_acc[j]*0.1)

                if vel[j+1] < 0:
                    vel[j+1] = 0

                dv[j+1] = vel[j+1] - input_df.iloc[j +
                                                   1]['preceding_Vehicle_Velocity']

                s_subject[j] = ((vel[j]*0.1) +
                                (0.5*pred_acc[j]*pow(0.1, 2)))

                local_y_subject[j+1] = local_y_subject[j] + s_subject[j]
                local_y_preceding[j+1] = input_df.iloc[j +
                                                       1]['preceding_Local_Y']

                spacing[j+1] = local_y_preceding[j+1] - \
                    local_y_subject[j+1] - length_preceding_vehicle

                # print(
                #    f"j: {j} input:{predict_for_input},subject localy:{local_y_subject[j]},preceding_local_y:{local_y_preceding[j]},spacing:{spacing[j]} pred_acc: {pred_acc[j]}")

            input_df['predicted_acceleration'] = pred_acc
            input_df['predicted_velocity'] = vel
            input_df['predicted_Local_Y'] = local_y_subject
            input_df['predicted_spacing'] = spacing
            input_df['preceding_Local_Y_used'] = local_y_preceding
            input_df['s_subject'] = s_subject
            predicted_df.append(input_df)

        result = pd.concat(predicted_df)
        return result

    def prediction_preprocessing(self, df, time_frame):
        shift_instance = time_frame*10
        df = self.create_prediction_columns(df, shift_instance)

        return df

    def display_display_prediction_plots(self, prediction, delta_time, modelname):

        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', delta_time, modelname)
        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_velocity', 'nextframesvel', 'Velocity', delta_time, modelname)
        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_spacing', 'nextFrameSpacing', 'Spacing', delta_time, modelname)
        return None

    def accuracy(self, df, actual, predicted):
        R2_score = r2_score(df[actual], df[predicted])
        RMSE = np.sqrt(mean_squared_error(df[actual], df[predicted]))
        return R2_score, RMSE

    def predict_test_dataset(self, file_name, delta_time, model_name):
        string_delta_time = str(delta_time).replace('.', '_')
        file_name = file_name + string_delta_time
        print(
            f"Running test Set on :{file_name}, Reaction Time {delta_time} for Model: {model_name}")
        file = FileProcessing.FileProcessing()
        trajectory_display = file.read_input(file_name)
        target_variable = 'nextframeAcc'
        if model_name == 'neural_network_model':
            model = file.read_model(model_name, delta_time, neural=True)
        else:
            model = file.read_model(model_name, delta_time)
        predict_data = self.prediction_preprocessing(
            trajectory_display, delta_time)
        predict_on_pair = self.prediction_test_pairs(predict_data, 0, 9999)
        print(
            f"Total number of unique Pairs in Test Dataset: {len(predict_on_pair)}")
        predicted_data = self.prediction(
            predict_data, predict_on_pair, target_variable, model, delta_time)
        # predicted_data.columns
        r_square, rmse = self.accuracy(
            predicted_data, 'nextframeAcc', 'predicted_acceleration')
        print(f"\n")
        print(f"{model_name}, Reaction Time:{delta_time} Statistics Below:")
        print(f"r_square: {r_square}")
        print(f"rmse: {rmse}")
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        model_obj = ModelClass.ModelClass()
        model_obj.display_display_prediction_plots(
            prediction_1, delta_time, model_name)
        return r_square, rmse
