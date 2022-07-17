import pandas as pd
import numpy as np

#import src
import FileProcessing
import ModelClass

fileProcessing = FileProcessing.FileProcessing()
model_obj = ModelClass.ModelClass()

file_name = 'Cleaned_NGSIM_Data'
ngsim = fileProcessing.read_input(file_name)

delta_time = 0.1
ngsim_1, train_df_1, val_df_1, test_df_1, X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1, predicted_data, model = model_obj.fit_and_run_KNN(
    ngsim, delta_time)
