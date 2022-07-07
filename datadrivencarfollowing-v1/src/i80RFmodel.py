from asyncio.windows_events import NULL
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
import pyreadr

class ModelClass():
    
    def preprocessing(self,df,time_frame):
        df["Vehicle.ID"]=df["Vehicle.ID"].astype(str)
        df["Preceding.Vehicle.ID"]=df["Preceding.Vehicle.ID"].astype(str)
        df["LF_pairs"] = df["Preceding.Vehicle.ID"]+ "_"+ df["Vehicle.ID"]
        df["nextframeAcc"] = df.groupby(
            ["LF_pairs"], as_index=False)["sacc"].shift(-10*time_frame)
        df["nextframesvel"] = df.groupby(
            ["LF_pairs"], as_index=False)["svel"].shift(-10*time_frame)
        df["nextframeposition"] = df.groupby(
            ["LF_pairs"], as_index=False)["Local.Y"].shift(-10*time_frame)
        df['Pair_Time_Duration']=(df.groupby(['LF_pairs'],as_index=False).cumcount()*0.1) +0.1
        df['PrecVehType'] = df['PrecVehType'].map({'Motorcycle': 1, 'Car': 2, 'Heavy-Vehicle': 3})
        df['Vehicle.type'] = df['Vehicle.type'].map({'Motorcycle': 1, 'Car': 2, 'Heavy-Vehicle': 3})
        df = df[df["PrecVehClass"].notna()]
        df = df[df["svel"].notna()]
        df = df[df["spacing"].notna()]
        df = df[df["frspacing"].notna()]
        df = df[df["Local.Y"].notna()]
        df = df[df["nextframeAcc"].notna()]
        df = df[df["Pair_Time_Duration"].notna()]
        df = df[df["nextframeposition"].notna()]
        df = df[df["nextframesvel"].notna()]
        df = df[df["PrecVehType"].notna()]
        df = df[df["PrecVehLength"].notna()]
        df = df[df["PrecVehLocalY"].notna()]
        df = df[df["Vehicle.type"].notna()]
        return df
    
    def select_training_pairs(self,df):
        random.seed(2109)
        pairs = df["LF_pairs"].unique()
        pairs = pairs.tolist()
        v = round(len(pairs)*0.7)   
        pairs = random.sample(pairs, v)
        return pairs

    def split_df_into_train_test(self,df,train_pairs):
        #converting the total dataset to 70/30% pair for train and test. 
        train = df[df['LF_pairs'].isin(train_pairs)]
        test = df[~df['LF_pairs'].isin(train_pairs)]
        return train, test

    def fit_rfmodel(self,train,test,number_of_estimators):
        X_train = train[["spacing",'PrecVehType','Vehicle.type','dV','svel']]
        y_train= train['nextframeAcc']
        X_test = test[["spacing",'PrecVehType','Vehicle.type','dV','svel']]
        y_test= test['nextframeAcc']
        rf = RandomForestRegressor(n_estimators = number_of_estimators,n_jobs=-1)
        rf.fit(X_train,y_train)
        return X_train, y_train, X_test, y_test,rf

    def prediction_test_pairs(self, df, pair_from, pair_to):
        unique_pairs_values = df['LF_pairs'].unique()
        unique_pairs_list = unique_pairs_values.tolist()
        unique_pairs_df = unique_pairs_list[pair_from:pair_to]
        return unique_pairs_df

    def prediction(self, test,unique_pairs_df,target_variable,rf):
        predicted_df = []
        input_df = pd.DataFrame()
    # unique_pairs_df is the test range
        for i in unique_pairs_df:
    # Q this is the input data frame
            input_df = test[test['LF_pairs']== i]
            vel=np.zeros(input_df.shape[0])
            PrecVehType =  np.zeros(input_df.shape[0])
            FollVehtype = np.zeros(input_df.shape[0])
            spacing = np.zeros(input_df.shape[0])
            dv = np.zeros(input_df.shape[0])
            local_y_subject = np.zeros(input_df.shape[0])
            local_y_preceding = np.zeros(input_df.shape[0])
            pred_acc = np.zeros(input_df.shape[0])
            
        
            #adding first value of the vehicle
            vel[0]=input_df.iloc[0]['svel']
            PrecVehType[0]=input_df.iloc[0]['PrecVehType']
            FollVehtype[0] = input_df.iloc[0]['Vehicle.type']
            spacing[0] = input_df.iloc[0]['spacing']
            local_y_subject[0]=input_df.iloc[0]['Local.Y']
            local_y_preceding[0]=input_df.iloc[0]['PrecVehLocalY']
            length_previous_vehicle=input_df.iloc[0]['PrecVehLength']        
            dv[0] = input_df.iloc[0]['dV']   
            
        #?? verify this     
            pred_acc[0] = input_df.iloc[1][target_variable]
        
        

    #     #predicting first value of acceleration
                #check here
            pred_acc[1]= rf.predict(np.array([vel[0],PrecVehType[0],FollVehtype[0],dv[0],spacing[0]]).reshape(1,-1))
                

    #     #calculating vel,frspacing,local.y,dv from the predicted acceleration.
                #check here
        
        
            for j in range(1,len(input_df)):
    #         ########
    #         #print(j)
    #         ########
                vel[j] = vel[j-1]+(pred_acc[j-1]*0.1)
                dv[j] = vel[j] - input_df.iloc[j]['FollVehVel']
                s = ((vel[j-1]*0.1)+ (0.5*pred_acc[j-1]*pow(0.1,2)))
                local_y_subject[j]=  local_y_subject[j-1] + s
                local_y_preceding[j]=  input_df.iloc[j-1]['FollVehLocalY']
                spacing[j]=local_y_preceding[j] - local_y_subject[j]-length_previous_vehicle
                PrecVehType[j]= PrecVehType[j-1]
                FollVehtype[j]=FollVehtype[j-1]
    #         ########
    #         ## localy: s = ut + 0.5*a*t^2
    #         ########
                
                if j == len(input_df)-1:
                    break
                pred_acc[j+1] = rf.predict(np.array([PrecVehType[j],FollVehtype[j],vel[j],dv[j],spacing[j]]).reshape(1, -1))
    #         ########
    #         #print(pred_acc)
    #         ########
            input_df['pacc']=pred_acc
            input_df['pvel']=vel
            input_df['pspace']=spacing
            input_df['pvel']=vel
            predicted_df.append(input_df)
            result = pd.concat(predicted_df)
            #r.append(r2_score(Q[target_variable], Q['pacc']))      
            return result

    def plot_1(self, df,nextframe,prediction,title):
        plt.figure(figsize=(10, 8))
        ax = sns.lineplot(x=df["Pair_Time_Duration"], y = df[nextframe], color="r", label="Actual Value")
        sns.lineplot(x=df["Pair_Time_Duration"], y =df[prediction],  color="b", label="Fitted Values" )
        plt.title(title)
        plt.show()
        plt.close()
        return plt
