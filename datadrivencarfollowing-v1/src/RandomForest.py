import numpy as np
import pandas as pd
from pathlib import Path
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error






class model():

    def CreateTrainPairsAndSplitData(df):
        '''
        dhdh
        '''
        random.seed(2109)
        pairs = df["L-F_Pair"].unique()
        pairs = pairs.tolist()
        v = round(len(pairs)*0.7)   
        pairs = random.sample(pairs, v)
        return pairs

    def Splidata(my_pairs,df):
        #converting the total dataset to 70/30% pair for train and test. 
        train = df[df['L-F_Pair'].isin(my_pairs)]
        test = df[~df['L-F_Pair'].isin(my_pairs)]
        return train, test

    
    def data_in_parts(df,rangefrom,rangeto):
        a=df['LF_pairs'].unique()
        b = a.tolist()
        b = b[rangefrom:rangeto]
        c= len(a)/30
        return c,b 

    def fitmodel(i,train,test):
        X_train = train[["Rear_to_Front_Space_Headway",'Vehicle_combination_cat','Local_Y','Velocity Difference_Following-Preceding','v_Vel']]
        y_train= train['nextframeAcc']
        X_test = test[["Rear_to_Front_Space_Headway",'Vehicle_combination_cat','Local_Y','Velocity Difference_Following-Preceding','v_Vel']]
        y_test= test['nextframeAcc']
        for i in range(0,1000):
            rf = RandomForestRegressor(n_estimators = i,n_jobs=-1)
            rf.fit(X_train,y_train)
            return rf

    def prediction(test,b,target_variable,rf):
        F_df = []
        r = []
        Q = pd.DataFrame()
        # b is the test range
        for i in b:
            # Q this is the input data frame
            Q = test[test['L-F_Pair']== i]
            vel=np.zeros(Q.shape[0])
            Vehicle_combination = np.zeros(Q.shape[0])
            Local_Y =  np.zeros(Q.shape[0])
            spacing = np.zeros(Q.shape[0])
            dv = np.zeros(Q.shape[0])

            pred_acc = np.zeros(Q.shape[0])

            #adding first value of the vehicle
            vel[0]=Q.iloc[0]['v_Vel']
            spacing[0] = Q.iloc[0]['Rear_to_Front_Space_Headway']
            Vehicle_combination[0] = Q.iloc[0]['Vehicle_combination_cat']
            Local_Y[0] =  Q.iloc[0]['Local_Y']
            dv[0] = Q.iloc[0]['Velocity Difference_Following-Preceding']   
                
            pred_acc[0] = Q.iloc[1][target_variable]

        #       #predicting first value of acceleration
                #check here
            pred_acc[1]= rf.predict(np.array([vel[0],Vehicle_combination[0],Local_Y[0],dv[0],spacing[0]]).reshape(1,-1))
                

        #     #calculating speed from the predicted acceleration.
                #check here
        
        
            for j in range(2,len(Q)):
       
                vel[j] = vel[j-1]+(pred_acc[j]*0.1)
                dv[j] = vel[j] - Q.iloc[j]['previous_Vehicle_Velocity']
                #spacing[j] = spacing[j-1]+((vel[j-1]*0.1)+ (0.5*pred_acc[j]*pow(0.1,2)))
                spacing[j] = (vel[j-1]*0.1)+ (0.5*pred_acc[j]*pow(0.1,2))
                Local_Y[j]= Local_Y[j-1]
                Vehicle_combination[j]=Vehicle_combination[j-1]
               
                if j == len(Q)-1:
                    break
                pred_acc[j+1] = rf.predict(np.array([Vehicle_combination[j],Local_Y[j],vel[j],dv[j],spacing[j]]).reshape(1, -1))
        
        Q['pacc']=pred_acc
        Q['pvel']=vel

        F_df.append(Q)
        result = pd.concat(F_df)
        return result

    def accuracy(self,F):
        mae_score = mean_absolute_error(F['V_Acc'], F['pacc'])
        r2_scores = r2_score(F['V_Acc'], F['pacc'])
        return mae_score, r2_score




            