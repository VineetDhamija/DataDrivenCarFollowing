import numpy as np
import pandas as pd
from pathlib import Path


class FileProcessing():
    p = Path().cwd()
    stringpath = str(p)[0:str(p).rfind('\\')] + '\\data'

#    def __init__(self):
#        self.initialize()

    def readInput(self, fileName):
        '''
        Read the input file into a dataframe. 
        Input: File name for the file present in Data folder. 
        Output: Dataframe name. 
        '''
        print(f"original File path: {self.p}")
        print(f"Data File path: { self.stringpath}")
        ngsimfile = self.stringpath + '/' + fileName + '.csv'
        df = pd.read_csv(ngsimfile, low_memory=False)
        return df

    def exportFile(self, df, fileName):
        '''
        Export the working Data frame into csv file of the mentioned name.  
        Input: 
            df
        Ouptut: 
            df
        '''

        ngsimfilteredfile = self.stringpath + '\\' + fileName + '.csv'
        df.to_csv(ngsimfilteredfile, index=False)
        return True

    def mergeFiles(self, df1, df2):
        '''
        Merge the I-80 and US-101 Highway dataframe.  
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
