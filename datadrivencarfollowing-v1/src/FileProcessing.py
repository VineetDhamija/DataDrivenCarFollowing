import pandas as pd
from pathlib import Path


class FileProcessing():
    p = Path().cwd()
    stringpath = str(p)[0:str(p).rfind('\\')] + '\\data'

#    def __init__(self):
#        self.initialize()

    def read_input(self, file_name):
        '''
        xyz
        Read the input file into a dataframe. 
        Input: File name for the file present in Data folder. 
        Output: Dataframe name. 
        '''
        print(f"original File path: {self.p}")
        print(f"Data File path: { self.stringpath}")
        ngsimfile = self.stringpath + '/' + file_name + '.csv'
        df = pd.read_csv(ngsimfile, low_memory=False)
        return df

    def export_file(self, df, file_name):
        '''
        Export the working Data frame into csv file of the mentioned name.  
        Input: 
            df
        Ouptut: 
            df
        '''

        ngsimfilteredfile = self.stringpath + '\\' + file_name + '.csv'
        df.to_csv(ngsimfilteredfile, index=False)
        return True

    def merge_files(self, df1, df2):
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
