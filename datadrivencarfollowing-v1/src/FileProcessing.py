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

        filepathname = self.stringpath + '\\' + file_name + '.csv'
        df.to_csv(filepathname, index=False)
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

        print(
            f" Merged Record Count:{df.shape[0]}, df1:{df1.shape[0]}, df2:{df2.shape[0]}")

        return df
