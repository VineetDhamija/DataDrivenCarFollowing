import numpy as np
import pandas as pd
from pathlib import Path


class FileProcessing():

    def read_input(self, file_name):
        '''
        Read the input file into a dataframe. 
        Input: File name for the file present in Data folder. 
        Output: Dataframe name. 
        '''
        return df