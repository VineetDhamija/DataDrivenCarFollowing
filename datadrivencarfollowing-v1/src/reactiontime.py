import numpy as np
import pandas as pd
import numpy as np



def reactiontime1(i,df):
    for i in range(1,20):
        df["nextframeAcc"]=df.groupby(["L-F_Pair"],as_index=False)["v_Acc"].shift(-i)
        df["nextframesvel"]=df.groupby(["L-F_Pair"],as_index=False)["v_Vel"].shift(-i)
        df["nextframeposition"]=df.groupby(["L-F_Pair"],as_index=False)["Local_Y"].shift(-i)
    return df