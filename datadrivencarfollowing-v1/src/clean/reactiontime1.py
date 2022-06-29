import numpy as np
import pandas as pd
import numpy as np

def reactiontime1(df):
    df["nextframeAcc"]=df.groupby(["L-F_Pair"],as_index=False)["v_Acc"].shift(-1)
    df["nextframesvel"]=df.groupby(["L-F_Pair"],as_index=False)["v_Vel"].shift(-1)
    df["nextframeposition"]=df.groupby(["L-F_Pair"],as_index=False)["Local_Y"].shift(-1)
    return df