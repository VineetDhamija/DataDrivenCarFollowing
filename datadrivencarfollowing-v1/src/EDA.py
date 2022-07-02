import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # visualization
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline


class EDA():

    def sniff(df):
        '''
        returns missing values count, data types of each columns, minimum and maximum values.
        '''
        with pd.option_context("display.max_colwidth", 20):
            info = pd.DataFrame()
            info['data type'] = df.dtypes
            info['percent missing'] = df.isnull().sum()*100/len(df)
            info['No. unique'] = df.apply(lambda x: len(x.unique()))
            info['Min Value'] = df.apply(lambda x: np.nanmin(x))
            info['Max Value'] = df.apply(lambda x: np.nanmax(x))
            info['unique values'] = df.apply(lambda x: x.unique())
            return info.sort_values('data type')

    def bar_plot(x, df, color=None):
        '''
        Creates bar plot
        '''
        if color is None:
            barPlot = sns.countplot(data=df, x=x, hue=None)
        else:
            barPlot = sns.countplot(data=df, x=x, hue=color)
        return(barPlot)

    def box_plot(x, y, df):
        '''
        Creates box plot
        '''
        sns.set(rc={'figure.figsize': (21.7, 13.5)})
        boxPlot = sns.boxplot(data=df, x=x, y=y)
        return(boxPlot)
