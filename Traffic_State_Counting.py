import pandas as pd
import numpy as np

def countCarState(output,frame):
    output = pd.read_csv('output.txt', header=None, sep =' ')
    output.to_csv
    dict_ls = {}
    for i in range(1,13):
        dict_ls[i] = 0
    output_1 = output.loc[output[1] <= frame]
    for i in range(len(output_1)):
        dict_ls[output_1.iloc[i][2]] += output_1.iloc[i][3]
    return dict_ls