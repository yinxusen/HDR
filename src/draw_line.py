from os import listdir
from os.path import isfile, join
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def reIndexTime(oriTime, fname, state, dictOfOriStartPoints, dictOfNewStartPoints):
    if state == 0:
        return oriTime
    oriStartPoint = dictOfOriStartPoints[fname][state]
    newStartPoint = dictOfNewStartPoints[state - 1]
    return oriTime - oriStartPoint + newStartPoint

def main(mypath):
    files = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    
    # 1. find max intervals of each stage in a dir
    fab = pd.DataFrame()
    for f in files:
        frame = pd.read_csv(join(mypath, f), sep="\t", header=0, skiprows=70)
        frame['fname'] = f
        fab = fab.append(frame, ignore_index = True)
    
    intervals = fab.groupby(["fname", "State"])["Force(g)"].count()
    max_intervals = intervals.max(level="State")
    cumsum_intervals = max_intervals.cumsum()
    
    # 2. align each stage with its corresponding largest interval
    start_points = fab.groupby(["fname", "State"])["#Time"].agg(np.min)
    fab["newTime"] = fab[["#Time", "fname", "State"]].apply(lambda row: reIndexTime(row["#Time"], row["fname"], row["State"], start_points, cumsum_intervals), axis=1)
    
    # 3. print each graph out
    print fab["newTime"].describe()
    print fab["#Time"].describe()
    
    #fab["value"] = fab[["Force(g)", "State"]].apply(lambda row: avg[row["State"]], axis=1)
    #
    #plt.plot(fab["#Time"], fab["Force(g)"], '--', linewidth=2)
    ##plt.plot(fab["#Time"], fab["value"], 'r--', linewidth=2)
    #
    #plt.show()

main(sys.argv[1])
