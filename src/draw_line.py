from os import listdir
from os.path import isfile, join
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def filteredData(filename1, filename2):
    file1 = pd.read_csv(filename1, sep = ",", header = 0)
    file2 = pd.read_csv(filename2, sep = ",", header = 0)
    bondHead1 = file1.merge(file2, left_on="UNIT", right_on="visual_id")
    bondHead1 = bondHead1[pd.notnull(bondHead1["#Time"])]
    data = bondHead1[(bondHead1["BOND_HEAD"] == "TCB1") & (bondHead1["DIE"] == " [8PB]") & \
        ((bondHead1["interface_bin"] == 8) | (bondHead1["interface_bin"] == 1))]
    return data

def splitData(data):
    resBin8 = data[data["interface_bin"] == 8]
    resBin1 = data[data["interface_bin"] == 1]
    return (resBin8, resBin1)

def plotCompare(filename1, filename2):
    a, b = splitData(filename1, filename2)
    plt.plot(b["#Time"], b["Force(g)"], "r--", linewidth=2)
    plt.plot(a["#Time"], a["Force(g)"], "--", linewidth=2)
    plt.show()

def reIndexTime(oriTime, fname, state, dictOfOriStartPoints, dictOfNewStartPoints):
    if state == 0:
        return oriTime
    oriStartPoint = dictOfOriStartPoints[fname][state]
    newStartPoint = dictOfNewStartPoints[state - 1]
    return oriTime - oriStartPoint + newStartPoint

def findMaxItervals(data, attr):
    intervals = data.groupby(["FILE", "State"])[attr].count()
    max_intervals = intervals.max(level="State")
    cumsum_intervals = max_intervals.cumsum()
    return cumsum_intervals

def alignIntervals(data, cumsumIntervals):
    startPoints = data.groupby(["FILE", "State"])["#Time"].agg(np.min)
    newTime = data[["#Time", "FILE", "State"]].apply(lambda row: \
        reIndexTime(row["#Time"], \
        row["FILE"], row["State"], startPoints, cumsumIntervals), axis=1)
    return newTime

def readDirOfLogFiles(myPath):
    files = [ f for f in listdir(myPath) if isfile(join(myPath,f)) ]
    data = pd.DataFrame()

    for f in files:
        frame = pd.read_csv(join(myPath, f), sep="\t", header=0, skiprows=70)
        frame['FILE'] = f
        data = data.append(frame, ignore_index = True)

    return data

def main(attr):
    filename1 = "/Users/panda/data/tmp/8513B027.csv"
    filename2 = "/Users/panda/data/tmp/TCBd2.csv"
    data = filteredData(filename1, filename2)
    cumsumIntervals = findMaxItervals(data, attr)
    newTime = alignIntervals(data, cumsumIntervals)
    data["newTime"] = newTime
    badBin, goodBin = splitData(data)
    
    plt.scatter(goodBin["newTime"], goodBin[attr])
    plt.scatter(badBin["newTime"], badBin[attr])
    for v in cumsumIntervals:
        plt.axvline(v)
    plt.show()

main("Force(g)")
