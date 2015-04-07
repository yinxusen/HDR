from os import listdir
from os.path import isfile, join
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def splitData(filename1, filename2):
    file1 = pd.read_csv("/Users/panda/data/tmp/8513B027.csv", sep = ",", header = 0)
    file2 = pd.read_csv("/Users/panda/data/tmp/TCBd2.csv", sep = ",", header = 0)
    bond_head1 = file1.merge(file2, left_on="UNIT", right_on="visual_id")
    bond_head1_tcb1 = bond_head1[bond_head1["BOND_HEAD"] == "TCB1"]
    bond_head1_tcb1_u1 = bond_head1_tcb1[bond_head1_tcb1["DIE"] == " [8PB]"]
    res_bin8 = bond_head1_tcb1_u1[bond_head1_tcb1_u1["interface_bin"] == 8]
    res_binx = bond_head1_tcb1_u1[bond_head1_tcb1_u1["interface_bin"] != 8]
    #res_bin8_sm = res_bin8[res_bin8["substrate_vendor"] == "SM"]
    #res_bin8_sh = res_bin8[res_bin8["substrate_vendor"] == "SH"]
    #res_binx_sm = res_binx[res_binx["substrate_vendor"] == "SM"]
    #res_binx_sh = res_binx[res_binx["substrate_vendor"] == "SH"]
    #return (res_bin8_sm, res_bin8_sh, res_binx_sm, res_binx_sh)
    return (res_bin8, res_binx)

def plotCompare(filename1, filename2):
    a, b = splitData(filename1, filename2)
    plt.plot(b["#Time"], b["HeightCmd(um)"], "r--", linewidth=2)
    plt.plot(a["#Time"], a["HeightCmd(um)"], "--", linewidth=2)
    plt.show()

def reIndexTime(oriTime, fname, state, dictOfOriStartPoints, dictOfNewStartPoints):
    if state == 0:
        return oriTime
    oriStartPoint = dictOfOriStartPoints[fname][state]
    newStartPoint = dictOfNewStartPoints[state - 1]
    return oriTime - oriStartPoint + newStartPoint

def main(mypath, content):
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
    fab["newTime"] = fab[["#Time", "fname", "State"]].apply(lambda row: \
        reIndexTime(row["#Time"], \
        row["fname"], row["State"], start_points, cumsum_intervals), axis=1)
    
    # 3. print each graph out
    plt.plot(fab["newTime"], fab[content], '--', linewidth=2)
    for v in cumsum_intervals:
        plt.axvline(v)
    plt.show()

#main("/Users/panda/data/tmp", sys.argv[1])
plotCompare("/Users/panda/data/tmp/8513B027.csv", "/Users/panda/data/tmp/TCBd.csv")
