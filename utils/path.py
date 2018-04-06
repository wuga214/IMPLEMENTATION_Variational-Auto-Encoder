import os
import pandas as pd
from numpy import genfromtxt


def pathfinder(path):
    script_dir = os.path.dirname('__file__')
    fullpath = os.path.join(script_dir, path)
    return fullpath


def readdata(path, pandas=False):
    fullpath = pathfinder(path)
    if not pandas:
        return genfromtxt(fullpath, delimiter=',')
    else:
        return pd.read_csv(fullpath)


def getfiles(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
    return files