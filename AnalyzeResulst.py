import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("opening data")
df = pd.read_csv("network_samples.csv")
cols = df.columns
golfers = []
tournaments = []
for col in cols:
    col.replace(" ","")
    # print(col[:5])
    if col[:5] == "GOLFE":
    # if col.startswith("GOFLE"):
        print(col[:5])
        golfers.append(col)
    elif col.startswith("TOURN"):
        tournaments.append(col)
    # else:
    #     print("meh")

print(golfers)
gdf = df[golfers]
# remove burnin
gdf = gdf.iloc[1000:]
print(gdf)
values = list(gdf.mean())
print(values)
indices = np.argsort(values)
print(indices)
np.flip(indices)
print(indices)
golfers = np.asarray(golfers)
print(golfers[indices])

def getNintyPercent(array):
    array = np.sort(array)
    numToInclude = int(0.9 * array.shape[0])
    startIndex = int(0.05 * array.shape[0])
    endIndex = startIndex + numToInclude
    return array[startIndex], array[endIndex]

with open("golfer_results.csv", "w+") as oFile:
    oFile.write("golfer,mean,lower_bound_90percent,upper_bound_90percent\n")
    for golfer in golfers:
        lower, upper = getNintyPercent(gdf[golfer])
        mean = np.mean(gdf[golfer])
        golfer = golfer.replace(",", "")
        line = golfer + "," + str(mean) + "," + str(lower) + "," + str(upper) + "\n"
        oFile.write(line)
