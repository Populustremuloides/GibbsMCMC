import pandas as pd
import numpy as np

def getNintyPercent(array):
    array = np.sort(array)
    numToInclude = int(0.9 * array.shape[0])
    startIndex = int(0.05 * array.shape[0])
    endIndex = startIndex + numToInclude
    return array[startIndex], array[endIndex]


df = pd.read_csv("gaussian_process_network_samples.csv")
print(df)
with open("gaussian_results.csv", "w+") as oFile:
    oFile.write("node,mean_state,lower_bound90,upper_bound90\n")
    for col in df.columns:
        data = np.asarray(df[col])
        lower, upper = getNintyPercent(data)
        mean = np.mean(data)
        line = col + "," + str(mean) + "," + str(lower) + "," + str(upper) + "\n"
        oFile.write(line)



