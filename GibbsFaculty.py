from GenerateFacultyNetwork import *
import pandas as pd
from tqdm import tqdm

# only sample from unobserved nodes
# record the states of all the unobserved nodes
def recordSamples(dataDict, network, unobservedIndices):
    for index in unobservedIndices:
        name = network[index].name
        state = network[index].state
        dataDict[name].append(state)
    return dataDict

print("initializing network")
network, unobservedIndices = generateNetwork()
print("network initialized")

# generate dictionary to store data
dataDict = {}
for index in unobservedIndices:
    dataDict[network[index].name] = []

# generate a gazillion samples
numSamples = 100000
saveInterval = 1000
loop = tqdm(total=numSamples)
for i in range(numSamples):
    for index in unobservedIndices:
        node = network[index]
        print(node.name)
        node.sample()
    dataDict = recordSamples(dataDict, network, unobservedIndices)
    loop.update()
# fixme: figure out why we go off to positive or negative infinity for different parameters
    if i % saveInterval == 0:
        outDF = pd.DataFrame.from_dict(dataDict)
        outDF.to_csv("network_samples_faculty.csv", index=False)

outDF = pd.DataFrame.from_dict(dataDict)
outDF.to_csv("network_samples_faculty.csv", index=False)