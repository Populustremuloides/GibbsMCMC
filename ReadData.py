import pandas as pd
import numpy as np

dataDict = {"name":[],"score_golfer":[],"score_tournament":[]}
with open("golfdataR.dat", "r+") as gd:
    for line in gd:
        line = line.replace("\n","")
        line = line.split(" ")
        if len(line) == 3 or len(line) == 4:
            dataDict["name"].append(line[0])
            dataDict["score_golfer"].append(line[1])
            dataDict["score_tournament"].append(line[-1])

df = pd.DataFrame.from_dict(dataDict)
print(df)
tourneys = list(set(df["score_tournament"]))
tourneys = [int(x) for x in tourneys]
tourneys.sort()
golfers = list(set(df["name"]))
golfers.sort()
data = np.empty((len(golfers), len(tourneys)))

dataDict = {}
for j, tourney in enumerate(tourneys):
    dataDict[tourney] = []
    print(j, tourney)
    for i, golfer in enumerate(golfers):
        ldf = df[df["name"] == golfer]
        golferScore = list(ldf[df["score_tournament"] == str(tourney)]["score_golfer"])
        if len(golferScore) == 1:
            golferScore = golferScore[0]
            data[i][j] = float(golferScore) + float(tourney)
        else:
            data[i][j] = 0
            golferScore = None
        dataDict[tourney].append(golferScore)

outDf = pd.DataFrame.from_dict(dataDict)
outDf.index = golfers
outDf.to_csv("preparedData.csv")

