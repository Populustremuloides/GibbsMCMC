from Nodes import *
import pandas as pd

'''
The code in this file is specific to the problem at hand
For any other problem, similar code would be generated using 
the other generalist code in this folder
'''

def getCombinedMu(parentNodes):
    mu = 0
    for parentNode in parentNodes:
        mu += parentNode.state
    return mu

def getParentState(parentNode): # by default, parentNode must be a list (in this case of length 1)
    return parentNode[0].state

def generateNetwork():
    data = pd.read_csv("preparedData.csv", index_col=0)

    golfers = list(data.index)
    tournaments = np.asarray(list(data.columns))

    # get all the parent nodes first
    obsVar = GammaNode(shape=83, scale=1./0.0014, sampleVariance=1, gType="logInverse", name="HYPER_obsevation_variance")
    hyperGolferVar = GammaNode(shape=18, scale=1/0.015, sampleVariance=1, gType="logInverse", name="HYPER_golfer_variance")
    hyperTournMean = NormalNode(mu=72, sigma=2, sampleVariance=1, name="HYPER_tournament_mean")
    hyperTournVar = GammaNode(shape=18, scale=1/0.015, sampleVariance=1, gType="logInverse", name="HYPER_tournament_variance")
    hyperNodes = [obsVar, hyperGolferVar, hyperTournMean, hyperTournVar] # save this group of hyper nodes

    network = [obsVar, hyperGolferVar, hyperTournMean, hyperTournVar] # initialize the network

    # golfers
    golfersToNodes = {} # for ease later
    for golfer in golfers:

        hyperGolferLinker = LinkerNode([hyperGolferVar], getParentState)
        node = NormalNode(mu=0, sigma=hyperGolferLinker, sampleVariance=1, name="GOLFER_" + golfer)
        network.append(node)
        golfersToNodes[golfer] = node

    # tournaments
    tourneysToNodes = {} # for ease later
    tourneysToChildren = {}
    for tourney in tournaments:
        hyperTournMeanLinker = LinkerNode([hyperTournMean], getParentState) # linker to dynamically grab state from parents
        hyperTournVarLinker = LinkerNode([hyperTournVar], getParentState) # linker to dynamically grab state from parents
        node = NormalNode(mu=hyperTournMeanLinker, sigma=hyperTournVarLinker, name="TOURNAMENT_" + tourney)
        node.state = 72
        network.append(node)
        tourneysToNodes[tourney] = node
        tourneysToChildren[tourney] = [] # initialize this for use later

    # data nodes
    observedNodes = []
    for golfer, row in data.iterrows(): # the golfer names are the index
        # keep only tournaments the golfer plays in
        mask = np.asarray(~pd.isna(row))
        tournamentsPlayedIn = tournaments[mask]
        # select the non NAN observations from the row
        row = np.asarray(row)
        combinedScores = row[mask]

        # create a node for each tournament this golfer has played in
        childNodes = []
        for i in range(len(tournamentsPlayedIn)):
            tourney = tournamentsPlayedIn[i]

            # create linker nodes between each parameters that are conditional on parents
            parentsMu = [golfersToNodes[golfer], tourneysToNodes[tourney]]
            linkerMu = LinkerNode(parentsMu, getCombinedMu)
            linkerSigma = LinkerNode([obsVar], getParentState)

            # create node with linkers that will dynamically reference parent nodes
            node = NormalNode(mu=linkerMu, sigma=linkerSigma, name="OBSERVATION_" + golfer + "_" + tourney)
            node.initialize([]) # no child indices

            # set the observation
            observation = combinedScores[i]
            node.setObservation(observation)

            # store a pointer to the node in various locations
            childNodes.append(node)
            tourneysToChildren[tourney].append(node)
            network.append(node)
            observedNodes.append(node)

        # connect the children of golfer nodes
        golfersToNodes[golfer].initialize(childNodes)

    # connect the children of the tournament nodes
    for tourney in list(tourneysToNodes.keys()):
        tourneysToNodes[tourney].initialize(list(tourneysToChildren[tourney]))

    # connect children nodes to hyper parameters
    obsVar.initialize(observedNodes)
    hyperGolferVar.initialize(list(golfersToNodes.values()))
    hyperTournMean.initialize(list(tourneysToNodes.values()))
    hyperTournVar.initialize(list(tourneysToNodes.values()))

    unobservedIndices = []
    for index, node in enumerate(network):
        if not node.observed:
            unobservedIndices.append(index)

    return network, unobservedIndices
