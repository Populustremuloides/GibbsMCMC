from Nodes import *

def getParentState(parentNode): # by default, parentNode must be a list (in this case of length 1)
    return parentNode[0].state

def generateNetwork():

    # set up prior nodes and posterior nodes for mean
    priorMuMu = NormalNode(mu=0, sigma=1, name="priorMeanMu")
    priorMuSigma = GammaNode(shape=1,scale=1, gType="logInverse", name="priorMeanSigma")
    priorMuMuLinker = LinkerNode([priorMuMu], getParentState)
    priorMuSigmaLinker = LinkerNode([priorMuSigma], getParentState)
    posteriorMu = NormalNode(mu=priorMuMuLinker, sigma=priorMuSigmaLinker, name="posteriorMu")
    # attach children
    priorMuMu.initialize([posteriorMu])
    priorMuSigma.initialize([posteriorMu])

    # set up prior nodes and posterior nodes for variance
    priorSigmaShape = GammaNode(shape=1, scale=1, gType="logInverse", name="priorSigmaShape")
    priorSigmaScale = GammaNode(shape=1, scale=1, gType="logInverse", name="priorSigmaScale")
    priorSigmaShapeLinker = LinkerNode([priorSigmaShape], getParentState)
    priorSigmaScaleLinker = LinkerNode([priorSigmaScale], getParentState)
    posteriorSigma = GammaNode(shape=priorSigmaShapeLinker, scale=priorSigmaScaleLinker, gType="logInverse", name="posteriorSigma")
    # attach children
    priorSigmaShape.initialize([posteriorSigma])
    priorSigmaScale.initialize([posteriorSigma])

    # save to the network
    network = [priorMuMu, priorMuSigma,posteriorMu, priorSigmaShape, priorSigmaScale,  posteriorSigma]
    unobservedIndices = np.arange(0, len(network))

    # add the observations
    obs = []
    with open("faculty_dat.csv", "r+") as data:
        i = 1
        children = []
        for line in data:
            line = line.replace("\n","")
            observation = float(line)
            obs.append(observation)
            posteriorMuLinker = LinkerNode([posteriorMu], getParentState)
            posteriorSigmaLinker = LinkerNode([posteriorSigma], getParentState)
            obsNode = NormalNode(mu=posteriorMuLinker, sigma=posteriorSigmaLinker, name=str(i))
            obsNode.setObservation(observation)
            obsNode.initialize([])
            children.append(obsNode)
            i = i + 1

    print("mean observation: ", np.mean(obs))
    print("var observation: ", np.var(obs))
    quit()
    # add children
    network[2].initialize(children)
    network[5].initialize(children)

    network = network + children

    return network, unobservedIndices

