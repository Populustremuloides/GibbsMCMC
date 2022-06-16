from Nodes import *
import numpy as np
import math

def toThePi(parentNode):
    parentNode = parentNode[0] # unpack from list
    return math.pow(parentNode.state, math.pi)
def getParentState(parentNode):
    parentNode = parentNode[0] # unpack from list
    return parentNode.state

def generateNetwork():
    # A
    A = NormalNode(mu=20, sigma=1, name="A")
    A.state = 20

    # E
    E = BetaNode(alpha=1, beta=1, name="E")
    E.state = 0.3

    # B
    aPiLinker = LinkerNode([A], toThePi)
    B = GammaNode(shape=aPiLinker, gType="log", scale= 1. / 7., name="B")
    B.state = 1

    # D
    aLinker = LinkerNode([A], getParentState)
    eLinker = LinkerNode([E], getParentState)
    D = BetaNode(alpha=aLinker, beta=eLinker, name="D")
    D.state = 0.1

    # C
    dLinker = LinkerNode([D], getParentState)
    C = BernouliNode(p=dLinker, name="C")
    C.state = 1

    # F
    F = PoissonNode(lamda=dLinker, name="F")
    F.state = 10

    # G
    fLinker = LinkerNode([F], getParentState)
    G = NormalNode(mu=eLinker, sigma=fLinker, name="G")
    G.state = 0.9
    # combine together in 1 network
    network = [A,B,C,D,E,F,G]
    unobservedIndices = np.arange(0, len(network))

    # add children nodes to parents
    aChildren = [B,D]
    bChildren = []
    cChildren = []
    dChildren = [C]
    eChildren = [D]
    fChildren = [G]
    gChildren = []

    A.initialize(aChildren)
    B.initialize(bChildren)
    C.initialize(cChildren)
    D.initialize(dChildren)
    E.initialize(eChildren)
    F.initialize(fChildren)
    G.initialize(gChildren)

    for node in network:
        print(node.pdfFunction)


    return network, unobservedIndices