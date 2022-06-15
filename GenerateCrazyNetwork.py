from Nodes import *
import numpy as np

def toThePi(parentNode):
    parentNode = parentNode[0] # unpack from list
    return parentNode.state ** np.pi
def getParentState(parentNode):
    parentNode = parentNode[0] # unpack from list
    return parentNode.state

def generateNetwork():
    # A
    A = NormalNode(mu=20, sigma=1, name="A")

    # E
    E = BetaNode(alpha=1, beta=1, name="E")

    # B
    aPiLinker = LinkerNode([A], toThePi)
    B = GammaNode(shape=aPiLinker, scale= 1. / 7., name="B")

    # D
    aLinker = LinkerNode([A], getParentState)
    eLinker = LinkerNode([E], getParentState)
    D = BetaNode(alpha=aLinker, beta=eLinker, name="D")

    # C
    dLinker = LinkerNode([D], getParentState)
    C = BernouliNode(p=dLinker, name="C")

    # F
    F = PoissonNode(lamda=dLinker, name="F")

    # G
    fLinker = LinkerNode([F], getParentState)
    G = NormalNode(mu=eLinker, sigma=fLinker, name="G")

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

    return network, unobservedIndices