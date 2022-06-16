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
    # Simulate a Gaussian Process

    A = NormalNode(mu=0, sigma=1, name="A")

    aLinker = LinkerNode([A], getParentState)
    B = NormalNode(mu=aLinker, sigma=1, name="B")

    bLinker = LinkerNode([B], getParentState)
    C = NormalNode(mu=bLinker, sigma=1, name="C")

    cLinker = LinkerNode([C], getParentState)
    D = NormalNode(mu=cLinker, sigma=1, name="D")

    dLinker = LinkerNode([D], getParentState)
    E = NormalNode(mu=dLinker, sigma=1, name="E")

    eLinker = LinkerNode([E], getParentState)
    F = NormalNode(mu=eLinker, sigma=1, name="F")

    fLinker = LinkerNode([F], getParentState)
    G = NormalNode(mu=fLinker, sigma=1, name="G")

    gLinker = LinkerNode([G], getParentState)
    H = NormalNode(mu=gLinker, sigma=1, name="H")

    hLinker = LinkerNode([H], getParentState)
    I = NormalNode(mu=hLinker, sigma=1, name="I")

    iLinker = LinkerNode([I], getParentState)
    J = NormalNode(mu=iLinker, sigma=1, name="J")


    # combine together in 1 network
    network = [A,B,C,D,E,F,G,H,I,J]
    unobservedIndices = np.arange(0, len(network))

    A.setObservation(0)
    G.setObservation(10)
    J.setObservation(0)

    # add children nodes to parents
    aChildren = [B]
    bChildren = [C]
    cChildren = [D]
    dChildren = [E]
    eChildren = [F]
    fChildren = [G] # observed, so still listed as a child
    gChildren = [H]
    hChildren = [I]
    iChildren = [J]
    jChildren = []

    A.initialize(aChildren)
    B.initialize(bChildren)
    C.initialize(cChildren)
    D.initialize(dChildren)
    E.initialize(eChildren)
    F.initialize(fChildren)
    G.initialize(gChildren)
    H.initialize(hChildren)
    I.initialize(iChildren)
    J.initialize(jChildren)

    return network, unobservedIndices