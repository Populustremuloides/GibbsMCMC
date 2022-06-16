import numpy as np
from PDFs import *

class LinkerNode():
    '''
    class for storing complex relationships between parent nodes and children nodes
    '''
    def __init__(self, parentNodes, function):
        self.function = function
        self.parentNodes = parentNodes

    def computeValue(self):
        return self.function(self.parentNodes)

class Node():
    '''
    parent class for storing the full conditional of a distribution within a bayesian network
    *** shouldn't be implemented without taking on a specific distribution ***
    '''

    def __init__(self):
        self.children = []
        self.observed = False
        self.pdfFunction = None
        self.sampleVariance = 1
        self.args = {}
        self.observed = False # should the observed value be the "state"?
        self.initialized = False

    def getProbabilityRelativeToObservedChildren(self, x, xPrime):
        densitiesCurrent = 1
        for child in self.children:
            # if child.observed: # FIXME: put this back
                observedChildState = child.state
                density = child.pdf(observedChildState)
                densitiesCurrent = densitiesCurrent + density

        # change the current state temporarily to see how our children like it :)
        self.state = xPrime
        densitiesCandidate = 1
        for child in self.children:
            # if child.observed: # FIXME: put this back
                observedChildState = child.state
                density = child.pdf(observedChildState)
                densitiesCandidate = densitiesCandidate + density

        self.state = x  # replace the original state
        return densitiesCandidate - densitiesCurrent

    def inSupport(self, x):  # override in child nodes if needs be
        return True

    def metropolisSampler(self):
        xPrime = np.random.normal(loc=self.state, scale=self.sampleVariance)

        # hack to make the sampler more efficient in the case of limited-support distributions
        if not self.inSupport(xPrime):
            previousSampleVariance = self.sampleVariance # store what the previous sample variance was
            while not self.inSupport(xPrime):
                xPrime = np.random.normal(loc=self.state, scale=self.sampleVariance)
                # self.sampleVariance += 1 # increase the odds of getting something in the support
            self.sampleVariance = previousSampleVariance # revert to original sample variance

        if self.args["round"]:  # throwing a bone to discrete pmfs
            x = round(self.state)
            xPrime = round(xPrime)
        else:
            x = self.state

        # probability relative to parents
        try:
            a1 = self.pdfFunction(xPrime, self.args) - self.pdfFunction(x, self.args)
        except:
            print("error")
            a1 = self.pdfFunction(xPrime, self.args) - self.pdfFunction(x, self.args)

        # include the probability of any observed children
        a2 = self.getProbabilityRelativeToObservedChildren(x, xPrime)

        # decide whether to keep the candidate value
        acceptanceProb = np.min([1, a1 + a2])
        randomNum = -np.log(np.random.uniform())

        if randomNum < acceptanceProb: # FIXME: put this back to a >
            self.state = xPrime
        else:
            self.state = x

    def initialize(self, children):
        self.children = children
        self.initialized = True

    def updateArgs(self): # should be instantiated in each child class
        pass # admittedly a little sloppy

    def setObservation(self, observation):
        self.observed = True
        self.state = observation

    def pdf(self, x):
        self.updateArgs() # get the most recent parameterization of this node based on parents
        density = self.pdfFunction(x, self.args) # get density
        return density

    def sample(self):
        if not self.initialized:
            return "error: node not initialized"
        if self.observed:
            return self.state
        else:
            self.updateArgs()
            self.metropolisSampler()



class NormalNode(Node):
    def __init__(self, mu, sigma, children=None, sampleVariance=1,  name="uninitialized node"):
        super().__init__()
        if children is None:
            children = []
        self.state = 0
        self.args = {"mu":0,"sigma":1,"round":False}
        self.children = children
        self.pdfFunction = logNormalPDF
        self.name = name

        # check if the parameters are pre-determined or vary based on another node
        if (type(mu) == type(1)) or (type(mu) == type(1.)):
            self.muPredetermined = True
            self.args["mu"] = mu
        else:
            self.muPredetermined = False

        if (type(sigma) == type(1)) or (type(sigma) == type(1.)):
            self.sigmaPredetermined = True
            self.args["sigma"] = sigma
        else:
            self.sigmaPredetermined = False

        self.mu = mu # either an int or a linker node
        self.sigma = sigma # either an int or a linker node
        self.sampleVariance = sampleVariance

    def updateArgs(self):
        if not self.muPredetermined:
            muValue = self.mu.computeValue()
            self.args["mu"] = muValue
        if not self.sigmaPredetermined:
            sigmaValue = self.sigma.computeValue()
            self.args["sigma"] = sigmaValue

class GammaNode(Node):
    def __init__(self, shape, scale, sampleVariance=1, gType="typical",  name="uninitialized node"):
        super().__init__()
        self.state = 1
        self.args = {"shape": 1, "scale": 1, "round": False}
        self.gType = gType
        self.name = name

        # if self.gType == "inverse":
        #     self.pdfFunction = inverseGammaPDF
        if self.gType == "log":
            self.pdfFunction = logGammaPDF
        elif self.gType == "logInverse":
            self.pdfFunction = inverseLogGammaPDF
        # elif self.gType == "typical":
        #     self.pdfFunction = gammaPDF
        else:
            print("error: gamma distribution incorrectly specified. Node" + self.name + " not initialized properly.")

        # check if the parameters are pre-determined or vary based on another node
        if (type(shape) == type(1)) or (type(shape) == type(1.)):
            self.shapePredetermined = True
            self.args["shape"] = shape
        else:
            self.shapePredetermined = False

        if (type(scale) == type(1)) or (type(scale) == type(1.)):
            self.scalePredetermined = True
            self.args["scale"] = scale
        else:
            self.scalePredetermined = False

        self.shape = shape  # either an int or a pointer to a node
        self.scale = scale  # either an int or a pointer to a node
        self.sampleVariance = sampleVariance

    def inSupport(self, x):
        if x < 0:
            return False
        return True

    def updateArgs(self):
        if not self.shapePredetermined:
            self.args["shape"] = self.shape.computeValue()
        if not self.scalePredetermined:
            self.args["scale"] = self.scale.computeValue()

class PoissonNode(Node):
    def __init__(self, lamda, sampleVariance=1,  name="uninitialized node"):
        super().__init__()
        self.state = 0
        self.args = {"lamda":1,"round":True}
        self.pdfFunction = logPoissonPMF
        self.name = name

        # check if the parameters are pre-determined or vary based on another node
        if (type(lamda) == type(1)) or (type(lamda) == type(1.)):
            self.lamdaPredetermined = True
            self.args["lamda"] = lamda
        else:
            self.lamdaPredetermined = False

        self.lamda = lamda # either an int or a pointer to a node
        self.sampleVariance = sampleVariance

    def inSupport(self, x):
        if x < 1:
            return False
        return True

    def updateArgs(self):
        if not self.lamdaPredetermined:
            self.args["lamda"] = self.lamda.computeValue()

class BetaNode(Node):
    def __init__(self, alpha, beta, sampleVariance=1,  name="uninitialized node"):
        super().__init__()
        self.state = 0.5
        self.args = {"alpha":0,"beta":1,"round":False}
        self.pdfFunction = logBetaPDF
        self.name = name

        # check if the parameters are pre-determined or vary based on another node
        if (type(alpha) == type(1)) or (type(alpha) == type(1.)):
            self.alphaPredetermined = True
            self.args["alpha"] = alpha
        else:
            self.alphaPredetermined = False

        if (type(beta) == type(1)) or (type(beta) == type(1.)):
            self.betaPredetermined = True
            self.args["beta"] = beta
        else:
            self.betaPredetermined = False

        self.alpha = alpha # either an int or a pointer to a node
        self.beta = beta # either an int or a pointer to a node
        self.sampleVariance = sampleVariance

    def inSupport(self, x):
        if x < 0:
            return False
        if x > 1:
            return False
        return True

    def updateArgs(self):
        if not self.alphaPredetermined:
            self.args["alpha"] = self.alpha.computeValue()
        if not self.betaPredetermined:
            self.args["beta"] = self.beta.computeValue()

class BernouliNode(Node):
    def __init__(self, p, sampleVariance=1,  name="uninitialized node"):
        super().__init__()
        self.state = 0
        self.name = name

        # check if the parameters are pre-determined or vary based on another node
        if (type(p) == type(1)) or (type(p) == type(1.)):
            self.pPredetermined = True
            self.args["p"] = p
        else:
            self.pPredetermined = False

        self.p = p # either an int or a pointer to a node
        self.sampleVariance = sampleVariance

    def updateArgs(self):
        if not self.pPredetermined:
            self.args["p"] = self.p.computeValue()

    def sample(self): # override the inherited sample code because bernouli and metropolis don't get along
        self.updateArgs()
        randomNum = np.random.uniform()
        if randomNum < self.args["p"]:
            self.state = 1 # True
        else:
            self.state = 0 # False
