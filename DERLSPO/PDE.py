import numpy as np
import random

class Model():
    def __init__(self, PDEModel, paramNum, data, particleNum=100, maxIter=50, layersList=[4, 6, 8, 10], upper=10, lower=1e-10, threshold=1e-09):



        self.layersList = layersList


        self.PDEModel = PDEModel

        self.upper = upper
        self.lower = lower

        self.particleNum = particleNum
        self.paramNum = paramNum
        self.maxIter = maxIter

        self.X = np.zeros((self.particleNum, self.paramNum))
        self.V = np.zeros((self.particleNum, self.paramNum))
        self.pBest = np.zeros((self.particleNum, self.paramNum))
        self.gBest = np.zeros((1, self.paramNum))
        self.pFit = np.zeros(self.particleNum)
        self.fit = float('inf')

        self.data = data

        self.qTable = np.zeros((len(self.layersList), len(self.layersList)))
        self.preState = 0
        self.currentState = 0

        self.layers = []
        self.fitness = []

        self.phi = 0.4
        self.epsilon = 0.9
        self.alpha = 0.4
        self.gamma = 0.8

        self.threshold = threshold

        # self.index = np.array(index)
        # self.mask = [i for i, value in enumerate(self.index) if value == 1]

    def getData(self):
        return self.data


    def MSELoss(self, X):
        predictedData = self.PDEModel(X)
        mse = np.mean((self.data - predictedData) ** 2)
        return mse

    def selectAction(self):
        if np.random.rand() < self.epsilon:
            nextAction = np.argmax(self.qTable[self.currentState])
        else:
            nextAction = np.random.randint(0, len(self.layersList))

        self.preState = self.currentState
        self.currentState = nextAction
        return self.layersList[nextAction]

    def divideParticles(self, currentTotalLayer, fitness):
        baseCount = self.particleNum // currentTotalLayer
        remainder = self.particleNum % currentTotalLayer
        layerCounts = [baseCount] * (currentTotalLayer - 1) + [baseCount + remainder]

        particles = list(range(0, self.particleNum))
        sortedParticles = [x for _, x in sorted(zip(fitness, particles))]

        self.layers.clear()
        startIdx = 0
        for count in layerCounts:
            endIdx = startIdx + count
            self.layers.append(sortedParticles[startIdx:endIdx])
            startIdx = endIdx


    def levelCompetition(self, lec, FECount):
        prob = (FECount / self.maxIter) ** 2
        exemplarLevels = [None, None]
        for i in range(2):
            if random.random() < prob:
                lec1 = random.randint(0, lec - 1)
                lec2 = random.randint(0, lec - 1)
                if lec1 < lec2:
                    exemplarLevels[i] = lec1
                else:
                    exemplarLevels[i] = lec2
            else:
                exemplarLevels[i] = random.randint(0, lec - 1)

        if exemplarLevels[1] < exemplarLevels[0]:
            exemplarLevels[0], exemplarLevels[1] = exemplarLevels[1], exemplarLevels[0]
        return exemplarLevels

    def initParticles(self):
        for i in range(self.particleNum):
            if (i <= self.particleNum / 2):
                self.X[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1, self.paramNum)) * np.random.choice([-1, 1], self.paramNum)
                self.V[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1, self.paramNum)) * np.random.choice([-1, 1], self.paramNum)
            else:
                self.X[i] = [random.uniform(0, self.upper) for _ in range(self.paramNum)]
                self.V[i] = [random.uniform(0, self.upper) for _ in range(self.paramNum)]

            # if (self.index != None and len(self.index) == self.paramNum):

                # self.X[i][self.mask] = np.abs(self.X[i][self.mask])

            self.pBest[i] = self.X[i]
            tmp = self.MSELoss(self.X[i])
            self.fitness.append(tmp)
            self.pFit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gBest = self.X[i]


    def iterator(self):
        for t in range(self.maxIter):
            pregBest = self.gBest
            currentTotalLayer = self.selectAction()

            self.divideParticles(currentTotalLayer, self.fitness)
            for i in range(currentTotalLayer - 1, 1, -1):
                for j in self.layers[i]:
                    exemplarLevels = self.levelCompetition(i, t)
                    if (exemplarLevels[0] == exemplarLevels[1]):
                        index1 = random.randint(0, len(self.layers[exemplarLevels[0]]) - 2)
                        index2 = random.randint(index1 + 1, len(self.layers[exemplarLevels[0]]) - 1)
                        id1 = self.layers[exemplarLevels[0]][index1]
                        id2 = self.layers[exemplarLevels[0]][index2]
                    else:
                        id1 = random.choice(self.layers[exemplarLevels[0]])
                        id2 = random.choice(self.layers[exemplarLevels[1]])
                    X1 = self.X[id1]
                    X2 = self.X[id2]

                    r1 = random.uniform(0, 1)
                    r2 = random.uniform(0, 1)
                    r3 = random.uniform(0, 1)

                    self.V[j] = r1 * self.V[j] + r2 * (X1 - self.X[j]) + r3 * self.phi * (X2 - self.X[j])
                    self.X[j] = self.X[j] + self.V[j]

                    # if(self.index != None and len(self.index) == self.paramNum):
                    #     self.X[i][self.mask] = np.abs(self.X[i][self.mask])

            for k in self.layers[1]:
                index1 = random.randint(0, len(self.layers[0]) - 2)
                index2 = random.randint(index1 + 1, len(self.layers[0]) - 1)
                id1 = self.layers[0][index1]
                id2 = self.layers[0][index2]
                X1 = self.X[id1]
                X2 = self.X[id2]

                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                r3 = random.uniform(0, 1)

                self.V[k] = r1 * self.V[k] + r2 * (X1 - self.X[k]) + r3 * self.phi * (X2 - self.X[k])
                self.X[k] = self.X[k] + self.V[k]

                # if (self.index != None and len(self.index) == self.paramNum):
                #     self.X[k][self.mask] = np.abs(self.X[k][self.mask])

            if (t == int(self.maxIter / 2) and self.fit > self.threshold):
                for i in range(self.particleNum):
                    self.X[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1,
                                                                                                                   self.paramNum)) * np.random.choice(
                        [-1, 1], self.paramNum)
                    self.V[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1,
                                                                                                                   self.paramNum)) * np.random.choice(
                        [-1, 1], self.paramNum)

                    # if (self.index != None and len(self.index) == self.paramNum):
                    #     self.X[i][self.mask] = np.abs(self.X[i][self.mask])

            self.fitness.clear()
            for i in range(self.particleNum):
                temp = self.MSELoss(self.X[i])
                self.fitness.append(temp)
                if temp < self.pFit[i]:
                    self.pFit[i] = temp
                    self.pBest[i] = self.X[i]
                    if self.pFit[i] < self.fit:
                        self.gBest = self.X[i]
                        self.fit = self.pFit[i]

            preFitness = self.MSELoss(pregBest)
            curFitness = self.MSELoss(self.gBest)
            reward = abs(curFitness - preFitness) / abs(max(curFitness, 1e-10))
            newQ = (self.qTable[self.preState][self.currentState] +
                                                            self.alpha * (reward + self.gamma * max(self.qTable[self.currentState]) - self.qTable[self.preState][self.currentState]))
            self.qTable[self.preState][self.currentState] = newQ

    def getGBest(self):
        return self.gBest

    def getFit(self):
        return self.fit
