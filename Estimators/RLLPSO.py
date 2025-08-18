# coding: utf-8
from copy import deepcopy
import csv
import time

import math

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import truncnorm

from equation import Equation

from .estimator import Estimator

class PSO_ALGORITHM:
    def __init__(self, func: Equation, pN, dim, actual_data, time, initial_conditions,numberOfLayers_list=[4, 6, 8, 10], max_iter=200):

        self.func = func
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.upper = 10
        self.lower = 1e-10

        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.X = np.zeros((self.pN, self.dim))
        self.V = np.zeros((self.pN, self.dim))
        self.pBest = np.zeros((self.pN, self.dim))
        self.gBest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.fit = 1e20

        self.initial_conditions = initial_conditions
        self.t = time
        self.actual_data = actual_data

        self.numberOfLayers_list = numberOfLayers_list
        self.qTable = np.zeros((len(self.numberOfLayers_list), len(self.numberOfLayers_list)))
        self.preState = 0
        self.currentState = 0

        self.layers = []

        self.phi = 0.4

        self.epsilon = 0.9
        self.alpha = 0.4
        self.gamma = 0.8

        self.iteratorMse = []

    def get_actual_data(self):
        return self.actual_data

    def function(self, X):
        temp_x = tuple(X),
        predicted_data = odeint(self.func.f(), self.initial_conditions, self.t,
                                    args=temp_x, tfirst=self.func.t_first)
        mse = np.mean((self.actual_data - predicted_data) ** 2) #+ 0.0001 * (abs(X[0]) + abs(X[1]) + abs(X[2]) + abs(X[3]))
        return mse

    def selectAction(self):
        if np.random.rand() < self.epsilon:
            nextAction = np.argmax(self.qTable[self.currentState])
        else:
            nextAction = np.random.randint(0, len(self.numberOfLayers_list))

        self.preState = self.currentState
        self.currentState = nextAction
        return self.numberOfLayers_list[nextAction]

    def divideParticles(self, currentTotalLayer, fitness):
        baseCount = self.pN // currentTotalLayer
        remainder = self.pN % currentTotalLayer
        layerCounts = [baseCount] * (currentTotalLayer - 1) + [baseCount + remainder]

        particles = list(range(0, self.pN))
        sortedParticles = [x for _, x in sorted(zip(fitness, particles))]

        self.layers.clear()
        start_idx = 0
        for count in layerCounts:
            end_idx = start_idx + count
            self.layers.append(sortedParticles[start_idx:end_idx])
            start_idx = end_idx


    def levelCompetition(self, lec, FECount):
        prob = (FECount / self.max_iter) ** 2
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

    def init_population(self):
        for i in range(self.pN):
            if (i <= self.pN / 2):
                self.X[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1, self.dim)) * np.random.choice([-1, 1], self.dim)
                self.V[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1, self.dim)) * np.random.choice([-1, 1], self.dim)
            else:
                self.X[i] = [random.uniform(-10, 10) for _ in range(self.dim)]
                self.V[i] = [random.uniform(-10, 10) for _ in range(self.dim)]
            self.pBest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gBest = self.X[i]


    def random_small_value(self):
        lower_bound = -1
        upper_bound = 1
        return random.uniform(lower_bound, upper_bound)

    def iterator(self):
        for t in range(self.max_iter):
            pregBest = self.gBest
            currentTotalLayer = self.selectAction()
            fitness = [self.function(x) for x in self.X]


            self.divideParticles(currentTotalLayer, fitness)
            for i in range(currentTotalLayer - 1, 1, -1):
                for j in self.layers[i]:
                    exemplarLevels = self.levelCompetition(i, t)
                    if(exemplarLevels[0] == exemplarLevels[1]):
                        index1 = random.randint(0, len(self.layers[exemplarLevels[0]]) - 2)
                        index2 = random.randint(index1 + 1, len(self.layers[exemplarLevels[0]]) - 1)
                        id1 = self.layers[exemplarLevels[0]][index1]
                        id2 = self.layers[exemplarLevels[0]][index2]

                    else:

                        id1 = random.choice(self.layers[exemplarLevels[0]])
                        id2 = random.choice(self.layers[exemplarLevels[1]])

                    X1 = self.X[id1]
                    X2 = self.X[id2]

                    self.r1 = random.uniform(0, 1)
                    self.r2 = random.uniform(0, 1)
                    self.r3 = random.uniform(0, 1)

                    self.V[j] = self.r1 * self.V[j] + self.r2 * (X1 - self.X[j]) + self.r3 * self.phi * (X2 - self.X[j])
                    self.X[j] = self.X[j] + self.V[j]



            for k in self.layers[1]:
                index1 = random.randint(0, len(self.layers[0]) - 2)
                index2 = random.randint(index1 + 1, len(self.layers[0]) - 1)
                id1 = self.layers[0][index1]
                id2 = self.layers[0][index2]
                X1 = self.X[id1]
                X2 = self.X[id2]

                self.r1 = random.uniform(0, 1)
                self.r2 = random.uniform(0, 1)
                self.r3 = random.uniform(0, 1)

                self.V[k] = self.r1 * self.V[k] + self.r2 * (X1 - self.X[k]) + self.r3 * self.phi * (X2 - self.X[k])
                self.X[k] = self.X[k] + self.V[k]


            if (t == int(self.max_iter / 2) and self.fit > 1e-04):
                for i in range(self.pN):
                        self.X[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1,
                                                                                                                       self.dim)) * np.random.choice(
                            [-1, 1], self.dim)
                        self.V[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1,
                                                                                                                       self.dim)) * np.random.choice(
                            [-1, 1], self.dim)


            for i in range(self.pN):
                temp = self.function(self.X[i])
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pBest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:
                        self.gBest = self.X[i]
                        self.fit = self.p_fit[i]

            preFitness = self.function(pregBest)
            curFitness = self.function(self.gBest)
            reward = abs(curFitness - preFitness) / abs(max(curFitness, 1e-10))
            newQ = (self.qTable[self.preState][self.currentState] +
                                                            self.alpha * (reward + self.gamma * max(self.qTable[self.currentState]) - self.qTable[self.preState][self.currentState]))
            self.qTable[self.preState][self.currentState] = newQ


    def get_global_best(self):
        return self.gBest

    def get_fit(self):
        return self.fit

    def get_iter(self):
        return self.iteratorMse


class RLLPSO(Estimator):
    def __init__(self, func: Equation, particle_num=100,
                 max_iter=200, layers_list=[4, 6, 8, 10], upper=10,
                 lower=1e-10, threshold=1e-04):
        self.func = func
        self.particle_num = particle_num
        self.max_iter = max_iter
        self.layers_list = layers_list
        self.upper = upper
        self.lower = lower
        self.threshold = threshold

    def train(self):
        raise Exception("RLLPSO does not need to train.")

    def predict(self, data, time):
        estimator = PSO_ALGORITHM(self.func, self.particle_num, self.func.num_param, data, time, data[0])
        estimator.init_population()
        estimator.iterator()
        return estimator.get_global_best()
