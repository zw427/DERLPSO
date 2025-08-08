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

from .estimator import Estimator

class RLLPSO(Estimator):
    def __init__(self, pN, dim, max_iter, α, β, δ, γ, numberOfLayers_list, actual_data, time, initial_conditions):
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

        self.α = α
        self.β = β
        self.δ = δ
        self.γ = γ
        self.param = [self.α, self.β, self.δ, self.γ]
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

    def lotka_volterra(self, state, t, α, β, γ, δ):
        l, v = state
        dxdt = α * l - β * l * v
        dydt = δ * l * v - γ * v
        return [dxdt, dydt]

    def lorenz(self, state, t, σ, β, ρ):
        x, y, z = state
        dxdt = σ * (y - x)
        dydt = x * (ρ - z) - y
        dzdt = x * y - β * z
        return [dxdt, dydt, dzdt]

    def fitzhugh_nagumo(self, state, t, a, b):
        v, w = state
        dvdt = 3.0 * (v - v ** 3 / 3 + w - 0.4)
        dwdt = (-1 / 3.0) * (v - a + b * w)
        return [dvdt, dwdt]

    def function(self, X):
        predicted_data = odeint(self.lotka_volterra, self.initial_conditions, self.t, args=(X[0], X[1], X[2], X[3]))
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

            true = np.array([self.α, self.β, self.δ, self.γ])
            diff = np.array(true) - np.array(self.gBest)
            mse = np.mean(diff ** 2)
            self.iteratorMse.append(mse)


    def get_gBest(self):
        return self.gBest

    def get_fit(self):
        return self.fit

    def get_iter(self):
        return self.iteratorMse


# time = []
# current_group = []
# with open('no_noise/data1/20/time.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         row_data = [float(x) for x in row]
#         time.append(row_data)

# param = []
# with open('no_noise/data1/20/param.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         row_data = [float(x) for x in row]
#         param.append(row_data)

# data = []
# current_group = []
# with open('no_noise/data1/20/data.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         row_data = [float(x) for x in row]
#         # data.append(row_data)
#         data.append([row_data[i:i + 2] for i in range(0, len(row_data), 2)])

# numberOfLayers_list = [4, 6, 8, 10]
# max_iter = 200
# fits = []


# resultName = 'no_noise/data1/20/result_all_iter.csv'
# with open(resultName, 'a', newline='') as resultfile:
#     result_writer = csv.writer(resultfile)

#     for i in range(100):
#         my_pso = PSO(pN=100, dim=4, max_iter=max_iter, α=param[i][0], β=param[i][1], δ=param[i][2], γ=param[i][3],
#                      numberOfLayers_list=numberOfLayers_list, actual_data=data[i], time=time[i],
#                      initial_conditions=data[i][0])
#         my_pso.init_population()
#         my_pso.iterator()

#         print(my_pso.get_iter())
#         result_writer.writerow(my_pso.get_iter())

#         # print("param:", ["{:.16f}".format(x) for x in param[i]])
#         # print("gBest:", ["{:.16f}".format(x) for x in my_pso.get_gBest()])
#         # mse = np.mean((np.array(param[i]) - np.array(my_pso.get_gBest())) ** 2)
#         # print(mse)

#         # predicted_data = odeint(my_pso.lotka_volterra, data[i][0], time[i], args=(my_pso.get_gBest()[0], my_pso.get_gBest()[1], my_pso.get_gBest()[2], my_pso.get_gBest()[3]))
#         # mse = np.mean((data[i] - predicted_data) ** 2)
#         # print("MSE:", "{:.16f}".format(my_pso.get_fit()))
#         # print("-------")
#         # fits.append(my_pso.get_fit())

#         # solution = odeint(lotka_volterra, data[i][0], time[i], args=tuple(my_pso.get_gBest()))
#         # print(solution)
#         # plt.figure(figsize=(4, 3))
#         # plt.plot(time[i], solution[:, 0])
#         # plt.plot(time[i], solution[:, 1])
#         # plt.axis('off')
#         # plt.margins(0.05, 0.05)
#         # plt.savefig('lv-10-pred.png', bbox_inches='tight', pad_inches=0)
#         # plt.show()

#     #     result_writer.writerow(my_pso.get_gBest())
#     #     result_writer.writerow([mse])
#     # mean_fit = np.mean(fits)
#     # print(mean_fit)
#     # result_writer.writerow([mean_fit])