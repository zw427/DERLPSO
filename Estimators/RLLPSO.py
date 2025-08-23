import numpy as np
from scipy.integrate import odeint

from equation import Equation

from Estimators.estimator import Estimator


class PSO_ALGORITHM:
    def __init__(self, func: Equation, data, times, particle_num=100,
                 max_iter=200, layers_list=[4, 6, 8, 10], 
                 upper=10, lower=1e-10):
        
        self.func = func
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.upper = upper
        self.lower = lower

        self.pN = particle_num
        self.dim = self.func.num_param
        self.max_iter = max_iter
        self.X = np.zeros((self.pN, self.dim))
        self.V = np.zeros((self.pN, self.dim))
        self.pBest = np.zeros((self.pN, self.dim))
        self.gBest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.fit = 1e20

        self.initial_conditions = data[0, ]
        self.t = times
        self.actual_data = data

        self.numberOfLayers_list = layers_list
        self.qTable = np.zeros((len(self.numberOfLayers_list), len(self.numberOfLayers_list)))
        self.preState = 0
        self.currentState = 0

        self.layers = []

        self.phi = 0.4

        self.epsilon = 0.9
        self.alpha = 0.4
        self.gamma = 0.8

        self.iteratorMse = []

    def get_data(self):
        return self.actual_data

    def mse_loss(self, X):
        """Calculate MSE loss between predicted and actual data."""
        temp_x = tuple(X),
        predicted_data = odeint(self.func.f(), self.initial_conditions, self.t,
                                    args=temp_x, tfirst=self.func.t_first)
        mse = np.mean((self.actual_data - predicted_data) ** 2)
        return mse

    def select_action(self):
        """
        Select action using epsilon-greedy strategy.
        
        Returns:
            Selected number of layers
        """
        if np.random.rand() < self.epsilon:
            nextAction = np.argmax(self.qTable[self.currentState])
        else:
            nextAction = np.random.randint(0, len(self.numberOfLayers_list))

        self.preState = self.currentState
        self.currentState = nextAction
        return self.numberOfLayers_list[nextAction]

    def divide_particles(self, currentTotalLayer, fitness):
        """
        Divide particles into layers based on fitness.
        
        Args:
            current_total_layer: Current number of layers
            fitness: Fitness values of all particles
        """
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

    def level_competition(self, lec, FECount):
        """
        Perform level competition to select exemplar levels.
        
        Args:
            lec: Current level
            fe_count: Function evaluation count
            
        Returns:
            List of two exemplar levels
        """
        prob = (FECount / self.max_iter) ** 2
        exemplarLevels = [None, None]
        for i in range(2):
            if np.random.random() < prob:
                lec1 = np.random.randint(0, lec - 1)
                lec2 = np.random.randint(0, lec - 1)
                if lec1 < lec2:
                    exemplarLevels[i] = lec1
                else:
                    exemplarLevels[i] = lec2
            else:
                exemplarLevels[i] = np.random.randint(0, lec - 1)

        if exemplarLevels[1] < exemplarLevels[0]:
            exemplarLevels[0], exemplarLevels[1] = exemplarLevels[1], exemplarLevels[0]
        return exemplarLevels

    def init_population(self):
        """Initialize population with random positions and velocities."""
        for i in range(self.pN):
            if (i <= self.pN / 2):
                self.X[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1, self.dim)) * np.random.choice([-1, 1], self.dim)
                self.V[i] = np.exp(np.log(self.lower) + np.log(self.upper / self.lower) * np.random.uniform(0, 1, self.dim)) * np.random.choice([-1, 1], self.dim)
            else:
                self.X[i] = [np.random.uniform(-10, 10) for _ in range(self.dim)]
                self.V[i] = [np.random.uniform(-10, 10) for _ in range(self.dim)]
            self.pBest[i] = self.X[i]
            tmp = self.mse_loss(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gBest = self.X[i]

    def iterator(self):
        """Main optimization iteration loop."""
        for t in range(self.max_iter):
            pregBest = self.gBest
            currentTotalLayer = self.select_action()
            fitness = [self.mse_loss(x) for x in self.X]

            self.divide_particles(currentTotalLayer, fitness)
            for i in range(currentTotalLayer - 1, 1, -1):
                for j in self.layers[i]:
                    exemplarLevels = self.level_competition(i, t)
                    if(exemplarLevels[0] == exemplarLevels[1]):
                        index1 = np.random.randint(0, len(self.layers[exemplarLevels[0]]) - 2)
                        index2 = np.random.randint(index1 + 1, len(self.layers[exemplarLevels[0]]) - 1)
                        id1 = self.layers[exemplarLevels[0]][index1]
                        id2 = self.layers[exemplarLevels[0]][index2]

                    else:

                        id1 = np.random.choice(self.layers[exemplarLevels[0]])
                        id2 = np.random.choice(self.layers[exemplarLevels[1]])

                    X1 = self.X[id1]
                    X2 = self.X[id2]

                    self.r1 = np.random.uniform(0, 1)
                    self.r2 = np.random.uniform(0, 1)
                    self.r3 = np.random.uniform(0, 1)

                    self.V[j] = self.r1 * self.V[j] + self.r2 * (X1 - self.X[j]) + self.r3 * self.phi * (X2 - self.X[j])
                    self.X[j] = self.X[j] + self.V[j]

            for k in self.layers[1]:
                index1 = np.random.randint(0, len(self.layers[0]) - 2)
                index2 = np.random.randint(index1 + 1, len(self.layers[0]) - 1)
                id1 = self.layers[0][index1]
                id2 = self.layers[0][index2]
                X1 = self.X[id1]
                X2 = self.X[id2]

                self.r1 = np.random.uniform(0, 1)
                self.r2 = np.random.uniform(0, 1)
                self.r3 = np.random.uniform(0, 1)

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
                temp = self.mse_loss(self.X[i])
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pBest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:
                        self.gBest = self.X[i]
                        self.fit = self.p_fit[i]

            preFitness = self.mse_loss(pregBest)
            curFitness = self.mse_loss(self.gBest)
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
    def __init__(self, func: Equation, particle_num=100, max_iter=200, 
                 layers_list=[4, 6, 8, 10], upper=10, lower=1e-10):
        self.func = func
        self.particle_num = particle_num
        self.max_iter = max_iter
        self.layers_list = layers_list
        self.upper = upper
        self.lower = lower

    def train(self):
        raise Exception("RLLPSO does not need to train.")

    def predict(self, data, time):
        estimator = PSO_ALGORITHM(
            self.func, data, time,
            self.particle_num, self.max_iter, self.layers_list,
            self.upper, self.lower
        )
        estimator.init_population()
        estimator.iterator()
        return estimator.get_global_best()
