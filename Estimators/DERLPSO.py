from scipy.integrate import odeint
import numpy as np

from Estimators.estimator import Estimator
from equations import Equation, ODE_Equation, PDE_Equation
from sklearn.metrics import mean_squared_error

class DERLPSO(Estimator):
    def __init__(self, func: Equation, param_num, data, times, particle_num=100,
                 max_iter=200, layers_list=[4, 6, 8, 10], upper=10,
                 lower=1e-10, threshold=1e-04):

        self.func = func
        self.upper = upper
        self.lower = lower
        self.particle_num = particle_num
        self.param_num = param_num
        self.max_iter = max_iter
        self.layers_list = layers_list

        # Initialize particle arrays
        self.X = np.zeros((self.particle_num, self.param_num))
        self.V = np.zeros((self.particle_num, self.param_num))
        self.p_best = np.zeros((self.particle_num, self.param_num))
        self.global_best = np.zeros((1, self.param_num))
        self.p_fit = np.zeros(self.particle_num)
        self.fit = float('inf')

        # Data and time setup
        if isinstance(self.func, ODE_Equation):
            self.initial = data[0, ]
            self.times = times
            self.data = data
        elif isinstance(self.func, PDE_Equation):
            self.data = data
            self.times = None
        else:
            raise Exception("Only ODEs and PDEs are supported.")

        # Q-learning parameters
        self.q_table = np.zeros((len(self.layers_list), len(self.layers_list)))
        self.pre_state = 0
        self.current_state = 0

        # Layer and fitness tracking
        self.layers = []
        self.fitness = []

        # Algorithm parameters
        self.phi = 0.4
        self.epsilon = 0.9
        self.alpha = 0.4
        self.gamma = 0.8
        self.threshold = threshold

    def get_data(self):
        return self.data
    
    def get_global_best(self):
        return self.global_best

    def get_fit(self):
        return self.fit

    def mse_loss(self, X):# Debugging line to inspect X
        temp_x = tuple(X),
        if isinstance(self.func, ODE_Equation):
            predicted_data = odeint(self.func.f(), self.initial, self.times,
                                    args=temp_x, tfirst=True)
        elif isinstance(self.func, PDE_Equation):
            predicted_data = self.func.f()(X)

        mse = np.mean((self.data - predicted_data) ** 2)
        return mse

    def select_action(self):
        """
        Select action using epsilon-greedy strategy.
        
        Returns:
            Selected number of layers
        """
        if np.random.rand() < self.epsilon:
            next_action = np.argmax(self.q_table[self.current_state])
        else:
            next_action = np.random.randint(0, len(self.layers_list))

        self.pre_state = self.current_state
        self.current_state = next_action
        return self.layers_list[next_action]

    def divide_particles(self, current_total_layer, fitness):
        """
        Divide particles into layers based on fitness.
        
        Args:
            current_total_layer: Current number of layers
            fitness: Fitness values of all particles
        """
        base_count = self.particle_num // current_total_layer
        remainder = self.particle_num % current_total_layer
        layer_counts = ([base_count] * (current_total_layer - 1) +
                       [base_count + remainder])

        particles = list(range(0, self.particle_num))
        sorted_particles = [x for _, x in sorted(zip(fitness, particles))]

        self.layers.clear()
        start_idx = 0
        for count in layer_counts:
            end_idx = start_idx + count
            self.layers.append(sorted_particles[start_idx:end_idx])
            start_idx = end_idx

    def level_competition(self, lec, fe_count):
        """
        Perform level competition to select exemplar levels.
        
        Args:
            lec: Current level
            fe_count: Function evaluation count
            
        Returns:
            List of two exemplar levels
        """
        prob = (fe_count / self.max_iter) ** 2
        exemplar_levels = [None, None]
        
        for i in range(2):
            if np.random.random() < prob:
                lec1 = np.random.randint(0, lec - 1)
                lec2 = np.random.randint(0, lec - 1)
                exemplar_levels[i] = min(lec1, lec2)
            else:
                exemplar_levels[i] = np.random.randint(0, lec - 1)

        if exemplar_levels[1] < exemplar_levels[0]:
            exemplar_levels[0], exemplar_levels[1] = (exemplar_levels[1],
                                                     exemplar_levels[0])
        return exemplar_levels

    def init_particles(self):
        """Initialize particles with random positions and velocities."""
        for i in range(self.particle_num):
            if i <= self.particle_num / 2:
                # Log-uniform initialization
                self.X[i] = (np.exp(np.log(self.lower) +
                           np.log(self.upper / self.lower) *
                           np.random.uniform(0, 1, self.param_num)) *
                           np.random.choice([-1, 1], self.param_num))
                self.V[i] = (np.exp(np.log(self.lower) +
                           np.log(self.upper / self.lower) *
                           np.random.uniform(0, 1, self.param_num)) *
                           np.random.choice([-1, 1], self.param_num))
            else:
                # Uniform initialization
                if isinstance(self.func, ODE_Equation):
                    self.X[i] = [np.random.uniform(-self.upper, self.upper)
                            for _ in range(self.param_num)]
                    self.V[i] = [np.random.uniform(-self.upper, self.upper)
                            for _ in range(self.param_num)]
                elif isinstance(self.func, PDE_Equation):
                    self.X[i] = [np.uniform(0, self.upper) for _ in range(self.paramNum)]
                    self.V[i] = [np.random.uniform(0, self.upper) for _ in range(self.paramNum)]
            
            self.p_best[i] = self.X[i]
            tmp = self.mse_loss(self.X[i])
            self.fitness.append(tmp)
            self.p_fit[i] = tmp
            
            if tmp < self.fit:
                self.fit = tmp
                self.global_best = self.X[i]

    def iterator(self):
        """Main optimization iteration loop."""
        for t in range(self.max_iter):
            pre_g_best = self.global_best
            current_total_layer = self.select_action()

            self.divide_particles(current_total_layer, self.fitness)
            
            # Update particles in layers 2 to current_total_layer-1
            for i in range(current_total_layer - 1, 1, -1):
                for j in self.layers[i]:
                    exemplar_levels = self.level_competition(i, t)
                    
                    if exemplar_levels[0] == exemplar_levels[1]:
                        index1 = np.random.randint(
                            0, len(self.layers[exemplar_levels[0]]) - 2)
                        index2 = np.random.randint(
                            index1 + 1, len(self.layers[exemplar_levels[0]]) - 1)
                        id1 = self.layers[exemplar_levels[0]][index1]
                        id2 = self.layers[exemplar_levels[0]][index2]
                    else:
                        id1 = np.random.choice(self.layers[exemplar_levels[0]])
                        id2 = np.random.choice(self.layers[exemplar_levels[1]])
                    
                    X1 = self.X[id1]
                    X2 = self.X[id2]

                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    r3 = np.random.uniform(0, 1)

                    self.V[j] = (r1 * self.V[j] + r2 * (X1 - self.X[j]) +
                               r3 * self.phi * (X2 - self.X[j]))
                    self.X[j] = self.X[j] + self.V[j]

            # Update particles in layer 1
            for k in self.layers[1]:
                index1 = np.random.randint(0, len(self.layers[0]) - 2)
                index2 = np.random.randint(index1 + 1, len(self.layers[0]) - 1)
                id1 = self.layers[0][index1]
                id2 = self.layers[0][index2]
                X1 = self.X[id1]
                X2 = self.X[id2]

                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                r3 = np.random.uniform(0, 1)

                self.V[k] = (r1 * self.V[k] + r2 * (X1 - self.X[k]) +
                           r3 * self.phi * (X2 - self.X[k]))
                self.X[k] = self.X[k] + self.V[k]

            # Restart particles if needed at mid-iteration
            if t == int(self.max_iter / 2) and self.fit > self.threshold:
                for i in range(self.particle_num):
                    self.X[i] = (np.exp(np.log(self.lower) +
                               np.log(self.upper / self.lower) *
                               np.random.uniform(0, 1, self.param_num)) *
                               np.random.choice([-1, 1], self.param_num))
                    self.V[i] = (np.exp(np.log(self.lower) +
                               np.log(self.upper / self.lower) *
                               np.random.uniform(0, 1, self.param_num)) *
                               np.random.choice([-1, 1], self.param_num))

            # Update fitness and best positions
            self.fitness.clear()
            for i in range(self.particle_num):
                temp = self.mse_loss(self.X[i])
                self.fitness.append(temp)
                
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.p_best[i] = self.X[i]
                    
                    if self.p_fit[i] < self.fit:
                        self.global_best = self.X[i]
                        self.fit = self.p_fit[i]

            # Update Q-table
            pre_fitness = self.mse_loss(pre_g_best)
            cur_fitness = self.mse_loss(self.global_best)
            reward = (abs(cur_fitness - pre_fitness) /
                     abs(max(cur_fitness, 1e-10)))
            new_q = (self.q_table[self.pre_state][self.current_state] +
                    self.alpha * (reward + self.gamma *
                                 max(self.q_table[self.current_state]) -
                                 self.q_table[self.pre_state][self.current_state]))
            self.q_table[self.pre_state][self.current_state] = new_q

    def train(self):
        Exception("DERLPSO does not need to train.")

    def predict(self, param):

        self.init_particles()
        self.iterator()

        est_params = np.asarray(self.get_global_best())
        err = est_params - param
        temp_est_params = tuple(est_params),
        fit  = odeint(self.func.f(), self.initial, self.times, args=temp_est_params, tfirst=True)
        mse = mean_squared_error(self.data, fit)

        return {
                "true_params": param,
                "est_params": est_params.tolist(),
                "err": err.tolist(),
                "mse0": mse
            }
