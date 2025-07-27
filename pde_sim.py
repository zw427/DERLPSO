from DERLSPO.PDE_Demo import heat, helmholtz, transient
import numpy as np
from DERLSPO.PDE import Model
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.stats import truncnorm


np.random.seed(100)

class ht():
    def __init__(self):
        self.func = heat
        self.mu = 0.5
        self.sigma = 0.5
        self.lower, self.upper = 0.0001, 1
        self.a, self.b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
        truncated_normal_dist = truncnorm(self.a, self.b, loc=self.mu, scale=self.sigma)
        self.x = truncated_normal_dist.rvs()

class hh():
    def __init__(self):
        self.func = helmholtz
        self.mu = 0.5
        self.sigma = 0.5
        self.lower, self.upper = 0.0001, 1
        self.a, self.b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
        truncated_normal_dist = truncnorm(self.a, self.b, loc=self.mu, scale=self.sigma)
        self.x = truncated_normal_dist.rvs()


class ts():
    def __init__(self):
        self.func = transient
        self.mu = 0.5
        self.sigma = 0.5
        self.lower, self.upper = 0.0001, 1
        self.a, self.b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
        truncated_normal_dist = truncnorm(self.a, self.b, loc=self.mu, scale=self.sigma)
        self.x = truncated_normal_dist.rvs()


output=[]

for tp in [5, 10, 20, 40]:
    for rep in range(100):

        f = ht()

        data = f.func([f.a])

        model = Model(heat, 1, data)
        model.initParticles()
        model.iterator()
        

        est_params = model.getGBest()
        err = est_params - f.a
        fit  = f.func([est_params])

        mse0 = mean_squared_error(data, fit)

        output.append({
            "rep": rep,
            "est_params": est_params,
            "true_params": [f.a],
            "err": err.tolist,
            "mse0": mse0
        })

        print(
            f"rep={rep:3d},"
            f"true_params={[f.a]}, "
            f"est_params={np.atleast_1d(est_params).tolist()}, "
            f"err={np.atleast_1d(err).tolist()}, MSE={mse0:.5e}"
        )

df = pd.DataFrame(output)
func_name = f.func.__name__
df.to_csv(f"{func_name}_output.csv", index=False)


err_array = np.vstack(df["err"].to_numpy())
mse_array = df["mse0"].to_numpy()

print("\n==========  SUMMARY  ==========")
print("Mean parameter error :", err_array.mean(axis=0))
print("Std. parameter error :", err_array.std(axis=0))
print("Mean MSE            :", mse_array.mean())
print("Std. MSE            :", mse_array.std())
