from DERLSPO.ODE_Demo import lotka_volterra, fitzhugh_nagumo, lorenz
import numpy as np
from scipy.integrate import odeint
from DERLSPO.ODE import Model
from sklearn.metrics import mean_squared_error
import pandas as pd

np.random.seed(100)


class fn():
    def __init__(self):
        self.func = fitzhugh_nagumo
        self.y0   = [0.0, 0.0]
        a = np.random.normal(0.4, 0.1)
        b = np.random.normal(1.3, 0.1)
        c = np.random.normal(1.0, 0.1)
        d = np.random.normal(1.0, 0.1)
        self.param = np.array([a, b, c, d])
        self.paramNum = len(self.param)

class lv():
    def __init__(self):
        self.func = lotka_volterra
        self.y0   = [1.0, 1.0]
        α = np.random.normal(0.4, 0.1)
        β = np.random.normal(0.1, 0.1)
        γ = np.random.normal(0.4, 0.1)
        δ = np.random.normal(0.1, 0.1)
        self.param = np.array([α, β, γ, δ])
        self.paramNum = len(self.param)


class lz():
    def __init__(self):
        self.func = lorenz
        self.y0   = [1.0, 1.0, 1.0]
        σ = np.random.normal(10.0, 1.0)
        ρ = np.random.normal(28.0, 1.0)
        β = np.random.normal(8.0 / 3.0, 0.1)
        self.param = np.array([σ, ρ, β])
        self.paramNum = len(self.param)


output=[]

for tp in [5, 10, 20, 40]:
    for rep in range(100):

        t = np.arange(0.0, 4.0, step = 4.0 / tp)
        f = fn()

        data  = odeint(f.func, f.y0, t, args=tuple(f.param))

        import pdb; pdb.set_trace()

        model = Model(f.func, paramNum=f.paramNum, data=data, time=t, threshold=1e-4)
        model.initParticles()
        model.iterator()
        
        est_params = np.asarray(model.getGBest())
        err = est_params - f.param
        fit  = odeint(f.func, f.y0, t, args=tuple(est_params))

        mse0 = mean_squared_error(data, fit)

        output.append({
            "rep": rep,
            "tp": tp,
            "true_params": f.param.tolist(),
            "est_params": est_params.tolist(),
            "err": err.tolist(),
            "mse0": mse0
        })

        print(
            f"rep={rep:3d}, tp={tp:2d}, "
            f"true_params={f.param.tolist()}, "
            f"est_params={est_params.tolist()}, "
            f"err={err.tolist()}, MSE={mse0:.5e}"
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
