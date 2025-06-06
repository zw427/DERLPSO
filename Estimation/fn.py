import numpy as np
import matplotlib.pyplot as plt

# 参数值
sigma = 10.0
rho = 28.0

# 时间步长与总时间
dt = 0.01
t_end = 200.0

# 初始条件
x = y = 1.0

# 预分配数组
x_sol, y_sol = [x], [y]
t_sol = [0.0]

# 求解微分方程
for t in np.arange(0, t_end, dt):
    x_dot = x - x ** 3 / 3 - y
    y_dot = x - y

    x += x_dot * dt
    y += y_dot * dt

    x_sol.append(x)
    y_sol.append(y)
    t_sol.append(t)

# 绘制解曲线
plt.plot(t_sol, x_sol, 'r-', label='x(t)')
plt.plot(t_sol, y_sol, 'g-', label='y(t)')

plt.xlabel('Time t')
plt.legend()
plt.show()