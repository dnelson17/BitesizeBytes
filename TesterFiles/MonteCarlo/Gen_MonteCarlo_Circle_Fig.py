from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

def monte_carlo(attempts):
    bullets_x = np.random.rand(1,attempts)
    bullets_y = np.random.rand(1,attempts)
    dots_x = np.square( bullets_x )
    dots_y = np.square( bullets_y )
    hits_x = []
    hits_y = []
    misses_x = []
    misses_y = []
    for i in range(attempts):
        if dots_x[0,i] + dots_y[0,i] <= 1:
            hits_x.append(bullets_x[0,i])
            hits_y.append(bullets_y[0,i])
        else:
            misses_x.append(bullets_x[0,i])
            misses_y.append(bullets_y[0,i])
    return hits_x, hits_y, misses_x, misses_y

hits_x, hits_y, misses_x, misses_y = monte_carlo(5000)

plt.scatter(hits_x,hits_y,color=["lime"],s=4)
plt.scatter(misses_x,misses_y,color=["red"],s=4)

t = np.linspace(0,np.pi/2,100)
plt.plot(np.cos(t), np.sin(t), linewidth=5)

plt.gca().set_aspect('equal')

p = Path.cwd()

plt.savefig(f"{p.parent.parent}\Figures\MonteCarlo_Pi.png")
