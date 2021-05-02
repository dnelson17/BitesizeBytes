from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

def S(p,f):
    return p/(p*f+(1-f))


f_list = [0,0.001,0.01,0.1,0.5]
x = np.linspace(1,32,32)

for f in f_list:
    plt.plot(x, S(x,f))

plt.legend(["0%","0.1%","1%","10%","50%"])
plt.show()
p = Path.cwd()
#plt.savefig(f"{p.parent.parent}\Figures\MonteCarlo\MonteCarlo_Pi.png")
