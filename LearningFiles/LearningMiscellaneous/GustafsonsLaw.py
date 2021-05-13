from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

def S(p,alpha):
    return p**2/(p*(1-alpha)+alpha)


alpha_list = [0,0.001,0.01,0.1,0.5]
x = np.linspace(1,32,32)

for alpha in alpha_list:
    print(f"S(x,{alpha})={max(S(x,alpha))}")
    plt.plot(x, S(x,alpha))

plt.legend(["0%","0.1%","1%","10%","50%"])
plt.show()
p = Path.cwd()
#plt.savefig(f"{p.parent.parent}\Figures\MonteCarlo\MonteCarlo_Pi.png")
