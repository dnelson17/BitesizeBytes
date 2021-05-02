from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

def S(p,alpha):
    return p/(p*alpha+(1-alpha))


alpha_list = [0,0.001,0.01,0.1,0.5]
x = np.linspace(1,32,32)

for alpha in alpha_list:
    print(f"S(x,{alpha})={max(S(x,alpha))}")
    plt.plot(x, S(x,alpha))

plt.legend(["0%","0.1%","1%","10%","50%"])
plt.xlabel("Number of Processors")
plt.ylabel("Runtime Speedup")
p = Path.cwd()
plt.savefig(f"{p.parent.parent}\Figures\Ideal\AmdahlsLawPercentages.png")
