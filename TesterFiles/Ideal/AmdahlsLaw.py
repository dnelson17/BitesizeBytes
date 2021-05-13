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

plt.legend(["\u03B1=0% - Ideal","\u03B1=0.1%","\u03B1=1%","\u03B1=10%","\u03B1=50%"])
plt.xlabel("Number of Processors (P)")
plt.ylabel("Runtime Speedup (S)")
p = Path.cwd()
plt.savefig(f"{p.parent.parent}\Figures\Ideal\AmdahlsLawPercentages.png")
