from pathlib import Path

#p = Path('.')
#print([x for x in p.iterdir() if x.is_dir()])

#print(list(p.glob('**/*.py')))

p = Path.cwd()
print(p)

print(list(p.parents))

print(f"{list(p.parents)[0]}\Figures")

print(f"{p.parent}\Figures")

print(f"{p.parent.parent}\Figures\MonteCarlo_Pi.png")
