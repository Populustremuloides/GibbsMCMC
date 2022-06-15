import scipy.stats as stats
import numpy as np

xs = np.linspace(0,10,10)
ys = [stats.lognorm.pdf(x, s=1, loc=1,scale=1) for x in xs]
print(ys)