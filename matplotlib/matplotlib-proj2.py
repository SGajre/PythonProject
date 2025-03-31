from ast import increment_lineno

import inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%matplotlib inline

x1 = np.linspace(0,10,100)

#fig = plt.figure()

plt.plot(x1, np.sin(x1), '-')
plt.plot(x1, np.cos(x1), '--')
plt.plot(x1, np.tan(x1), '*')
plt.show()