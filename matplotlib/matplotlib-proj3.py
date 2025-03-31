
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x1 = np.linspace(0,10,100)

fig = plt.figure()
plt.subplot(2,1,1)

plt.plot(x1, np.sin(x1), '-')

plt.subplot(3,1,1)

plt.plot(x1, np.cos(x1), '-')
plt.show()

plt.plot([1,8,27,81], [2,16,64,96], 'go')
plt.show()