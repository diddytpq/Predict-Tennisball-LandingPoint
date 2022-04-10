import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

np.random.seed(0)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 5

mu = 0.17721273
sigma = 0.1214032
x = np.linspace(-1.5 + mu , 1.5 + mu, 1000)
y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
y_cum = 0.5 * (1 + erf((x - mu)/(np.sqrt(2 * sigma**2))))

plt.plot(x, y, 'darkorange', alpha=1, linewidth = 3 ,label='PDF of N({}, {})'.format(np.round(mu,2),np.round(sigma,2)))
# plt.plot(x, y_cum, alpha=0.7, label='CDF of N(0, 1)')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.legend(loc='upper left')
plt.grid()
plt.show()