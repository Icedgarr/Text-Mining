import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_iter = 100
prop_perplexity = 0.05
count_perp = []
for i in range(n_iter):
    if (i % (round(n_iter * prop_perplexity) + 1)) == 0:
        count_perp.append(i)
count_perp = np.array(count_perp)

perp_2 = np.load("/media/chpmoreno/TOSHIBA/results/2/perplexity.npy")
perp_10 = np.load("/media/chpmoreno/TOSHIBA/results/10/perplexity.npy")
perp_50 = np.load("/media/chpmoreno/TOSHIBA/results/50/perplexity.npy")

d = {'k=2': perp_2, 'k=10': perp_10, 'k=50': perp_50}
index = count_perp.tolist()
df = pd.DataFrame(data = d, index = index)

plt.plot(df)
plt.ylabel('perplexity')
plt.xlabel('iteration')
plt.title('perplexity across different K')
plt.legend(["k=2", "k=3", "k=5"])
plt.show()

theta_2 = np.load("/media/chpmoreno/TOSHIBA/results/2/theta.npy")
theta_10 = np.load("/media/chpmoreno/TOSHIBA/results/10/theta.npy")
theta_50 = np.load("/media/chpmoreno/TOSHIBA/results/50/theta.npy")


np.average(np.array(theta_2), axis=0).mean(axis=0)
np.average(np.array(theta_10), axis=0).mean(axis=0)
np.average(np.array(theta_50), axis=0).mean(axis=0)
