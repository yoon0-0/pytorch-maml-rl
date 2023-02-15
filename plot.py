import numpy as np
import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')

def plot_combined(name, results, color):
    results = np.array(results)
    xs = np.arange(results.shape[1]) * 300 * 800
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.fill_between(xs, ys-yerrs, ys+yerrs, alpha=0.2, color=color)
    plt.plot(xs, ys, label=name, c=color)

names  = ['maml_train']
# names  = ['snapbot_6', 'snapbot_5', 'snapbot_4']
seeds  = [1]
lists  = [[] for _ in range(len(names))]
colors = [plt.cm.rainbow(a) for a in np.linspace(0.0,1.0,len(names))]

for name_idx, name in enumerate(names):
    for seed in seeds:
        lists[name_idx].append(np.load('snapbot-maml-{}/reward{}.npy'.format(seed, seed))[:500])

plt.figure(figsize=(10, 5))
plt.axis([0, 500*300*800, -200, 200])
for name_idx, name in enumerate(names):
    plot_combined(name=name, results=lists[name_idx], color=colors[name_idx])
plt.xlabel('step'); plt.ylabel('reward'); plt.legend()
plt.savefig('maml_train.png')
plt.show()
