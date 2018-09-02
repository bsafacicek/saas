import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.close("all")

# plot losses_all
colors = ['blue','green','red','cyan','magenta']
corrs = [0.0, 0.25, 0.5, 0.75,1.0]
keys = ["0.0","0.25","0.5","0.75","1.0"]
file_name = "training_losses.npy"
logs = np.load(file_name)[()]

font = 15
linew = 3
fig = plt.figure()
ax = plt.subplot(111)
legends = []
last_ep = 30
T = 10 # period at which screenshot is taken
loss_at_T=[]; std_at_T=[] 
for i, key in enumerate(keys):
    losses = logs[key]
    means = np.average(losses, axis=0)
    stds = np.std(losses, axis=0)    
    ax.errorbar(np.arange(0, last_ep), np.array(means[0:last_ep]),stds[0:last_ep], \
             linewidth=linew, color=colors[i])
    plt.fill_between(np.arange(0, last_ep), np.array(means[0:last_ep])-\
                   stds[0:last_ep], np.array(means[0:last_ep])+\
                   stds[0:last_ep], color=colors[i], alpha=0.1)
    plt.ylabel('Training loss', fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    legends.append('Label corruption=%s' % key)
    # loss at T:
    loss_at_T.append(means[T])
    std_at_T.append(stds[T])
    
plt.legend(legends, loc='best', fontsize=font)
plt.xlim(0, last_ep-1)
plt.ylim(0.1, 2.5)
ax.grid(color='gray', linestyle='dashdot', linewidth=1)
fig.savefig('random_loss.png')

fig = plt.figure()
ax = plt.subplot(111)
loss_at_T = np.array(loss_at_T); std_at_T = np.array(std_at_T)
ax.errorbar(corrs, loss_at_T, std_at_T,linestyle='dashed',marker='o', linewidth=linew,color=colors[-1])
plt.fill_between(corrs,loss_at_T-std_at_T,loss_at_T+std_at_T,color=colors[-1],alpha=0.1)
plt.ylabel('Training loss after %d epochs'%T, fontsize=font)
plt.xlabel('Label corruption', fontsize=font)
ax.grid(color='gray', linestyle='dashdot', linewidth=1)
fig.savefig('loss_reduction.png')

