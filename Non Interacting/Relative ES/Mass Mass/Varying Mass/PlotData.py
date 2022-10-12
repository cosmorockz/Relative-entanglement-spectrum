import numpy as np
import matplotlib.pyplot as plt

n_cutoff = 10
mz = 2
Mz1 = -2
# Mz2 = -4.0
M1,EE1 = np.loadtxt("M1_"+str(mz)+"M2_"+str(Mz1)+"Cutoff"+str(n_cutoff)+".dat",unpack=True,usecols=(0,1))
# M2,EE2 = np.loadtxt("M1_"+str(mz)+"M2_"+str(Mz2)+"Cutoff"+str(n_cutoff)+".dat",unpack=True,usecols=(0,1))
# Mz = Mz1

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('axes', axisbelow=True)

# fig=plt.figure(figsize=(10, 5))

fig,ax1 = plt.subplots(1,1,sharex=True,sharey=True,figsize=(7,12))

# plt.xticks([-10,-5,0,5,10], fontsize=16)
# ax = fig.gca()
# ax.yaxis.set_tick_params(width=2, length=5)
# ax.xaxis.set_tick_params(width=2, length=5)
# for axes in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axes].set_linewidth(0.5)

ax = fig.gca()

# Major ticks every 20, minor ticks every 5
xmajor_ticks = np.arange(-10, 15, 5)
xminor_ticks = np.arange(-10, 15, 2.5)
ymajor_ticks = np.arange(0, 1.2, 0.2)
yminor_ticks = np.arange(0, 1.2, 0.1)

ax.set_xticks(xmajor_ticks)
ax.set_xticks(xminor_ticks, minor=True)
ax.set_yticks(ymajor_ticks)
ax.set_yticks(yminor_ticks, minor=True)

# And a corresponding grid
ax1.grid(which='both')
# ax2.grid(which='both')

# Or if you want different settings for the grids:
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.5)
# ax2.grid(which='minor', alpha=0.2)
# ax2.grid(which='major', alpha=0.5)

# ax1 = axs[0,0]
# ax2 = axs[1,0]

ax1.yaxis.set_tick_params(width=2, length=5)
ax1.xaxis.set_tick_params(width=2, length=5)
for axes in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axes].set_linewidth(0.5)

# ax2.yaxis.set_tick_params(width=2, length=5)
# ax2.xaxis.set_tick_params(width=2, length=5)
# for axes in ['top', 'bottom', 'left', 'right']:
#     ax2.spines[axes].set_linewidth(0.5)

# plt.grid()
ax1.scatter(M1,EE1,s=24,color="#2E4272",marker='x')
# ax2.scatter(M2,EE2,s=24,color="#2E4272",marker='x')
ax1.grid(True)
# ax2.grid(True)
ax1.set_ylabel("$\lambda$",color="#061539")
# ax2.set_ylabel("$\lambda$",color="#061539")
plt.xlabel("m",color="#061539")
# plt.ylabel("$\lambda$",color="#061539")
# plt.margins(y=8)
plt.xlim([-10,10])
# plt.title("$M_{Lower \; Half}=$"+str(mz)+";$M_{Upper \; Half}=$"+str(Mz),color="#061539")
# plt.savefig("M1_"+str(mz)+"M2_"+str(Mz)+"Cutoff"+str(n_cutoff)+".pdf",bbox_inches="tight",pad_inches=0.4)
plt.savefig("mass2_mass4_-4"+".pdf",bbox_inches="tight",pad_inches=0.4)
plt.show()



