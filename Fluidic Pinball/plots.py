# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:10:50 2023

@author: QihongLi
"""
#%%PLOTTING parameter
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:16:24 2023

@author: itirelli
"""

import mat73
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

#%%
############### CHANNEL FLOW

Uinf = 7.5
EPTV = scipy.io.loadmat("EPTV_stat_cha.mat")
MeanPIV = scipy.io.loadmat("UmPIV_channel.mat")
UKK = mat73.loadmat("ChannelSynth_w10_img_1_11856_011001.mat")
UDNS = scipy.io.loadmat("UDNS11001.mat")
UPIV = scipy.io.loadmat("UPIV11001.mat")
RBF =  scipy.io.loadmat("ChannelSynth_img_1_11856_011001.mat")
X = RBF['X']
Y = RBF['Y']
UmEPTV = EPTV['UmEPTV']
VmEPTV = EPTV['VmEPTV']
XX = EPTV['XX']
YY = EPTV['YY']
UmEPTV[np.isnan(UmEPTV)] = 0
VmEPTV[np.isnan(VmEPTV)] = 0
UmHR = griddata((XX.flatten(), YY.flatten()), UmEPTV.flatten(), (X.flatten(), Y.flatten()), method='linear')
VmHR = griddata((XX.flatten(), YY.flatten()), VmEPTV.flatten(), (X.flatten(), Y.flatten()), method='linear') 
UmHR = np.reshape(UmHR,X.shape)
VmHR = np.reshape(VmHR,X.shape)
UPIVKK = UPIV['Upiv']-MeanPIV['UmPIV']+UmHR

# plt.rcParams['font.size'] = 10 # set the font size for all text elements
# plt.rcParams['axes.labelsize'] = 10 # set the font size for the axis labels
# plt.rcParams['xtick.labelsize'] = 10 # set the font size for the x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 10 # set the font size for the y-axis tick labels

# plt.rcParams['font.family'] = 'Latex'
plt.rcParams['font.size'] = 11
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

X = X/512
Y = Y/512
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',figsize = (5,5*0.56))
im = ax1.pcolormesh(X,Y,np.flipud(UPIVKK/Uinf), clim=[0.5, 1])
ax1.set_aspect('equal', adjustable='box')
ax1.set_yticks([0, 0.5, 1])
# ax1.text(0.05, 0.85, '$a) PIV$', transform=ax1.transAxes, color='black', fontsize=11, fontweight='bold')
ax2.pcolormesh(X,Y,np.flipud(UKK['UKNN']/Uinf), clim=[0.5, 1])
ax2.set_aspect('equal', adjustable='box')
ax2.set_yticks([0, 0.5, 1])
# ax2.text(0.05, 0.85, 'b) KNN-PTV', transform=ax2.transAxes, color='black', fontsize=11, fontweight='bold')
ax3.pcolormesh(X,Y,np.flipud(RBF['U']/Uinf), clim=[0.5, 1])
ax3.set_aspect('equal', adjustable='box')
ax3.set_yticks([0, 0.5, 1])
# ax3.text(0.05, 0.85, 'c) KNN-PTV + RBF', transform=ax3.transAxes, color='black', fontsize=11, fontweight='bold')
ax4.pcolormesh(X,Y,np.flipud(UDNS['UDNS']/Uinf), clim=[0.5, 1])
ax4.set_aspect('equal', adjustable='box')
ax4.set_yticks([0, 0.5, 1])
# ax4.text(0.05, 0.85, 'd) DNS', transform=ax4.transAxes, color='black', fontsize=11, fontweight='bold')
f.text(0.5, 0.02, '$x / h$', ha='center')
f.text(0.01, 0.5, '$y / h$', va='center', rotation='vertical')
cbar_ax = f.add_axes([0.93, 0.12, 0.015, 0.8]) # adjust position and size of the colorbar axis
cb = f.colorbar(im, cax=cbar_ax)
cb.ax.set_title('$U / U_b$')
# cb.ax.set_title('$U / U_b$', fontsize = 11, font = 'Times New Roman')
# cb.ax.tick_params(labelsize=10) # set the font size for the colorbar tick labels
f.subplots_adjust(left=0.09, right=0.90, bottom=0.1, top=0.95, wspace=0.1, hspace=0.1)
# f.tight_layout()
plt.savefig('Channel.png', dpi=300, bbox_inches='tight')

#%% TBL word
Uinf = 12.8

RBFK1 = scipy.io.loadmat("ChannelK1_img_1_30000_000001.mat")
DNS = mat73.loadmat("G:/.shortcut-targets-by-id/1u9q9kyDMW8T7KIY12Tji3k8rKFpQhzQ8/GANsPIV_experimental/database/PIV_HR/PIV_HR_000001.mat")  
PIV = mat73.loadmat("G:/.shortcut-targets-by-id/1u9q9kyDMW8T7KIY12Tji3k8rKFpQhzQ8/GANsPIV_experimental/database/PIV/PIV_000001.mat")
RBF =  scipy.io.loadmat("Channel_img_1_30000_000001.mat")
KNN = mat73.loadmat("Channel_w10_img_1_30000_000001.mat")

X = RBF['X']
Y = RBF['Y']
res = 48500
delta99 = 24.7
XX = X/(res*(delta99/1000))
YY = Y/(res*(delta99/1000))
XX = XX-min(XX.flatten())
YY = YY-min(YY.flatten())
YY = np.flipud(YY)

UPIV = griddata((PIV['XPIV'].flatten(), PIV['YPIV'].flatten()), PIV['U'].flatten(), (X.flatten(), Y.flatten()), method='nearest')
UPIV = np.reshape(UPIV,X.shape)
URBFK1 = RBFK1['U']
URBF = RBF['U']
UDNS = DNS['U'][:,4:128]
UKNN = KNN['UKNN'][:,4:128]

# f, ((ax1, ax2,ax5), (ax3, ax4,ax5)) = plt.subplots(2, 3, sharex='col', sharey='row',figsize = (7,7*0.56))
# im = ax1.pcolormesh(XX,YY,UPIV/Uinf, clim=[0.4, 1])
# ax1.set_aspect('equal', adjustable='box')
# ax1.set_yticks([0, 0.5, 1,1.5])
# ax1.text(0.05, 0.85, 'PIV', transform=ax1.transAxes, color='white', fontsize=12, fontweight='bold')
# ax2.pcolormesh(XX,YY,UKNN/Uinf, clim=[0.4, 1])
# ax2.set_aspect('equal', adjustable='box')
# ax2.set_yticks([0, 0.5, 1])
# ax2.text(0.05, 0.85, 'KNN-PTV', transform=ax2.transAxes, color='white', fontsize=12, fontweight='bold')
# ax3.pcolormesh(XX,YY,URBF/Uinf, clim=[0.4, 1])
# ax3.set_aspect('equal', adjustable='box')
# ax3.set_yticks([0, 0.5, 1])
# ax3.text(0.05, 0.85, 'KNN-PTV + RBF', transform=ax3.transAxes, color='white', fontsize=12, fontweight='bold')
# ax4.pcolormesh(XX,YY,UDNS/Uinf, clim=[0.4, 1])
# ax4.set_aspect('equal', adjustable='box')
# ax4.set_yticks([0, 0.5, 1])
# ax4.text(0.05, 0.85, 'DNS', transform=ax4.transAxes, color='white', fontsize=12, fontweight='bold')
# ax5.pcolormesh(XX,YY,UDNS/Uinf, clim=[0.4, 1])
# ax5.set_aspect('equal', adjustable='box')
# ax5.set_yticks([0, 0.5, 1])
# ax5.text(0.05, 0.85, 'DNS', transform=ax4.transAxes, color='white', fontsize=12, fontweight='bold')
# f.text(0.5, 0.02, '$X / \delta_{99}$', ha='center')
# f.text(0.01, 0.5, '$Y / \delta_{99}$', va='center', rotation='vertical')
# cbar_ax = f.add_axes([0.93, 0.12, 0.015, 0.8]) # adjust position and size of the colorbar axis
# cb = f.colorbar(im, cax=cbar_ax)
# cb.ax.set_title('$U / Ub$', fontsize = 10, font = 'Times New Roman')
# # cb.ax.tick_params(labelsize=10) # set the font size for the colorbar tick labels
# f.subplots_adjust(left=0.07, right=0.9, bottom=0.1, top=0.95, wspace=0.1, hspace=0.1)
# # f.tight_layout()
# # plt.savefig('myplot.png', dpi=300)
# plt.subplots(2, 3, sharex='col', sharey='row',figsize = (7,7*0.56))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

fig = plt.figure(constrained_layout=True,figsize = (7,7*0.56))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.pcolormesh(XX,YY,UPIV/Uinf, clim=[0.4, 1])
ax1.set_aspect('equal', adjustable='box')
ax1.set_yticks([0, 0.5, 1, 1.5])
ax1.text(0.05, 0.85, 'PIV128', transform=ax1.transAxes, color='black', fontsize=12, fontweight='bold')
ax1.set_yticks([0, 0.5, 1, 1.5])
ax1.set_xticks([0, 0.5, 1, 1.5])
ax1.set_xticklabels([])
ax1.set_yticklabels([0, 0.5, 1, 1.5])
ax2 = fig.add_subplot(gs[0, 1])
ax2.pcolormesh(XX,YY,UKNN/Uinf, clim=[0.4, 1])
ax2.set_aspect('equal', adjustable='box')
ax2.set_yticks([0, 0.5, 1, 1.5])
ax2.set_xticks([0, 0.5, 1, 1.5])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.text(0.05, 0.85, 'KNN-PTV', transform=ax2.transAxes, color='black', fontsize=12, fontweight='bold')
ax3 = fig.add_subplot(gs[:, 2])
ax3.pcolormesh(XX,YY,UDNS/Uinf, clim=[0.4, 1])
ax3.set_aspect('equal', adjustable='box')
ax3.set_yticks([0, 0.5, 1, 1.5])
ax3.set_xticks([0, 0.5, 1, 1.5])
ax3.set_xticklabels([0, 0.5, 1, 1.5])
ax3.set_yticklabels([0, 0.5, 1, 1.5])
ax3.text(0.05, 0.85, 'PIV32', transform=ax3.transAxes, color='black', fontsize=12, fontweight='bold')
ax4 = fig.add_subplot(gs[1, 0])
ax4.pcolormesh(XX,YY,URBFK1/Uinf, clim=[0.4, 1])
ax4.set_aspect('equal', adjustable='box')
ax4.set_yticks([0, 0.5, 1, 1.5])
ax4.set_xticks([0, 0.5, 1, 1.5])
ax4.set_xticklabels([0, 0.5, 1, 1.5])
ax4.set_yticklabels([0, 0.5, 1, 1.5])
ax4.text(0.05, 0.85, 'RBF', transform=ax4.transAxes, color='black', fontsize=12, fontweight='bold')
ax5 = fig.add_subplot(gs[1, 1])
ax5.pcolormesh(XX,YY,URBF/Uinf, clim=[0.4, 1])
ax5.set_aspect('equal', adjustable='box')
ax5.set_yticks([0, 0.5, 1, 1.5])
ax5.set_xticks([0, 0.5, 1, 1.5])
ax5.set_xticklabels([0, 0.5, 1, 1.5])
ax5.set_yticklabels([])
ax5.text(0.05, 0.85, 'KNN-PTV + RBF', transform=ax5.transAxes, color='black', fontsize=12, fontweight='bold')
fig.text(0.5, 0.02, '$X / \delta_{99}$', ha='center')
fig.text(0.01, 0.5, '$Y / \delta_{99}$', va='center', rotation='vertical')
cbar_ax = fig.add_axes([0.93, 0.11, 0.015, 0.82]) # adjust position and size of the colorbar axis
cb = fig.colorbar(im, cax=cbar_ax)
cb.ax.set_title('$U / Ub$', fontsize = 10, font = 'Times New Roman')
# cb.ax.tick_params(labelsize=10) # set the font size for the colorbar tick labels
fig.subplots_adjust(left=0.07, right=0.9, bottom=0.13, top=0.95, wspace=0.1, hspace=0.1)
plt.savefig('myplotTBL.png', dpi=300)

#%% TBL latex

Uinf = 12.8

# RBFK1 = scipy.io.loadmat("ChannelK1_img_1_30000_000001.mat")
DNS = mat73.loadmat("G:/.shortcut-targets-by-id/1u9q9kyDMW8T7KIY12Tji3k8rKFpQhzQ8/GANsPIV_experimental/database/PIV_HR/PIV_HR_000001.mat")  
PIV = mat73.loadmat("G:/.shortcut-targets-by-id/1u9q9kyDMW8T7KIY12Tji3k8rKFpQhzQ8/GANsPIV_experimental/database/PIV/PIV_000001.mat")
RBF =  scipy.io.loadmat("Channel_img_1_30000_000001.mat")
KNN = mat73.loadmat("Channel_w10_img_1_30000_000001.mat")

X = RBF['X']
Y = RBF['Y']
res = 48500
delta99 = 24.7
XX = X/(res*(delta99/1000))
YY = Y/(res*(delta99/1000))
XX = XX-min(XX.flatten())
YY = YY-min(YY.flatten())
YY = np.flipud(YY)

UPIV = griddata((PIV['XPIV'].flatten(), PIV['YPIV'].flatten()), PIV['U'].flatten(), (X.flatten(), Y.flatten()), method='nearest')
UPIV = np.reshape(UPIV,X.shape)
# URBFK1 = RBFK1['U']
URBF = RBF['U']
UDNS = DNS['U'][:,4:128]
UKNN = KNN['UKNN'][:,4:128]


# plt.rcParams['font.family'] = 'Latex'
plt.rcParams['font.size'] = 11
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, sharey='row', figsize=(5, 5*0.56))
im = ax1.pcolormesh(XX, YY, UPIV/Uinf, clim=[0.5, 1.1])
ax1.set_aspect('equal', adjustable='box')
ax1.set_yticks([0, 0.5, 1, 1.5])
ax1.set_xticks([0, 0.5, 1, 1.5])
ax2.pcolormesh(XX, YY, UKNN/Uinf, clim=[0.5, 1.1])
ax2.set_aspect('equal', adjustable='box')
ax2.set_xticks([0, 0.5, 1, 1.5])
ax3.pcolormesh(XX, YY, URBF/Uinf, clim=[0.5, 1.1])
ax3.set_aspect('equal', adjustable='box')
ax3.set_xticks([0, 0.5, 1, 1.5])
ax4.pcolormesh(XX, YY, UDNS/Uinf, clim=[0.5, 1.1])
ax4.set_aspect('equal', adjustable='box')
ax4.set_xticks([0, 0.5, 1, 1.5])


# Add a horizontal colorbar below the subplots
cb_ax = f.add_axes([0.1, 0.15, 0.86, 0.03])  # [left, bottom, width, height]
cb = f.colorbar(im, cax=cb_ax, orientation='horizontal')
cb.set_label( '$U/U_\infty$')

f.text(0.5, 0.25, '$x / \delta_{99}$', ha='center')
f.text(0.01, 0.5, '$y / \delta_{99}$', va='center', rotation='vertical')
f.subplots_adjust(left=0.1, right=0.97, bottom=0.13, top=0.98, wspace=0.15, hspace=0.6)
plt.savefig('myplotTBL.png', dpi=300)


#%% PSD
DNS = mat73.loadmat("G:/.shortcut-targets-by-id/1u9q9kyDMW8T7KIY12Tji3k8rKFpQhzQ8/GANsPIV_experimental/database/PIV_HR/PIV_HR_000001.mat")
RBF =  scipy.io.loadmat("Channel_img_1_30000_000001.mat")
# plt.rcParams['font.family'] = 'Latex'
plt.rcParams['font.size'] = 11
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

X = RBF['X']
Y = RBF['Y']
res = 48500
delta99 = 24.7
XX = X/(res*(delta99/1000))
YY = Y/(res*(delta99/1000))
XX = XX-min(XX.flatten())
YY = YY-min(YY.flatten())
YY = np.flipud(YY)
UDNS = DNS['U'][:,4:128]
Uinf = 12.8

PSD = scipy.io.loadmat("C:/Users/itirelli/Desktop/ISPIV/TBLspect.mat")
freq = PSD['freq'][:,0:len(PSD['freq'][0,:])//2].flatten()
freq128 = PSD['freq128'][:,0:len(PSD['freq128'][0,:])//2].flatten()
U32 = PSD['PSDu32_total'][:,0:len(PSD['PSDu32_total'][0,:])//2]
V32 = PSD['PSDv32_total'][:,0:len(PSD['PSDu32_total'][0,:])//2]
UK = PSD['PSDuK_total'][:,0:len(PSD['PSDuK_total'][0,:])//2]
VK = PSD['PSDvK_total'][:,0:len(PSD['PSDuK_total'][0,:])//2]
UKRBF = PSD['PSDuKRBF_total'][:,0:len(PSD['PSDuKRBF_total'][0,:])//2]
VKRBF = PSD['PSDvKRBF_total'][:,0:len(PSD['PSDuKRBF_total'][0,:])//2]
U128 = PSD['PSDu128_total'][:,0:len(PSD['PSDu128_total'][0,:])//2]
V128 = PSD['PSDv128_total'][:,0:len(PSD['PSDu128_total'][0,:])//2]

fig = plt.figure(constrained_layout=False,figsize = (5.3,5.3*0.56))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.loglog(freq,(U32[0,:].flatten()*freq)**(5/3),'k',linewidth=2.5)
ax1.loglog(freq128,(U128[0,:].flatten()*freq128)**(5/3),'grey')
ax1.loglog(freq,(UK[0,:].flatten()*freq)**(5/3),'-.k')
ax1.loglog(freq,(UKRBF[0,:].flatten()*freq)**(5/3),'--k')
ax1.spines['left'].set_color('red')
ax1.spines['bottom'].set_color('red')
ax1.spines['right'].set_color('red')
ax1.spines['top'].set_color('red')

ax1.set_xticklabels([])
ax1.set_ylim([1e-6, 1e-1]) # set the y-axis limits for ax1
# ax1.set_xlim(freq[0], freq[len(freq)-1]) # set the y-axis limits for ax1
plt.grid()
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(freq,(U32[1,:].flatten()*freq)**(5/3),'k',linewidth=2.5)
ax2.loglog(freq128,(U128[1,:].flatten()*freq128)**(5/3),'grey')
ax2.loglog(freq,(UK[1,:].flatten()*freq)**(5/3),'-.k')
ax2.loglog(freq,(UKRBF[1,:].flatten()*freq)**(5/3),'--k')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
legend_labels = ['$PIV32$','$PIV128$', '$KNN-PTV$', '$KNN-PTV+RBF$' ]
ax2.legend(legend_labels, loc='upper right', bbox_to_anchor=(2.33, 1))
ax2.spines['left'].set_color('blue')
ax2.spines['bottom'].set_color('blue')
ax2.spines['right'].set_color('blue')
ax2.spines['top'].set_color('blue')


ax2.set_ylim([1e-6, 1e-1]) # set the y-axis limits for ax1
plt.grid()
ax3 = fig.add_subplot(gs[1, 2])
ax3.pcolormesh(XX, YY, UDNS/Uinf, clim=[0.5, 1.1])
ax3.set_aspect('equal', adjustable='box')
ax3.set_xticks([0, 0.5, 1, 1.5])
ax3.set_yticks([0, 0.5, 1, 1.5])
ax3.plot(XX[0,:],np.zeros_like(XX[0,:])+0.1,'r')
ax3.plot(XX[0,:],np.zeros_like(XX[0,:])+0.7,'b')
ax3.set_xlabel('$x / \delta_{99}$')
ax3.set_ylabel('$y / \delta_{99}$')
# ax3.set_position([0.63, 0.08, 0.4, 0.4])
ax4 = fig.add_subplot(gs[1, 0])
ax4.loglog(freq,(V32[0,:].flatten()*freq)**(5/3),'k',linewidth=2.5)
ax4.loglog(freq128,(V128[0,:].flatten()*freq128)**(5/3),'grey')
ax4.loglog(freq,(VK[0,:].flatten()*freq)**(5/3),'-.k')
ax4.loglog(freq,(VKRBF[0,:].flatten()*freq)**(5/3),'--k')
ax4.set_ylim([1e-6, 1e-1])
ax4.spines['left'].set_color('red')
ax4.spines['bottom'].set_color('red')
ax4.spines['right'].set_color('red')
ax4.spines['top'].set_color('red')

plt.grid()
ax5 = fig.add_subplot(gs[1, 1])
ax5.loglog(freq,(V32[1,:].flatten()*freq)**(5/3),'k',linewidth=2.5)
ax5.loglog(freq128,(V128[1,:].flatten()*freq128)**(5/3),'grey')
ax5.loglog(freq,(VK[1,:].flatten()*freq)**(5/3),'-.k')
ax5.loglog(freq,(VKRBF[1,:].flatten()*freq)**(5/3),'--k')
ax5.set_ylim([1e-6, 1e-1])
ax5.set_yticklabels([])
ax5.spines['left'].set_color('blue')
ax5.spines['bottom'].set_color('blue')
ax5.spines['right'].set_color('blue')
ax5.spines['top'].set_color('blue')
plt.grid()

fig.text(0.35, 0.017, '$freq$', ha='center')
fig.text(0.01, 0.82, '$PSD_u$', va='center', rotation='vertical')
fig.text(0.01, 0.35, '$PSD_v$', va='center', rotation='vertical')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.13, top=0.98, wspace=0.3, hspace=0.25)
# plt.savefig('PSDTBL.png', dpi=300)