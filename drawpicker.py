"""
Module that stores all the 'under the hood'-functions used to make the plots
for the visualize_pick_info() method. Each function plots a specific set of
data onto on matplotlib.axes instance and returns the corresponding figure
and ax if passed to the function correctly

"""
import numpy as np

boxplotstyle = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

def draw_trace(self,fig,ax,trace_figtitle,trace_channelname):
    ax.axhline(0,color='grey',linewidth=0.5)
    ax.plot(self.time_axis,self.trace,color='black',linewidth=0.5,label='trace')
    if hasattr(self, 'pick_Allen'):
        t_Allen = self.time_axis[self.pick_Allen] if (self.pick_Allen is not None) else -1
        ax.axvline(t_Allen,color='red',label='Pick Allen',linewidth=2)
    if hasattr(self, 'pick_BK'):
        t_BK = self.time_axis[self.pick_BK] if (self.pick_BK is not None) else -1
        ax.axvline(t_BK,color='darkorange',label='Pick BK',linestyle='dashdot',linewidth=2)
    ax.set_title(trace_figtitle)
    ax.text(0.02, 0.92, trace_channelname, transform=ax.transAxes,va='top',ha='left', bbox=boxplotstyle)
    ax.legend()
    return fig,ax

def draw_AllenCF(self,fig,ax,CF,E,alpha,beta,gamma):
    axtw = ax.twinx()
    axtw.plot(self.time_axis,E,label='Envl.',color='black',linewidth=0.1)
    ax.plot(self.time_axis,CF,label='CF',color='tab:blue')
    ax.axhline(gamma,label=r'$\gamma$='+f"{gamma:.1f}",color='tab:green')
    ax.set_ylim(bottom=0)
    axtw.set_ylim(bottom=0)
    ax.text(0.02, 0.92,r'$\alpha$='+f"{alpha:.2f}s"+"\n"+r'$\beta$='+f"{beta:.2f}s", transform=ax.transAxes,va='top',ha='left', bbox=boxplotstyle)
            
    ax.patch.set_visible(False)
    axtw.patch.set_visible(True)
    ax.set_zorder(axtw.get_zorder() + 1)
    lin, lab = ax.get_legend_handles_labels()
    lin2, lab2 = axtw.get_legend_handles_labels()
    ax.legend(lin2+lin,lab2+lab)
    return fig,ax

def draw_AllenSTA(self,fig,ax,sta,lta,delta):
    ax.plot(self.time_axis,lta,label='LTA',color='tab:brown')
    ax.plot(self.time_axis,sta,label='STA',color='tab:orange')
    ax.plot(self.time_axis,delta,label=r'$\delta$',color='tab:green',linewidth=0.5)
    ax.set_yscale('log')
    ax.legend()
    return fig,ax

def draw_AllenM(self,fig,ax,M,L,s,tMin,MMin):
    ax.plot(self.time_axis,M,color='black',label='M')
    ax.plot(self.time_axis,s,color='tab:blue',label='s')
    ax.plot(self.time_axis,L,color='tab:red',label='L')
    ax.text(0.02, 0.92,r'$t_{Min}$='+f"{tMin:.2f}s"+"\n"+r'$M_{Min}$='+f"{MMin:d}", transform=ax.transAxes,va='top',ha='left', bbox=boxplotstyle)
    ax.legend()
    return fig,ax

def draw_BKCF(self,fig,ax,CF,TUp,TDown,S1):
    ax.semilogy(self.time_axis,CF,label='CF',color='tab:blue',linewidth=0.5)
    ax.axhline(S1,label=r'$S_1$='+f"{S1:.1f}",color='tab:green')
    ax.axhline(2*S1,label=r'2$S_1$',color='tab:green',linewidth=0.5,linestyle='dashed')
    ax.text(0.02, 0.92,r'$T_{Up}$='+f"{TUp:.2f}s"+"\n"+r'$T_{Down}$='+f"{TDown:.2f}s", transform=ax.transAxes,va='top',ha='left', bbox=boxplotstyle)
    ax.legend()
    return fig,ax

def draw_BKE(self,fig,ax,E,cumultative_mean,dynamic_variance):
    show_indices = np.nonzero(dynamic_variance)
    axtw = ax.twinx()
    la = ax.semilogy(self.time_axis[show_indices],np.abs(E**2-cumultative_mean)[show_indices],label=r'$|e^4-\mu(t)|$',color='tab:orange',linewidth=0.5)
    ld = ax.semilogy(self.time_axis[show_indices],dynamic_variance[show_indices],label=r'$\sigma^2[e^2]_{dyn}$',color='tab:brown')
    le = axtw.plot(self.time_axis,E**2,label=r'$e^4$',color='black',linewidth=0.5)
    axtw.set_ylim(bottom=0)
    lines = le[0],la[0],ld[0]
    labels = [l.get_label() for l in lines]
    ax.legend(lines,labels)
    return fig,ax
    