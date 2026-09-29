"""Generate data-driven diagnostic charts from local CSV results, without TeX."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures'

def scatter(ax,x,y,xlabel,ylabel):
    high=max(x.max(),y.max())*1.05;ax.plot([0,high],[0,high],'--',color='0.6',lw=1);ax.scatter(x,y,s=13,alpha=.65)
    ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(0,high),ylim=(0,high));ax.set_title(f'n = {len(x)}; Spearman rho = {spearmanr(x,y).statistic:.3f}')
def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=200)
    plt.close(fig)
def main():
    OUT.mkdir(parents=True,exist_ok=True);core=ROOT/'core/results/development';cross=ROOT/'cross_specimen/results'
    u=pd.read_csv(core/'universe_decomposition.csv');fig,ax=plt.subplots(figsize=(9,5));ax.barh(u.universe,u.C,label='Within-body C');ax.barh(u.universe,u.R,left=u.C,label='Between-body R');ax.set_xlabel('Bits');ax.legend();save(fig,'input_universes')
    if (core/'sampling_sensitivity.csv').exists():
        s=pd.read_csv(core/'sampling_sensitivity.csv');fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,metric in zip(axes,['A','C','R']):
            for method,q in s.groupby('method'):
                v=q.groupby('fraction')[metric].agg(['mean','min','max']);ax.plot(v.index,v['mean'],marker='o',label=method);ax.fill_between(v.index,v['min'],v['max'],alpha=.15)
            ax.set(xscale='log',xlabel='Sampling fraction',ylabel=metric+' (bits)')
        axes[0].legend(fontsize=7);save(fig,'sampling_designs')
    if (cross/'male_female_primary_paired_types.csv').exists():
        p=pd.read_csv(cross/'male_female_primary_paired_types.csv');fig,axes=plt.subplots(1,2,figsize=(8,4));scatter(axes[0],p.R_a,p.R_b,'Male R (bits)','Female R (bits)');scatter(axes[1],p.R_fraction_a,p.R_fraction_b,'Male R/A','Female R/A');save(fig,'matched_type_agreement')
    if (cross/'spatial_global.csv').exists():
        s=pd.read_csv(cross/'spatial_global.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
        for dataset,q in s[s['shift']==0].groupby('dataset'):q=q.sort_values('width');axes[0].plot(q.width,q.spatial_share_R,marker='o',label=dataset)
        axes[0].set(xlabel='Bin width in coordinate steps',ylabel='Spatial share of R (%)');axes[0].legend(fontsize=7)
        p=pd.read_csv(cross/'spatial_prediction_types.csv')
        for i,(dataset,q) in enumerate(p.groupby('dataset')):axes[1].scatter([i]*len(q),q.gain_bits,label=dataset,s=15)
        axes[1].axhline(0,color='0.6',ls=':');axes[1].set(ylabel='Held-out gain (bits/contact)',xticks=[]);axes[1].legend(fontsize=7);save(fig,'spatial_diagnostics')
    print('Saved local diagnostic charts to',OUT)
if __name__=='__main__':main()
