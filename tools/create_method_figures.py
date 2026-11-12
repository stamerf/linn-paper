"""Reproduce manuscript method diagrams as 300 dpi PNGs without experiment data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
INK = '#233444'
BLUE = '#e7f0f7'
GREEN = '#e7f2ed'
ORANGE = '#fff0dc'

def box(ax, xy, size, title, body, color):
    x, y = xy
    w, h = size
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.015',
                              linewidth=1.2, edgecolor=INK, facecolor=color))
    ax.text(x+w/2,y+h-.22,title,ha='center',va='top',weight='bold',color=INK,fontsize=12)
    ax.text(x+w/2,y+h/2-.16,body,ha='center',va='center',color=INK,linespacing=1.5)

def arrow(ax, start, end):
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','lw':1.5,'color':INK})

def architecture():
    fig, ax = plt.subplots(figsize=(12,6.7))
    ax.set(xlim=(0,12),ylim=(0,6.7))
    ax.axis('off')
    box(ax,(.2,4.25),(3.0,1.9),'Data and preprocessing',
        'Chronological partitions\nTrain-fitted encoding\nTarget scales and masks',BLUE)
    box(ax,(4.0,4.25),(3.3,1.9),'Generic neural predictor',
        'Tanh hidden layers\nUnrestricted linear outputs\nRescale predictions to hours',BLUE)
    box(ax,(8.15,4.25),(3.6,1.9),'Workstation constraints',
        'Elapsed ≥ 0 and booked ≥ 0\nBooked ≤ elapsed + tolerance\nPredictions and context only',GREEN)
    arrow(ax,(3.2,5.2),(4,5.2))
    arrow(ax,(7.3,5.2),(8.15,5.2))
    box(ax,(1.0,1.55),(3.9,1.7),'Observed-target loss',
        'Normalized squared error\nOnly retained target entries',BLUE)
    box(ax,(7.05,1.55),(3.9,1.7),'Domain loss',
        'Squared inequality violations\nAll training records',GREEN)
    arrow(ax,(2,4.25),(2,3.25))
    arrow(ax,(4.7,4.25),(4,3.25))
    arrow(ax,(9.95,4.25),(9.95,3.25))
    ax.text(6, .7, r'Training objective:  $L = L_{obs} + \lambda L_{domain}$',
            ha='center',va='center',fontsize=14,color=INK,weight='bold')
    arrow(ax,(3,1.55),(4.8,1.02))
    arrow(ax,(9,1.55),(7.2,1.02))
    ax.text(6,.13,'Adam updates predictor parameters. Validation selects the checkpoint.',
            ha='center',va='center',color=INK,fontsize=11)
    fig.savefig(OUT/'method_architecture.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def observation_protocol():
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set(xlim=(0,12),ylim=(0,6))
    ax.axis('off')
    panels=[(.25,'Training','Fit predictor and target scales',[(1,0),(0,1),(0,0),(1,0),(0,1)]),
            (4.3,'Validation','Select checkpoint with retained labels',[(0,1),(0,0),(1,0),(0,1),(1,0)]),
            (8.35,'Test','Evaluate against complete reference',[(1,1)]*5)]
    for x,title,subtitle,mask in panels:
        ax.text(x+1.65,5.6,title,ha='center',weight='bold',fontsize=14,color=INK)
        for j,head in enumerate(['Inputs','Elapsed','Booked']):
            ax.text(x+.53+j*1.08,5.0,head,ha='center',fontsize=11,color=INK)
        for i,pair in enumerate(mask):
            for j,visible in enumerate([1,*pair]):
                y=4.2-i*.56
                fill=BLUE if j==0 else (GREEN if visible else '#f2f2f2')
                ax.add_patch(Rectangle((x+j*1.08,y),1.02,.46,facecolor=fill,
                                      edgecolor='#89949d',lw=.7,hatch=None if visible else '///'))
                ax.text(x+j*1.08+.51,y+.23,'Observed' if visible else 'Hidden',
                        ha='center',va='center',fontsize=9,color=INK)
        ax.text(x+1.65,1.6,subtitle,ha='center',va='center',fontsize=10,color=INK)
    ax.plot([4,4],[1.25,5.85],color='#c6cdd2',lw=1)
    ax.plot([8.05,8.05],[1.25,5.85],color='#c6cdd2',lw=1)
    ax.text(6,.9,'Label masks act independently on the two targets. All input rows remain available.',
            ha='center',fontsize=12,color=INK)
    ax.text(6,.42,'Schematic of sparse-label confirmation. Test targets do not enter fitting or checkpoint selection.',
            ha='center',fontsize=10,color=INK)
    fig.savefig(OUT/'observation_protocol.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    architecture()
    observation_protocol()
    print(f'Created method_architecture.png and observation_protocol.png in {OUT}')
