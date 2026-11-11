from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parent
a=json.loads((D/'experiment_audit.json').read_text())
vs=list(a['train']);names={v:v.upper().replace('_','.') for v in vs}
colors=['#56677a','#d49439','#337e79','#56677a','#d49439','#337e79']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':160,'savefig.dpi':240,'axes.linewidth':0.6})
fig,axs=plt.subplots(1,2,figsize=(7,2.7),sharex=True,sharey=True)
for ax,group,title in zip(axs,[vs[:3],vs[3:]],['Frozen visual encoder','Trainable visual encoder']):
 for v in group:
  rows=[r for r in a['runs'] if r['version']==v];idx=vs.index(v)
  ax.plot([r['step']/1000 for r in rows],[r['rate'] for r in rows],marker='o',ms=4,lw=1.6,label=names[v],color=colors[idx])
 ax.set_title(title,fontsize=10);ax.set_xticks([2,4,6,8,10]);ax.set_ylim(0,105);ax.set_xlabel('Training step (thousands)');ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right',frameon=False)
axs[0].set_ylabel('Success rate (%)')
fig.tight_layout(pad=.8);fig.savefig(D/'success_curves.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,2.25))
final=[next(r for r in a['runs'] if r['version']==v and r['step']==10000) for v in vs]
y=[r['rate'] for r in final];err=[[r['rate']-r['ci95'][0] for r in final],[r['ci95'][1]-r['rate'] for r in final]]
bars=ax.bar(range(6),y,color=['#8a99a8']*3+['#4f8684']*3,width=.6)
ax.errorbar(range(6),y,yerr=err,fmt='none',ecolor='#263442',elinewidth=1,capsize=3)
for i,r in enumerate(final):ax.text(i,5,f"{r['rate']}/100",ha='center',fontsize=8,color='white')
ax.set_ylim(0,105);ax.set_xticks(range(6),[names[v] for v in vs]);ax.set_ylabel('Success rate (%)');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.tight_layout(pad=.7);fig.savefig(D/'final_intervals.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,2.65))
tasks=list(final[0]['per_task']);matrix=np.array([[r['per_task'][t]*100 for r in final] for t in tasks])
im=ax.imshow(matrix,aspect='auto',vmin=40,vmax=100,cmap='YlGnBu')
for i in range(10):
 for j in range(6):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>=80 else '#17232c')
ax.set_xticks(range(6),[names[v] for v in vs]);ax.set_yticks(range(10),[f'T{i}' for i in range(10)]);ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
fig.colorbar(im,ax=ax,fraction=.03,pad=.025,label='Success (%)')
fig.tight_layout(pad=.7);fig.savefig(D/'task_heatmap.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,2.65))
for i,v in enumerate(vs):
 arr=np.asarray(a['train'][v]['loss_curve']);smooth=np.convolve(arr[:,1],np.ones(20)/20,mode='valid')
 ax.plot(arr[19:,0]/1000,smooth,color=colors[i],ls='--' if i<3 else '-',lw=1.5,label=names[v])
ax.set_yscale('log');ax.set_xlabel('Training step (thousands)');ax.set_ylabel('Training loss (log scale)');ax.grid(alpha=.15);ax.legend(ncol=3,frameon=False,loc='upper right');ax.set_xticks([0,2,4,6,8,10])
fig.tight_layout(pad=.8);fig.savefig(D/'training_loss.png');plt.close(fig)
with (D/'results_summary.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['version','training_step','successes','episodes','success_percent','wilson95_lower','wilson95_upper','source'])
 for r in a['runs']:writer.writerow([r['version'],r['step'],r['successes'],r['n'],r['rate'],*r['ci95'],r['source']])
print('Generated four figures and summary CSV')
