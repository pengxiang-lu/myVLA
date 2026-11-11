from pathlib import Path
import json,pickle,zipfile,io,math,statistics,collections,hashlib
ROOT=Path('/home/lpx/python-project/myVLA')
VERS=['v1','v1_1','v1_2','v2','v2_1','v2_2']; STEPS=[2000,4000,6000,8000,10000]
class Dummy:
 def __new__(cls,*a,**k):return object.__new__(cls)
 def __init__(self,*a,**k):pass
 def __setstate__(self,s):pass
class Tensor:
 def __init__(self,shape):self.shape=shape
 def __setstate__(self,s):pass
 def numel(self):return math.prod(self.shape)
def rebuild(storage,offset,size,stride,*args):return Tensor(size)
class SafeMeta(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='collections' and name=='OrderedDict':return collections.OrderedDict
  if module=='torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:return rebuild
  return Dummy
 def persistent_load(self,pid):return Dummy()
def payload(path):
 with zipfile.ZipFile(path) as z:
  data=z.read(next(n for n in z.namelist() if n.endswith('/data.pkl')))
 return SafeMeta(io.BytesIO(data)).load()
def wilson(k,n):
 p=k/n;z=1.95996398454;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [100*(c-h),100*(c+h)]
report={'runs':[],'train':{},'checkpoints':{}}
for ver in VERS:
 d=ROOT/'outputs/train'/('libero_spatial_'+ver)
 cp=payload(d/'checkpoint_10000.pt')
 head=sum(t.numel() for t in cp['head'].values())
 vision=sum(t.numel() for t in cp.get('vision',{}).values())
 config=cp['config']; manifest=cp['manifest']
 f=next((d/f for f in ['metrics.jsonl','log.jsonl'] if (d/f).exists()),None)
 rows=[json.loads(s) for s in f.read_text().splitlines() if s.strip()] if f else []
 # Keep last log entry per step; logs may include resumed sessions.
 logs={r['step']:r for r in rows}; ordered=[logs[s] for s in sorted(logs)]
 resets=[(a['step'],b['step']) for a,b in zip(rows,rows[1:]) if b.get('elapsed_seconds',0)<a.get('elapsed_seconds',0)]
 train={'config':config,'head_parameters':head,'vision_parameters':vision,'total_parameters':head+149620737,'trainable_parameters':head+vision,'episodes':len(manifest),'frames':sum(m['length'] for m in manifest),'splits':dict(collections.Counter(m['split'] for m in manifest)),'manifest_identity':sorted((Path(m['path']).name,m['demo'],m['length'],m['split']) for m in manifest),'log_records':len(rows),'unique_log_steps':len(logs),'elapsed_resets':resets,'final_loss':ordered[-1]['loss'],'last20_mean_loss':statistics.mean(r['loss'] for r in ordered[-20:]),'last_elapsed_seconds':ordered[-1].get('elapsed_seconds'),'peak_vram_gib':max(r.get('peak_vram_gib',0) for r in rows),'logged_skips':sum(r.get('optimizer_skipped',False) for r in rows),'log_path':str(f.relative_to(ROOT)),'loss_curve':[[r['step'],r['loss']] for r in ordered]}
 report['train'][ver]=train
 for step in STEPS:
  d=ROOT/'outputs/eval'/('libero_spatial_'+ver)/f'ckpt{step}/seed42'
  progress=json.loads((d/'progress.json').read_text()); summary=json.loads((d/'log.json').read_text())
  records=[json.loads(s) for s in (d/'episodes.jsonl').read_text().splitlines() if s.strip()]
  metas=[r for r in records if r.get('type')=='evaluation_metadata']; eps=[r for r in records if 'success' in r and 'task' in r]
  counts=[v for task in progress.values() for v in task.values()]
  assert len(counts)==100 and all(isinstance(v,bool) for v in counts)
  assert all(len(task)==10 for task in progress.values())
  k=sum(counts)
  assert abs(summary['total_success_rate']-k/100)<1e-9
  for task,values in progress.items():assert abs(summary['success_rate'][task]-sum(values.values())/10)<1e-9
  assert len(eps)==100
  assert len({(e['task'],e['episode']) for e in eps})==100
  assert all(progress[e['task']][str(e['episode'])]==e['success'] for e in eps)
  cp2=payload(ROOT/'outputs/train'/('libero_spatial_'+ver)/f'checkpoint_{step}.pt')
  run={'version':ver,'step':step,'successes':k,'n':len(counts),'rate':k,'ci95':wilson(k,100),'per_task':summary['success_rate'],'progress':progress,'episodes':eps,'metadata':metas[0] if metas else None,'checkpoint_config':cp2['config'],'checkpoint_step':cp2['step'],'source':str(d.relative_to(ROOT)),'mean_success_steps':statistics.mean(e['steps'] for e in eps if e['success'])}
  assert cp2['step']==step
  report['runs'].append(run)
# Verify identical paired evaluation initial states across all runs.
base={(e['task'],e['episode']):e.get('initial_state_index') for e in report['runs'][0]['episodes']}
report['paired_initial_indices_all_equal']=all({(e['task'],e['episode']):e.get('initial_state_index') for e in r['episodes']}==base for r in report['runs'])
report['all_manifests_same']=all(t['manifest_identity']==report['train']['v1']['manifest_identity'] for t in report['train'].values())
# Paired descriptive contrasts and exact McNemar (two-sided), no multiplicity adjustment.
report['contrasts']=[]
for a,b in [('v1','v2'),('v1_1','v2_1'),('v1_2','v2_2'),('v2','v2_2'),('v2_1','v2_2'),('v1','v1_2')]:
 ra=next(r for r in report['runs'] if r['version']==a and r['step']==10000);rb=next(r for r in report['runs'] if r['version']==b and r['step']==10000)
 ea={(e['task'],e['episode']):e['success'] for e in ra['episodes']};eb={(e['task'],e['episode']):e['success'] for e in rb['episodes']}
 wins=sum(eb[k] and not ea[k] for k in ea);losses=sum(ea[k] and not eb[k] for k in ea);n=wins+losses
 p=min(1,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1
 report['contrasts'].append({'a':a,'b':b,'gain_pp':rb['rate']-ra['rate'],'wins':wins,'losses':losses,'mcnemar_exact_p':p})
for p in (ROOT/'outputs/downloads/google_drive').glob('checkpoint_*.pt'):
 d=payload(p);report['checkpoints'][str(p.relative_to(ROOT))]={'step':d['step'],'model':d['config']['model'],'train':d['config']['train']}
path=ROOT/'tmp/pdfs/experiment_audit.json';path.write_text(json.dumps(report,indent=2))
for ver in VERS:
 t=report['train'][ver];print(ver,'rates',[r['rate'] for r in report['runs'] if r['version']==ver],'head',t['head_parameters'],'trainable',t['trainable_parameters'],'batch',t['config']['train']['batch_size'],'accum',t['config']['train']['grad_accumulation'],'frames',t['frames'],'splits',t['splits'],'loss',round(t['last20_mean_loss'],4),'VRAM',round(t['peak_vram_gib'],2),'elapsed',t['last_elapsed_seconds'],'resets',t['elapsed_resets'])
print('PAIRED',report['paired_initial_indices_all_equal'],'MANIFESTS',report['all_manifests_same'])
print('CONTRASTS',report['contrasts'])
print('metadata coverage',sum(r['metadata'] is not None for r in report['runs']),'/30')
print('config changes',[(r['version'],r['step'],r['checkpoint_config']['train']['batch_size'],r['checkpoint_config']['train']['grad_accumulation'],r['checkpoint_config']['train']['output_dir']) for r in report['runs'] if r['version']=='v2_1'])
