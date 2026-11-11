from pathlib import Path
import os,json,subprocess,random,time,datetime,platform
ROOT=Path('/home/lpx/python-project/myVLA');OUT=ROOT/'outputs/benchmarks/inference_20261111';OUT.mkdir(parents=True,exist_ok=True)
versions=['v1','v1_1','v1_2','v2','v2_1','v2_2'];rng=random.Random(42)
language='pick up the black bowl between the plate and the ramekin and place it on the plate'
env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
info={'date_local':'2026-11-11','timezone':'Asia/Shanghai','protocol':'sequential randomized model order in each of three rounds','precision':'fp32','weights':'ema','warmup':50,'runs_per_round':200,'rounds':3,'language':language,'schedule':[],'observations':[],'cpu':platform.processor(),'OMP_NUM_THREADS':env.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':env.get('MKL_NUM_THREADS')}
for rep in range(1,4):
 order=versions.copy();rng.shuffle(order);info['schedule'].append(order)
 for version in order:
  result=OUT/f'{version}_fp32_round{rep}.json';log=result.with_suffix('.log')
  status=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
  print(f'ROUND {rep} MODEL {version} | GPU before: {status}',flush=True)
  cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/measure_inference_time.py'),'--checkpoint',str(ROOT/f'outputs/train/libero_spatial_{version}/checkpoint_10000.pt'),'--device','cuda','--precision','fp32','--weights','ema','--warmup','50','--runs','200','--language',language,'--output',str(result)]
  start=time.monotonic()
  with log.open('w') as f:completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  if completed.returncode:raise RuntimeError(f'Benchmark failed {version} round {rep}: see {log}')
  d=json.loads(result.read_text());assert d['output_shape']==[1,12,7] and d['weights']=='ema' and d['runs']==200
  info['observations'].append({'version':version,'round':rep,'gpu_before':status,'wall_seconds':time.monotonic()-start,'result_file':result.name,'latency':d['latency_ms']})
  (OUT/'measurement_manifest.json').write_text(json.dumps(info,indent=2))
  print(f"DONE {version} round {rep}: median={d['latency_ms']['median']:.3f} ms p95={d['latency_ms']['p95']:.3f} ms",flush=True)
print('ALL 18 BENCHMARKS COMPLETED',flush=True)
