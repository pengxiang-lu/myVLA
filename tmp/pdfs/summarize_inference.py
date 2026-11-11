from pathlib import Path
import json,csv,statistics,hashlib
import numpy as np
ROOT=Path('/home/lpx/python-project/myVLA');D=ROOT/'outputs/benchmarks/inference_20261111'
summary={'date':'2026-11-11','groups':{},'hardware':'NVIDIA GeForce RTX 3060 Laptop GPU','precision':'fp32','scope':'resident GPU inputs -> vision + uncached text encoder + action head + device-side decoded 12-step chunk','excluded':'loading, preprocessing, tokenization, input transfer, CPU output copy, simulator'}
for v in ['v1','v1_1','v1_2','v2','v2_1','v2_2']:
 rows=[json.loads((D/f'{v}_fp32_round{r}.json').read_text()) for r in [1,2,3]]
 samples=np.asarray([x for d in rows for x in d['samples_ms']])
 assert len(samples)==600 and np.isfinite(samples).all() and (samples>0).all()
 assert all(d['weights']=='ema' and d['precision']=='fp32' and d['output_shape']==[1,12,7] and d['text_tokens']==21 and not d['text_cache'] and d['checkpoint_step']==10000 for d in rows)
 med=float(np.median(samples));rmed=[d['latency_ms']['median'] for d in rows]
 summary['groups'][v]={'median_ms':med,'mean_ms':float(np.mean(samples)),'p95_ms':float(np.percentile(samples,95)),'min_ms':float(np.min(samples)),'max_ms':float(np.max(samples)),'round_medians_ms':rmed,'round_median_range_ms':[min(rmed),max(rmed)],'chunk_predictions_per_second':1000/med,'amortized_ms_per_action':med/12,'samples':len(samples),'fusion_layers':rows[0]['fusion_layers'],'train_vision_encoder':rows[0]['train_vision_encoder']}
summary['script_sha256']=hashlib.sha256((ROOT/'scripts/measure_inference_time.py').read_bytes()).hexdigest()
(D/'summary.json').write_text(json.dumps(summary,indent=2))
with (D/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['model','median_ms','mean_ms','p95_ms','round_median_min_ms','round_median_max_ms','chunk_predictions_per_second','amortized_ms_per_action','samples'])
 for v,g in summary['groups'].items():w.writerow([v,g['median_ms'],g['mean_ms'],g['p95_ms'],*g['round_median_range_ms'],g['chunk_predictions_per_second'],g['amortized_ms_per_action'],g['samples']])
print(json.dumps(summary,indent=2))
