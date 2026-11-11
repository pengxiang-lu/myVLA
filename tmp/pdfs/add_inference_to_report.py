from pathlib import Path
p=Path('/home/lpx/python-project/myVLA/tmp/pdfs/build_report.py');s=p.read_text()
s=s.replace("OUT=ROOT/'output/pdf/myVLA_experimental_report.pdf'","OUT=ROOT/'myVLA_experimental_report.pdf'")
s=s.replace("page();h('9. Conclusions", "page();h('10. Conclusions").replace("h2('9.1.","h2('10.1.").replace("h2('9.2.","h2('10.2.")
s=s.replace("page();h('8. Interpretation", "page();h('9. Interpretation")
for n in range(1,6):s=s.replace(f"h2('8.{n}.",f"h2('9.{n}.")
s=s.replace('Table 8. Paths are relative','Table 9. Paths are relative')
s=s.replace('and training-resource observations.','training-resource observations, and a new GPU inference benchmark with 3,600 timed predictions.')
s=s.replace('The report is an analysis of existing artifacts, not a new training or simulation run.', 'Training and closed-loop results are reanalyses of existing artifacts. The inference measurements are new GPU microbenchmarks; no new training or simulation run was performed.')
s=s.replace("['S5','configs/train_cfg_v4.yaml;", "['S6','scripts/measure_inference_time.py;<br/>outputs/benchmarks/inference_20261111/','18 timing JSON files (600 samples per model); measurement manifest; summary.json and summary.csv'],\n ['S5','configs/train_cfg_v4.yaml;")
s=s.replace("f'{doc.page} / 9'", "f'{doc.page} / 10'")
block='''# New measured inference section
page();h('8. Measured GPU inference latency')
b=json.loads((ROOT/'outputs/benchmarks/inference_20261111/summary.json').read_text())
h2('8.1. Measurement protocol and timing boundary')
p('The existing scripts/measure_inference_time.py was executed on the current RTX 3060 Laptop GPU (6 GiB), with PyTorch 2.6.0+cu124, CUDA build 12.4, and driver 595.99.02. All six models use their 10k EMA checkpoint, FP32 parameters and computation, batch size one, two 224 x 224 RGB inputs, an eight-dimensional state, and the same 21-token instruction. Three sequential measurement rounds randomize model order (schedule seed 42). Each process performs 50 warmup predictions followed by 200 timed predictions, giving 600 samples per model and 3,600 in total. The pilot run is excluded. [S6]')
p('Timing uses a CPU wall clock with CUDA synchronization immediately before and after inference. It includes visual encoding, a full text-encoder pass, the action head, and device-side decoding of the 12 x 7 action chunk. Inputs are fixed synthetic normalized tensors already on the GPU; tokenization is completed before timing. Model loading, preprocessing, input transfers, output copies to CPU, and environment execution are excluded. Inference is eager, without compilation or CUDA Graphs. The script deliberately bypasses the normal rollout text cache, so this is a resident-input uncached-text microbenchmark, not end-to-end rollout latency.')
rows=[['Model','Median (ms)','Mean (ms)','P95 (ms)','Round medians (ms)','Chunks/s']]
for v in vs:
 g=b['groups'][v]
 rows.append([names[v],f"{g['median_ms']:.2f}",f"{g['mean_ms']:.2f}",f"{g['p95_ms']:.2f}",f"{min(g['round_medians_ms']):.2f}-{max(g['round_medians_ms']):.2f}",f"{g['chunk_predictions_per_second']:.1f}"])
table(rows,[.7,1,1,1,1.6,.8])
cap('Table 8. New inference measurements on 11 November 2026. Median, mean, and P95 pool 600 synchronized timings per model; the round-median column is the minimum-to-maximum range across three independently loaded processes. Chunks/s is 1000 divided by pooled median latency. All measured outputs are finite and have shape [1, 12, 7]. [S6]')
h2('8.2. Latency, fusion depth, and closed-loop performance')
g=b['groups']['v2_2'];base=b['groups']['v2']
p(f"V2.2 requires {g['median_ms']:.2f} ms per complete prediction at the median (P95 {g['p95_ms']:.2f} ms), corresponding to {g['chunk_predictions_per_second']:.1f} chunk predictions/s. Relative to the no-fusion V2 model, this is an observed median increase of {g['median_ms']-base['median_ms']:.2f} ms ({100*(g['median_ms']/base['median_ms']-1):.1f}%), alongside the recorded success difference of 96% versus 92%. These are separate latency and behavior measurements; they do not establish a general speed-accuracy frontier.")
p('At a fixed fusion depth, freezing the visual encoder during training does not remove its forward computation at inference. Small timing differences between V1/V2, V1.1/V2.1, or V1.2/V2.2 should therefore be interpreted in light of runtime variation rather than as a speed advantage caused by freezing. Laptop temperature, GPU clock/power management, and background display activity were not locked; repeated synchronized timings do not eliminate these effects.')
h2('8.3. Interpretation for action execution')
p(f"Amortizing V2.2's median over its 12 predicted actions gives {g['amortized_ms_per_action']:.2f} ms per generated action. This is accounting only: a fresh observation still waits for a full {g['median_ms']:.2f} ms chunk prediction, and the action queue then executes the selected horizon. It is neither single-step reaction latency nor an achieved simulator control rate. Real deployment timing requires the cached-text rollout path, preprocessing, transfers, CPU action retrieval, and environment timing to be measured together.", 'small')

'''
s=s.replace("# 8\npage();h('9. Interpretation",block+"# 9\npage();h('9. Interpretation")
p.write_text(s)
print('Report builder updated: new section 8, 10 pages, timing provenance S6')
