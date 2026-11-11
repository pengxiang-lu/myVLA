from pathlib import Path
import json,math,statistics
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Flowable,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER,TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path('/home/lpx/python-project/myVLA'); D=ROOT/'tmp/pdfs'; OUT=ROOT/'myVLA_experimental_report.pdf'
a=json.loads((D/'experiment_audit.json').read_text());vs=list(a['train']);names={v:v.upper().replace('_','.') for v in vs}
for name,file in [('Serif','LiberationSerif-Regular.ttf'),('SerifBold','LiberationSerif-Bold.ttf'),('SerifItalic','LiberationSerif-Italic.ttf'),('Sans','LiberationSans-Regular.ttf'),('SansBold','LiberationSans-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/liberation/'+file))
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifBold',italic='SerifItalic',boldItalic='SerifBold')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansBold',italic='Sans',boldItalic='SansBold')
INK=colors.HexColor('#17232e');TEAL=colors.HexColor('#326e70');GRAY=colors.HexColor('#5a6570');LIGHT=colors.HexColor('#eef3f4')
W,H=A4; M=47; CW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='Serif',fontSize=10.3,leading=13.7,spaceAfter=6.5,alignment=TA_JUSTIFY,textColor=INK),
 'small':ParagraphStyle('small',fontName='Serif',fontSize=9.1,leading=11.7,spaceAfter=5,textColor=INK),
 'h1':ParagraphStyle('h1',fontName='SerifBold',fontSize=13.3,leading=17,spaceAfter=9,spaceBefore=3,textColor=INK,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='SerifBold',fontSize=10.9,leading=14,spaceAfter=5,spaceBefore=6,textColor=INK,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Serif',fontSize=9,leading=11.4,spaceAfter=9,textColor=GRAY),
 'cell':ParagraphStyle('cell',fontName='Sans',fontSize=8.3,leading=10.9,textColor=INK),
 'cellsmall':ParagraphStyle('cellsmall',fontName='Sans',fontSize=7.7,leading=10.2,textColor=INK),
 'headcell':ParagraphStyle('headcell',fontName='SansBold',fontSize=8.2,leading=10.6,textColor=INK),
 'title':ParagraphStyle('title',fontName='SerifBold',fontSize=21,leading=24.5,alignment=TA_CENTER,spaceAfter=10,textColor=INK),
 'subtitle':ParagraphStyle('subtitle',fontName='Sans',fontSize=9,leading=12,alignment=TA_CENTER,spaceAfter=17,textColor=GRAY),
 'abstract':ParagraphStyle('abstract',fontName='Serif',fontSize=9.5,leading=12.5,spaceAfter=9,alignment=TA_JUSTIFY,leftIndent=15,rightIndent=15,textColor=INK),
 'equation':ParagraphStyle('equation',fontName='SerifItalic',fontSize=10.8,leading=15,alignment=TA_CENTER,spaceAfter=9,spaceBefore=5,textColor=INK),
}
story=[]
def p(text,sty='body'):story.append(Paragraph(text,styles[sty]))
def h(text):p(text,'h1')
def h2(text):p(text,'h2')
def cap(text):p(text,'caption')
def page():story.append(PageBreak())
def table(rows,widths,small=False):
 data=[[Paragraph(str(c),styles['headcell'] if i==0 else styles['cellsmall' if small else 'cell']) for c in row] for i,row in enumerate(rows)]
 t=Table(data,colWidths=[CW*w/sum(widths) for w in widths],hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEABOVE',(0,0),(-1,0),.8,INK),('LINEBELOW',(0,0),(-1,0),.5,INK),('LINEBELOW',(0,-1),(-1,-1),.7,INK)]))
 story.append(t);story.append(Spacer(1,5))
def fig(file,height):story.append(Image(str(D/file),width=CW,height=height))
class Pipeline(Flowable):
 def __init__(self):Flowable.__init__(self);self.width=CW;self.height=66
 def draw(self):
  c=self.canv;labels=[['Two RGB views','+ instruction'],['CLIP encoders','+ state MLP'],['0 / 2 / 6 fusion','blocks'],['4-layer decoder','12 x 7 actions']];bw=(CW-45)/4
  for i,lines in enumerate(labels):
   x=i*(bw+15);c.setFillColor(LIGHT if i<3 else colors.HexColor('#dfeeed'));c.setStrokeColor(TEAL);c.setLineWidth(.6);c.roundRect(x,13,bw,42,3,fill=1)
   c.setFillColor(INK);c.setFont('Sans',8.3)
   for j,line in enumerate(lines):c.drawCentredString(x+bw/2,39-j*12,line)
   if i<3:
    x2=x+bw;c.setStrokeColor(GRAY);c.line(x2+2,34,x2+12,34);c.line(x2+9,37,x2+12,34);c.line(x2+9,31,x2+12,34)

def final(v):return next(r for r in a['runs'] if r['version']==v and r['step']==10000)
def rates(v):return [r['rate'] for r in a['runs'] if r['version']==v]

# 1
p('myVLA: An Experimental Study of Visual Adaptation and Multimodal Fusion','title')
p('Behavior cloning on LIBERO-Spatial | English experimental report<br/>Repository evidence audited on 11 November 2026','subtitle')
p('<b>Abstract.</b> This report consolidates six myVLA configurations evaluated at five training checkpoints on LIBERO-Spatial. The study varies visual-encoder fine-tuning and the depth of bidirectional vision-language fusion while keeping the text encoder frozen. The audited corpus comprises 500 demonstrations, 62,250 training frames, and 30 evaluations of 100 episodes each. At 10,000 training steps, success ranges from 79% to 96%. The six-block model with a trainable visual encoder (V2.2) achieves 96/100 successes, compared with 84/100 for the frozen baseline and 92/100 for visual fine-tuning without fusion. Fusion depth alone does not yield a monotonic improvement. We report checkpoint trajectories, paired comparisons, task-level outcomes, training-resource observations, and a new GPU inference benchmark with 3,600 timed predictions. Interpretation is limited by one recorded training seed, a small evaluation sample, incomplete historical hardware metadata, and a microbatch difference in V2.1.','abstract')
h('1. Experimental objectives and model variants')
p('The project implements a compact language-conditioned action-chunk policy using pretrained CLIP ViT-B/16 encoders [1] and evaluates manipulation behavior in LIBERO [2]. The present experiments are joint multi-task behavior cloning on the Spatial suite; they do not constitute a lifelong-learning or unseen-task evaluation. The analysis asks whether visual adaptation helps, whether deeper fusion improves control, and whether offline training loss predicts closed-loop success.')
story.append(Pipeline());cap('Figure 1. Shared policy structure. The state embedding joins the decoder memory after visual-language fusion. Text is frozen in every variant; visual fine-tuning and fusion depth define the experimental groups. [S3]')
rows=[['Model','Visual encoder','Fusion blocks','Total params.','Trainable params.']]
for v,t in a['train'].items():rows.append([names[v],'Trainable' if t['config']['model']['train_vision_encoder'] else 'Frozen',str(t['config']['model']['fusion_layers']),f"{t['total_parameters']/1e6:.2f} M",f"{t['trainable_parameters']/1e6:.2f} M"])
table(rows,[.7,1.3,.95,1.1,1.3])
cap('Table 1. Canonical variants, identified by checkpoint architecture rather than historical folder names. Totals include the original CLIP projection parameters, which are frozen and unused by the policy forward pass. [S2, S3]')
p('The current V4 configuration adds eight fusion blocks, but no completed V4 evaluation is present. MetaWorld MT50 support is implemented, but no MT50 training/evaluation results are available in the audited output directories. Both are excluded from quantitative comparisons.','small')

# 2
page();h('2. Dataset, implementation, and environment')
h2('2.1. Demonstrations and model inputs')
p('All six checkpoint manifests contain the same 500 training trajectories from 10 LIBERO-Spatial tasks, with 50 demonstrations per task and 62,250 frames in total. Every manifest entry belongs to the training split. No held-out demonstration validation set is used. Frames are sampled uniformly with replacement, so longer trajectories contribute more samples than shorter ones. A nominal 10,000-step run with effective batch 64 draws 640,000 frame samples, equivalent to about 10.28 dataset-sized sample counts, rather than 10.28 sequential epochs. [S2, S3]')
p('Each sample provides two RGB views (agent view and wrist view), a task instruction, and an eight-dimensional state consisting of end-effector position, axis-angle orientation, and two gripper positions. The inspected HDF5 observations are 128 x 128 RGB; preprocessing vertically flips LIBERO images, resizes them to 224 x 224 by bicubic interpolation, and applies CLIP channel normalization. State and continuous-action means and standard deviations are computed from training demonstrations. There is no configured image shift or color jitter. [S2-S4]')
h2('2.2. Recorded software and current host')
table([
 ['Component','Version or configuration','Evidence and scope'],
 ['Python / package','Python 3.10.20; myvla 0.1.0','Python: current environment; package: evaluation metadata'],
 ['Learning stack','PyTorch 2.6.0; Transformers 4.57.6; NumPy 1.26.4','Recorded in 24/30 evaluation metadata headers'],
 ['LIBERO simulation','robosuite 1.4.1; MuJoCo 2.3.7; relative OSC_POSE','Evaluation metadata; LIBERO Git revision is not recorded'],
 ['CLIP backbone','ViT-B/16; revision 57c21647...e4ae5f3','Same full pinned revision in all checkpoint configs'],
 ['Current CPU / RAM','Intel Core i9-12900H; about 15.35 GiB host RAM','Observed on report date; historical run host not certified'],
 ['Current GPU','NVIDIA GeForce RTX 3060 Laptop GPU; 6,144 MiB','Current host only; historical GPU model is not logged'],
 ['Current OS / CUDA','Ubuntu 24.04.5 LTS; CUDA build 12.4; driver 595.99.02','Current installed PyTorch is 2.6.0+cu124'],
 ],[1.1,2.35,2.2],True)
cap('Table 2. Environment provenance. Installed versions and the current host are distinguished from historical experiment records; cloud-style V2.1 paths do not establish the cloud GPU model. [S1, S2, S4]')
h2('2.3. Shared representation and output head')
p('Each camera contributes 196 CLIP patch tokens; CLS tokens are dropped. Features are projected to width 256 and augmented with camera embeddings. Fusion variants use bidirectional cross-attention with zero-initialized residual gates and eight attention heads. The decoder consumes 392 visual tokens plus language and state summaries. Four Transformer decoder layers predict 12 future seven-dimensional actions in parallel from learned action queries. Dropout is 0.1. The six continuous outputs control relative pose; the last output is a gripper logit. [S2, S3]')

# 3
page();h('3. Training and evaluation protocol')
table([
 ['Setting','Recorded value'],
 ['Optimizer / schedule','AdamW; weight decay 0.001; 500-step warmup; cosine decay to 10% of peak LR'],
 ['Learning rates','Fusion/action head: 1e-4; trainable CLIP visual encoder: 1e-5'],
 ['Training budget','10,000 nominal optimizer iterations; checkpoint every 2,000 steps'],
 ['Batching / workers','16 x 4 accumulation for five groups; V2.1 uses 8 x 8; effective batch 64; two workers'],
 ['Numerics / stabilization','FP16 autocast with FP32 parameters; gradient scaling; gradient clipping at 1.0'],
 ['EMA / seed','EMA decay 0.999 for the trainable modules; training seed 42'],
 ['Evaluation / control','10 tasks x 10 episodes; seed 42; 220 control steps; 10 settling steps'],
 ['Action execution','Execute 12 predicted actions before observing again; reset action queue per episode'],
 ],[1.3,4.4])
cap('Table 3. Checkpoint-derived training settings and recorded evaluation protocol. Effective batch equality does not remove all numerical and sampling differences between microbatch layouts. [S1-S3]')
h2('3.1. Masked behavior-cloning objective')
p('For each sampled frame, the target is the next 12 actions. Windows that extend beyond a trajectory end are padded and masked. The loss averages Smooth L1 over the six normalized motion dimensions and adds binary cross-entropy for gripper open/close classification; both terms exclude padded timesteps. With valid-timestep mask m, the implementation computes:')
p('L = [sum m (mean<sub>j=1..6</sub> SmoothL1(a<sub>j</sub>, y<sub>j</sub>) + BCE(g, y<sub>g</sub>))] / sum m.','equation')
p('During rollout, continuous outputs are denormalized, the gripper logit is thresholded at zero to obtain -1 or +1, and all actions are clipped to [-1, 1]. Training loss is evaluated on demonstration states, whereas rollout success depends on the full sequence of states induced by the policy. [S3]')
h2('3.2. Outcome definition and pairing')
p('A rollout succeeds when the environment success predicate becomes true within the horizon. The reported aggregate is the mean task success rate, which equals episode-level accuracy because every task contributes 10 episodes. All 30 episode logs contain the same task/episode initial-state indices, enabling paired comparisons across models and checkpoints. The 3,000 logged trials reuse 100 evaluation conditions; they are not 3,000 independent test conditions.')
h2('3.3. Evidence validation and uncertainty')
p('For every evaluation, progress.json contains exactly 100 Boolean outcomes and agrees with both log.json and the 100 unique episode records. Twenty-four evaluations retain a full metadata header confirming EMA weights and the protocol above. The six older evaluations for V1/V1.1 at 6k, 8k, and 10k lack that header: their outcomes and initial-state pairing are verified, but the historical weight selection and complete protocol cannot be independently re-established from those files. Wilson 95% intervals are descriptive binomial summaries; paired exact McNemar tests are exploratory, unadjusted, and do not account for task clustering or training-seed variability. [S1]')

# 4
page();h('4. Aggregate closed-loop results')
rows=[['Model','2k','4k','6k','8k','10k','Best observed']]
for v in vs:
 rr=[r for r in a['runs'] if r['version']==v];best=max(rr,key=lambda r:r['rate'])
 rows.append([names[v],*[f"{r['rate']}%" for r in rr],f"{best['rate']}% at {best['step']//1000}k"])
table(rows,[.8,.65,.65,.65,.65,.65,1.45])
cap('Table 4. Success rates over 100 episodes per checkpoint. Each cell is recomputed from committed episode outcomes. Best observed scores are post hoc maxima on the same evaluation set and are not independent validation estimates. [S1]')
fig('success_curves.png',194)
cap('Figure 2. Success trajectories across the five checkpoint stages. Colors indicate 0, 2, and 6 fusion blocks within each visual-training regime. The final checkpoint is used for the main matched-budget comparison. [S1]')
h2('4.1. Visual fine-tuning and fusion interact')
p('At the final checkpoint, visual fine-tuning raises success from 84% to 92% without fusion (+8 percentage points, pp), from 84% to 85% with two blocks (+1 pp), and from 79% to 96% with six blocks (+17 pp). Thus, the largest observed gain occurs when a deeper fusion module and visual adaptation are used together. V2.2 also exceeds the no-fusion trainable model V2 by 4 pp and the two-block V2.1 model by 11 pp; these are observed differences in this evaluation, not established population effects.')
h2('4.2. Training stage matters')
p('At 2k steps, all variants perform poorly (2%-26%). By 4k, V2 reaches 91%, ahead of V2.2 at 86%. V2.2 becomes the best observed configuration from 6k onward (95%, 91%, 96%). Success is non-monotonic: V1 peaks at 6k and drops by 5 pp by 10k; V2.1 drops by 3 pp from its 6k maximum. These fluctuations caution against treating later checkpoints or lower loss as automatic improvements.')
p('Averaging the three late checkpoints gives 80.3%-84.7% for frozen-vision groups and 86.3%-94.0% for trainable-vision groups. These averages summarize correlated stages of the same training runs; they are not repeated-seed estimates.','small')

# 5
page();h('5. Ablation comparisons and uncertainty')
fig('final_intervals.png',161)
cap('Figure 3. Final-checkpoint success with descriptive 95% Wilson intervals (n=100 per model). The intervals summarize episode outcomes under the recorded evaluation conditions and do not quantify variation across independently trained models. [S1]')
rows=[['Contrast (B minus A)','Gain','B-only wins','A-only wins','Exact p']]
for av,bv in [('v1','v2'),('v1_1','v2_1'),('v1_2','v2_2'),('v2','v2_2'),('v2_1','v2_2')]:
 r=next(r for r in a['contrasts'] if r['a']==av and r['b']==bv)
 rows.append([f'{names[bv]} vs {names[av]}',f"+{r['gain_pp']} pp",r['wins'],r['losses'],f"{r['mcnemar_exact_p']:.5f}" if r['mcnemar_exact_p']<.01 else f"{r['mcnemar_exact_p']:.3f}"])
table(rows,[2.0,.8,1,1,1])
cap('Table 5. Paired final-checkpoint contrasts. A B-only win is an initial condition where B succeeds and A fails. Exact two-sided McNemar p-values use only discordant pairs; they are nominal, not multiplicity-adjusted, and assume independent pairs despite shared task structure. [S1]')
h2('5.1. What the evidence supports')
p('The strongest within-record contrast is V2.2 versus V1.2: 19 evaluation conditions improve and two regress, producing +17 pp (nominal p=0.00022). This supports prioritizing joint visual adaptation for this architecture, while remaining conditional on this single training seed and task collection. By contrast, the 4 pp difference between V2.2 and V2 consists of seven wins and three regressions (p=0.344); this sample does not establish a reliable superiority of six fusion blocks over no fusion.')
p('V2.2 obtains a descriptive interval of 90.2%-98.4%, compared with 85.0%-95.9% for V2 and 75.6%-89.9% for V1. The broad overlap and only 10 episodes per task limit fine-grained rankings. The V2.1 comparison is additionally affected by its 8 x 8 batching layout and resumed training history. Equal effective batch sizes should therefore be described as budget matching, not complete experimental control.')
h2('5.2. Final checkpoint versus model selection')
p('Using 10k for all groups avoids selecting each model at a different budget, although the report itself remains retrospective. Choosing V1 at its 6k maximum yields 89%, while V2.2 at its best stage yields 96%; the resulting 7 pp gap is smaller than the 12 pp final-checkpoint gap. Neither selection rule substitutes for evaluating a chosen model on fresh seeds and unused initial conditions. A future protocol should predefine the selection rule and reserve a separate final test set.')

# 6
page();h('6. Task-level performance and remaining failures')
fig('task_heatmap.png',189)
cap('Figure 4. Final-checkpoint success (%) by task. Each cell uses only 10 episodes, so values change in increments of 10 pp. Task IDs follow the common episode logs. [S1]')
short=['Between plate and ramekin','Next to ramekin','Table center','On cookie box','Inside top drawer','On ramekin','Next to cookie box','On stove','Next to plate','On wooden cabinet']
rows=[['Task','Initial bowl location','V1','V2','V2.2']]
for i,t in enumerate(final('v1')['per_task']):rows.append([f'T{i}',short[i],f"{final('v1')['per_task'][t]*100:.0f}%",f"{final('v2')['per_task'][t]*100:.0f}%",f"{final('v2_2')['per_task'][t]*100:.0f}%"])
table(rows,[.5,3,.65,.65,.8],True)
cap('Table 6. Task key and selected final-checkpoint results. All tasks instruct the robot to pick up the black bowl and place it on the plate; only the initial spatial relation differs. [S1, S4]')
h2('6.1. Where the aggregate improvement occurs')
p('Relative to V1, V2.2 improves the wooden-cabinet task from 60% to 100% and the next-to-plate task from 70% to 100%. It matches V1 on the next-to-ramekin and next-to-cookie-box tasks, and drops from 100% to 90% at table center. Relative to V2, gains are concentrated on the ramekin (+20 pp), next-to-cookie-box (+20 pp), next-to-ramekin (+10 pp), and drawer (+10 pp) tasks, offset by 10 pp drops at table center and on the stove.')
p('V2.2 succeeds in all 10 episodes on six tasks and in nine episodes on the remaining four. Its four failures occur at T1, T2, T5, and T7. The outcome logs do not classify grasp slips, collisions, missed localization, or placement errors. These mechanisms therefore cannot be assigned as observed failure causes without reviewing trajectories or videos.','small')

# 7
page();h('7. Optimization behavior and resource observations')
fig('training_loss.png',189)
cap('Figure 5. Logged training loss, smoothed with a trailing mean over 20 logged observations (roughly 400 training iterations). Dashed curves freeze the visual encoder; solid curves train it. The loss is computed on raw online parameters rather than the evaluated EMA parameters. [S2, S3]')
rows=[['Model','Tail loss','Logged peak GPU memory','Logged training time']]
for v,t in a['train'].items():
 elapsed=t['last_elapsed_seconds']/3600
 if v=='v2_1':elapsed=4.9161849539833335
 rows.append([names[v],f"{t['last20_mean_loss']:.4f}",f"{t['peak_vram_gib']:.2f} GiB",f'{elapsed:.2f} h'+(' (sum of 3 sessions)' if v=='v2_1' else '')])
table(rows,[.8,1,1.7,2])
cap('Table 7. Tail loss is the mean of the final 20 unique logged steps. Memory is the maximum allocated-GPU-memory value reported by the training logs, not total device usage. V2.1 time sums the final elapsed counter of each resumed session; other times use the final counter. These are heterogeneous run observations, not controlled hardware benchmarks. [S2]')
h2('7.1. Demonstration fitting versus closed-loop behavior')
p('Visual fine-tuning reduces tail loss to about 0.029-0.030, versus 0.155-0.171 with frozen visual features. Yet V2.1 has the lowest tail loss (0.0290) and only 85% final success, whereas V2.2 has a slightly higher loss (0.0299) and 96% success. Likewise, frozen V1.2 fits demonstrations better than V1 by this loss summary but achieves lower final success (79% versus 84%). The loss ranking therefore fails to reproduce the rollout ranking.')
p('Several differences explain why the two quantities need not align: the loss averages local action errors on demonstration states; rollout success is a thresholded long-horizon outcome on policy-induced states; and evaluation uses EMA where recorded, while the loss comes from raw training weights. These are properties of the objective and evaluation design, rather than proof that any single failure is caused by distribution shift.')
h2('7.2. Capacity and cost')
p('Trainable parameters rise from 4.62 M in V1 to 99.89 M in V2.2. The highest logged memory rises from 0.86 GiB to 5.53 GiB. V2.2 records 5.30 h of training versus 3.28 h for V2, but historical GPU identity and system load are not captured. A speedup or cost-efficiency claim cannot be inferred from these times alone.','small')

# New measured inference section
page();h('8. Measured GPU inference latency')
b=json.loads((ROOT/'outputs/benchmarks/inference_20261111/summary.json').read_text())
h2('8.1. Measurement protocol and timing boundary')
p('The existing scripts/measure_inference_time.py was executed on the current RTX 3060 Laptop GPU (6 GiB), with PyTorch 2.6.0+cu124, CUDA build 12.4, and driver 595.99.02. All six models use their 10k EMA checkpoint, FP32 parameters and computation, batch size one, two 224 x 224 RGB inputs, an eight-dimensional state, and the same 21-token instruction. Three sequential measurement rounds randomize model order (schedule seed 42). Each process performs 50 warmup predictions followed by 200 timed predictions, giving 600 samples per model and 3,600 in total. Pilot and superseded diagnostic runs are excluded. [S6]')
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

# 9
page();h('9. Interpretation, causal hypotheses, and limitations')
h2('9.1. Why visual adaptation may help')
p('<b>Observed:</b> trainable-vision variants have substantially lower behavior-cloning loss, and the final success gain over the corresponding frozen model is +8, +1, or +17 pp. <b>Hypothesis:</b> action supervision may adapt CLIP patch features to task-relevant geometry, contact cues, and camera appearance. The large V1.2-to-V2.2 gain is consistent with this explanation, but feature probes, localization measurements, and repeated-seed ablations would be needed to demonstrate it. These experiments do not measure representation quality directly.')
h2('9.2. Why more fusion is not uniformly better')
p('<b>Observed:</b> with frozen vision, final success is 84%, 84%, and 79% for 0, 2, and 6 fusion blocks. With trainable vision, the corresponding results are 92%, 85%, and 96%. <b>Hypothesis:</b> fusion may be most useful when the visual features can co-adapt to the language-conditioned control objective. When vision remains fixed, extra capacity may mainly reduce demonstration error without improving the geometry of action decisions. Optimization sensitivity and finite-sample variation are alternative explanations. The recorded data do not justify a monotonic depth-scaling claim.')
h2('9.3. Why early checkpoints are weak and late success fluctuates')
p('<b>Observed:</b> 2k success is 2%-26%, followed by large gains at 4k, while several later trajectories dip. <b>Hypotheses:</b> early under-training and EMA lag can contribute to the initial deficit. With fixed decay 0.999, the EMA half-life is approximately 693 updates, and the initial parameter copy retains about 13.5% weight after 2,000 successful updates. This arithmetic suggests lag as a plausible mechanism, but there is no raw-versus-EMA evaluation to isolate it. Later fluctuations could reflect policy sensitivity, EMA dynamics, imperfect demonstration coverage, or small-sample noise; they do not by themselves prove overfitting. [S3]')
h2('9.4. Limits on causal and generalization claims')
p('Only one recorded training seed is available per architecture, and the same 100 initial conditions are reused across checkpoints. Task outcomes are clustered, and multiple retrospective comparisons increase selection risk. The six older V1/V1.1 evaluations lack full metadata. V2.1 uses different microbatching and has three logged training sessions; its historical output paths also change. All groups train and evaluate within the same 10-task suite, so the study establishes neither unseen-language generalization nor transfer to other LIBERO suites or MT50. The documented earlier validation-split change has no matched archived control in this corpus; no numerical benefit is attributed to that change here.')
h2('9.5. Experiments that would resolve the main uncertainties')
p('First, repeat the 2 x 3 visual-training/fusion design with at least three independent training seeds, identical microbatching, and a predeclared final-checkpoint protocol. Second, evaluate more initial states with task-aware uncertainty estimates. Third, compare raw and EMA weights at 2k and 4k to test the lag hypothesis. Fourth, vary action execution length (for example, 1, 4, 8, and 12 steps) with one fixed trained policy to measure replanning effects. Finally, label failures from videos, then evaluate the eight-block V4 and additional suites as separate studies. These are proposed follow-ups; none is counted as a completed result.')

# 9
page();h('10. Conclusions and evidence provenance')
p('The audited experiments favor adapting the visual encoder and show that fusion depth interacts with that adaptation. V2.2 is the best observed model at the shared 10k budget, reaching 96% on the 100 recorded LIBERO-Spatial episodes. Its gain is 17 pp over the matching frozen six-block model, 12 pp over the frozen no-fusion baseline, and 4 pp over the trainable no-fusion model. The last of these differences is small relative to the available sample and is not established as a robust advantage. Training loss is useful for monitoring optimization, but closed-loop evaluation is necessary to compare control performance.')
p('Training and closed-loop results are reanalyses of existing artifacts. The inference measurements are new GPU microbenchmarks; no new training or simulation run was performed. Numerical results were recomputed from outcome records; model identity and training settings were read from checkpoint metadata, which take precedence over renamed folders and current default YAML files. No outcome was inferred for an unevaluated model.')
h2('10.1. Local evidence map')
table([
 ['ID','Repository evidence','Use in this report'],
 ['S1','outputs/eval/libero_spatial_v*/ckpt{step}/seed42/<br/>progress.json, log.json, episodes.jsonl','30 evaluations; aggregate/task outcomes; paired initial conditions; metadata coverage'],
 ['S2','outputs/train/libero_spatial_v*/<br/>checkpoint_{step}.pt, manifest.json,<br/>metrics.jsonl or log.jsonl','Canonical architecture; actual training settings; demonstration counts; loss/time/memory logs'],
 ['S3','src/myvla/model.py, data.py, preprocess.py,<br/>train.py, evaluate.py, ema.py','Policy structure; sampling; loss; action decoding; rollout; EMA implementation'],
 ['S4','data/libero_spatial/*_demo.hdf5;<br/>pyproject.toml; installed package inventory','Example image schema; dependency specification and current local versions'],
 ['S5','configs/train_cfg_v4.yaml;<br/>README.md; model-version history document','Planned eight-block configuration and contextual documentation; not substituted for raw outcomes'],
 ['S6','scripts/measure_inference_time.py;<br/>outputs/benchmarks/inference_20261111/','18 timing JSON files (600 samples per model); measurement manifest; summary.json and summary.csv'],
 ],[.45,3.5,2.7],True)
cap('Table 9. Paths are relative to the project root. In S1/S2, v* expands to v1, v1_1, v1_2, v2, v2_1, or v2_2, and step expands to 2000, 4000, 6000, 8000, or 10000. Canonical labels use dots in the report and underscores in file paths.')
h2('10.2. Reproducibility details and unresolved records')
p('All models pin CLIP to revision <font name="Sans" size="8">57c216476eefef5ab752ec549e440a49ae4ae5f3</font>. Archived folders use older labels, including a two-block trainable configuration named v4 in cloud paths; this corresponds to canonical V2.1, not the current eight-block V4. Matching trajectory identities and lengths are verified across the six manifests, but file paths and timestamps alone do not establish a cryptographic hash of every HDF5. Historical GPU/driver details and the LIBERO source Git revision remain unrecorded. The evidence cutoff is 11 November 2026.')
h2('References')
p('[1] A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. <i>Proceedings of ICML</i>, PMLR 139:8748-8763, 2021. <link href="https://proceedings.mlr.press/v139/radford21a.html" color="#326e70">proceedings.mlr.press/v139/radford21a.html</link>','small')
p('[2] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. 2023. <link href="https://arxiv.org/abs/2306.03310" color="#326e70">arXiv:2306.03310</link>. Used for benchmark context; no published baseline scores are compared in this report.','small')

class NumberedCanvas(canvas.Canvas):
 def __init__(self,*args,**kwargs):canvas.Canvas.__init__(self,*args,**kwargs);self.saved=[]
 def showPage(self):self.saved.append(dict(self.__dict__));self._startPage()
 def save(self):
  n=len(self.saved)
  for state in self.saved:
   self.__dict__.update(state);self.setFillColor(GRAY);self.setFont('Sans',8)
   self.drawString(M,H-29,'myVLA | Experimental study on LIBERO-Spatial')
   self.setStrokeColor(colors.HexColor('#c4cdd1'));self.setLineWidth(.4);self.line(M,H-36,W-M,H-36)
   self.line(M,36,W-M,36);self.drawString(M,23,'Repository evidence through 11 November 2026');self.drawRightString(W-M,23,f'{self._pageNumber} / {n}')
   canvas.Canvas.showPage(self)
  canvas.Canvas.save(self)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=M,leftMargin=M,topMargin=49,bottomMargin=49,title='myVLA: An Experimental Study of Visual Adaptation and Multimodal Fusion',author='myVLA Project',subject='Audited experimental report on LIBERO-Spatial',pageCompression=1)
def decorate(c,doc):
 c.saveState()
 c.setFillColor(GRAY);c.setFont('Sans',8)
 c.drawString(M,H-29,'myVLA | Experimental study on LIBERO-Spatial')
 c.setStrokeColor(colors.HexColor('#c4cdd1'));c.setLineWidth(.4);c.line(M,H-36,W-M,H-36)
 c.line(M,36,W-M,36);c.drawString(M,23,'Repository evidence through 11 November 2026');c.drawRightString(W-M,23,f'{doc.page} / 10')
 c.restoreState()
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
print(OUT)
