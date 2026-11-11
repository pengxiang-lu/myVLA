from pathlib import Path
import json,math
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab.platypus.paragraph as rl_paragraph
import reportlab.lib.textsplit as rl_textsplit
# 补齐中文行首禁则，避免逗号、分号等标点单独落在行首。
for module in (rl_paragraph, rl_textsplit):
 module.ALL_CANNOT_START += '，；：！？）》〉”’'

ROOT=Path('/home/lpx/python-project/myVLA');D=ROOT/'tmp/pdfs';OUT=ROOT/'myVLA_experimental_report_zh.pdf'
a=json.loads((D/'experiment_audit.json').read_text());b=json.loads((ROOT/'outputs/benchmarks/inference_20261111/summary.json').read_text())
vs=list(a['train']);names={v:v.upper().replace('_','.') for v in vs}
pdfmetrics.registerFont(TTFont('Song','/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'))
pdfmetrics.registerFont(TTFont('Hei','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('Song',normal='Song',bold='Hei',italic='Song',boldItalic='Hei')
pdfmetrics.registerFontFamily('Hei',normal='Hei',bold='Hei',italic='Hei',boldItalic='Hei')
INK=colors.HexColor('#17232e');GRAY=colors.HexColor('#596875');TEAL=colors.HexColor('#2f7377');LIGHT=colors.HexColor('#eef3f4');W,H=A4;M=46;CW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='Song',fontSize=10.5,leading=17,spaceAfter=7,firstLineIndent=21,alignment=TA_JUSTIFY,wordWrap='CJK',textColor=INK),
 'small':ParagraphStyle('small',fontName='Song',fontSize=9.3,leading=14.5,spaceAfter=6,wordWrap='CJK',textColor=GRAY),
 'h1':ParagraphStyle('h1',fontName='Hei',fontSize=15,leading=21,spaceAfter=12,spaceBefore=2,keepWithNext=True,textColor=INK),
 'h2':ParagraphStyle('h2',fontName='Hei',fontSize=11.5,leading=17,spaceBefore=7,spaceAfter=6,keepWithNext=True,textColor=INK),
 'caption':ParagraphStyle('caption',fontName='Song',fontSize=9.2,leading=14,spaceAfter=10,wordWrap='CJK',textColor=GRAY),
 'cell':ParagraphStyle('cell',fontName='Hei',fontSize=8.5,leading=13,wordWrap='CJK',textColor=INK),
 'cellsmall':ParagraphStyle('cellsmall',fontName='Hei',fontSize=8,leading=12,wordWrap='CJK',textColor=INK),
 'headcell':ParagraphStyle('headcell',fontName='Hei',fontSize=8.6,leading=13,wordWrap='CJK',textColor=INK),
 'title':ParagraphStyle('title',fontName='Hei',fontSize=23,leading=32,alignment=TA_CENTER,spaceAfter=12,textColor=INK),
 'subtitle':ParagraphStyle('subtitle',fontName='Hei',fontSize=9.5,leading=15,alignment=TA_CENTER,spaceAfter=17,textColor=GRAY),
 'abstract':ParagraphStyle('abstract',fontName='Song',fontSize=10,leading=16,spaceAfter=12,wordWrap='CJK',alignment=TA_JUSTIFY,leftIndent=12,rightIndent=12,textColor=INK),
 'equation':ParagraphStyle('equation',fontName='Hei',fontSize=10,leading=17,alignment=TA_CENTER,spaceBefore=4,spaceAfter=10,textColor=INK),
}
story=[]
def p(t,style='body'):story.append(Paragraph(t,styles[style]))
def h(t):p(t,'h1')
def h2(t):p(t,'h2')
def cap(t):p(t,'caption')
def page():story.append(PageBreak())
def table(rows,widths,small=False):
 data=[[Paragraph(str(c),styles['headcell'] if i==0 else styles['cellsmall' if small else 'cell']) for c in row] for i,row in enumerate(rows)]
 t=Table(data,colWidths=[CW*x/sum(widths) for x in widths],repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEABOVE',(0,0),(-1,0),.8,INK),('LINEBELOW',(0,0),(-1,0),.5,INK),('LINEBELOW',(0,-1),(-1,-1),.7,INK)]));story.extend([t,Spacer(1,5)])
def fig(name,height):story.append(Image(str(D/name),width=CW,height=height))
def final(v):return next(r for r in a['runs'] if r['version']==v and r['step']==10000)

class Architecture(Flowable):
 def __init__(self):Flowable.__init__(self);self.width=CW;self.height=492
 def draw(self):
  c=self.canv
  blue=colors.HexColor('#467da4');green=colors.HexColor('#45806a');violet=colors.HexColor('#8272a0');orange=colors.HexColor('#b78443')
  def box(x,y,w,hh,lines,color=TEAL,fill=LIGHT,dashed=False,fs=8.5):
   c.saveState();c.setStrokeColor(color);c.setFillColor(fill);c.setLineWidth(.8)
   if dashed:c.setDash(4,2)
   c.roundRect(x,y,w,hh,4,fill=1,stroke=1);c.setDash();c.setFillColor(INK);c.setFont('Hei',fs)
   gap=13;start=y+hh/2+(len(lines)-1)*gap/2-3
   for i,line in enumerate(lines):c.drawCentredString(x+w/2,start-i*gap,line)
   c.restoreState()
  def arrow(points,color=GRAY,dashed=False):
   c.saveState();c.setLineWidth(.9);c.setStrokeColor(color)
   if dashed:c.setDash(3,2)
   path=c.beginPath();path.moveTo(*points[0])
   for xy in points[1:]:path.lineTo(*xy)
   c.drawPath(path);c.setDash();x,y=points[-1];px,py=points[-2];ang=math.atan2(y-py,x-px);length=5
   for direction in [-1,1]:c.line(x,y,x-length*math.cos(ang+direction*.5),y-length*math.sin(ang+direction*.5))
   c.restoreState()
  box(0,452,182,30,['双相机 RGB：2 × 3 × 224 × 224'],blue,fs=8)
  box(218,452,182,30,['任务指令 → 分词（长度 L ≤ 77）'],green,fs=8)
  box(425,452,76,30,['机器人状态 8 维'],violet,fs=8)
  box(0,378,182,52,['CLIP ViT-B/16 视觉编码器','V1 系列冻结；V2 系列参与训练','去除 CLS：2 × 196 × 768'],blue)
  box(218,378,182,52,['CLIP 文本编码器：始终冻结','词元序列 L × 512','句向量 1 × 512'],green,dashed=True)
  box(425,378,76,52,['状态 MLP','8 → 256','可训练'],violet,fs=8)
  arrow([(91,452),(91,430)],blue);arrow([(309,452),(309,430)],green);arrow([(463,452),(463,430)],violet)
  box(0,320,182,39,['视觉投影 + 相机嵌入','展平为 392 × 256 视觉词元'],blue)
  box(218,320,182,39,['文本线性投影','词元 L × 256；句向量 1 × 256'],green,fs=8)
  box(425,320,76,39,['状态嵌入','1 × 256'],violet,fs=8)
  arrow([(91,378),(91,359)],blue);arrow([(309,378),(309,359)],green);arrow([(463,378),(463,359)],violet)
  box(10,222,380,74,['双向视觉—语言交叉注意力融合 × N','视觉查询文本；文本查询视觉；残差门控以零初始化','N = 0 / 2 / 6；V2.2 使用 N = 6','保留更新后的视觉词元，并提取文本 EOS 交互残差'],TEAL,colors.HexColor('#e6f1ef'),fs=9)
  arrow([(91,320),(91,296)],blue);arrow([(309,320),(309,296)],green)
  box(10,130,380,58,['拼接多模态 memory：394 × 256','392 个视觉词元 + 1 个语言总结 + 1 个状态词元','语言总结 = 原始句向量投影 + EOS 交互残差','语言 / 状态词元加入模态嵌入'],violet,colors.HexColor('#f0edf5'),fs=8.5)
  arrow([(200,222),(200,188)],TEAL)
  arrow([(400,333),(412,333),(412,201),(310,201),(310,188)],green,True)
  c.setFont('Hei',7.3);c.setFillColor(green);c.drawCentredString(352,208,'原始句向量旁路')
  arrow([(463,320),(463,308),(493,308),(493,159),(390,159)],violet)
  c.setFillColor(violet);c.setFont('Hei',7.5);c.drawCentredString(450,251,'状态绕过融合层')
  box(10,51,380,54,['Transformer 动作解码器 × 4','隐藏维度 256；8 个注意力头','12 个查询并行预测，不逐步自回归生成'],orange,colors.HexColor('#faf0e3'),fs=8.5)
  box(425,51,76,54,['可学习的','动作查询','12 × 256'],orange,fs=8)
  arrow([(200,130),(200,105)],violet);arrow([(425,78),(390,78)],orange)
  box(35,0,330,32,['线性动作头 → 12 × 7 动作块','6 维位姿反归一化 + 夹爪阈值化 + 动作裁剪'],orange,colors.HexColor('#faf0e3'),fs=8)
  arrow([(200,51),(200,32)],orange)

# 第一页
p('myVLA 实验研究报告<br/>视觉编码适配与多模态融合的对比分析','title')
p('LIBERO-Spatial 行为克隆实验｜中文论文式实验报告<br/>数据与测量截止日期：2026 年 11 月 11 日','subtitle')
p('<b>摘要：</b>本报告汇总 myVLA 六组模型在五个训练阶段的闭环评估结果，研究视觉编码器是否参与训练、双向视觉—语言融合层深度，以及训练损失与任务成功率之间的关系。数据覆盖 500 条演示、62,250 帧和 30 组评估，共 3,000 个回合。在 10,000 步检查点，六层融合且训练视觉编码器的 V2.2 达到 96% 成功率；冻结视觉的无融合基线为 84%，训练视觉但无融合的 V2 为 92%。另在 RTX 3060 Laptop GPU 上完成 3,600 次推理计时，V2.2 预测完整 12 步动作块的中位耗时为 25.21 ms。结果表明，视觉适配与融合深度存在交互，增加融合层数并不保证性能单调提升。报告同时讨论单一训练种子、有限评估样本、历史环境记录不完整等局限。','abstract')
h('1　研究目标与实验分组')
p('本工程使用预训练 CLIP ViT-B/16 编码器［1］和可训练动作解码器，开展 LIBERO-Spatial［2］多任务行为克隆。实验重点是：视觉特征能否通过动作监督进一步适配控制任务；双向交叉注意力的深度是否改善闭环成功率；离线训练损失能否作为模型选择依据。当前实验属于同一套件内的联合多任务学习，不是持续学习或未见任务泛化评估。')
rows=[['模型','视觉编码器','融合层数','总参数量','可训练参数量']]
for v,t in a['train'].items():rows.append([names[v],'参与训练' if t['config']['model']['train_vision_encoder'] else '冻结',t['config']['model']['fusion_layers'],f"{t['total_parameters']/1e6:.2f} M",f"{t['trainable_parameters']/1e6:.2f} M"])
table(rows,[.8,1.2,.8,1.15,1.3])
cap('表 1　六组模型的规范名称与参数规模。模型按检查点中的实际结构识别，不仅按历史文件夹名判断。总参数包含冻结且未用于策略前向的 CLIP 原始投影参数。［S2、S3］')
p('所有组均冻结文本编码器，使用 4 层动作解码器并输出 12×7 动作。当前 V4 配置为 8 层融合，但没有已完成的闭环评估；工程虽支持 MetaWorld MT50，也没有相应训练与评估结果，因此两者不纳入数值比较。', 'small')

# 第二页：详细框图
page();h('2　VLA 模型架构框图')
p('图 1 展示当前 LIBERO 策略的完整数据流，以 V2.2 的六层融合为代表。图中维度省略 batch 维。视觉编码器的训练范围和融合层数随表 1 变化，其余主干流程相同。', 'small')
story.extend([Architecture(), Spacer(1,8)])
cap('图 1　myVLA 的视觉—语言—动作架构。虚线边框表示始终冻结的文本编码器；绿色虚线路径表示原始句向量的旁路。状态特征直接进入 memory，不经过双向融合。N=0 时跳过融合，语言总结直接使用原始句向量投影；当前八层 V4 尚无已完成评估。框图依据 model.py 与 preprocess.py 绘制。［S3］')
h2('2.1　融合、动作查询与闭环执行')
p('每个相机提供 196 个图像 patch 词元；双相机合计 392 个。融合模块在视觉和文本词元之间交换信息，文本分支的 EOS 交互残差加入原始句向量。随后将视觉词元、语言总结和状态词元拼接为 memory，由 12 个可学习查询并行解码未来动作。模型一次输出完整动作块，评估时顺序执行 12 步，再获取新观察并重新规划。')

# 第三页
page();h('3　数据集、实现与实验环境')
h2('3.1　演示数据及输入预处理')
p('六组检查点的清单均包含相同的 500 条训练轨迹，来自 10 个 LIBERO-Spatial 任务，每任务 50 条演示，共 62,250 帧；清单中全部样本均标记为训练集，未划分演示验证集。采样器按帧有放回均匀采样，因此较长轨迹贡献更多训练样本。10,000 次训练迭代、有效 batch=64，对应 640,000 次帧采样，约为数据规模的 10.28 倍；这不等同于按顺序遍历 10.28 个 epoch。［S2、S3］')
p('输入包括外部相机与腕部相机 RGB、任务指令，以及 8 维机器人状态：末端位置、轴角姿态和两个夹爪位置。检查到的 HDF5 图像为 128×128；预处理进行垂直翻转、双三次插值至 224×224，再按 CLIP 均值和标准差归一化。状态及连续动作统计量来自训练演示；图像平移与颜色扰动均关闭。［S2—S4］')
table([
 ['组件','版本或配置','证据范围'],
 ['Python / 工程包','Python 3.10.20；myvla 0.1.0','Python 为当前安装环境；工程包版本来自评估元数据'],
 ['学习框架','PyTorch 2.6.0；Transformers 4.57.6；NumPy 1.26.4','24/30 组评估头部保留了这些历史版本'],
 ['仿真环境','robosuite 1.4.1；MuJoCo 2.3.7；相对 OSC_POSE','来自评估元数据；未记录 LIBERO 源码 Git 修订号'],
 ['CLIP 主干','ViT-B/16；固定修订号 57c21647…e4ae5f3','六组检查点保存的完整修订号一致'],
 ['当前 CPU / 内存','Intel Core i9-12900H；约 15.35 GiB 内存','报告当天观测值，不能认定为全部历史训练环境'],
 ['当前 GPU','RTX 3060 Laptop GPU；6,144 MiB','用于新增推理测量；历史训练 GPU 型号未记录'],
 ['当前系统 / CUDA','Ubuntu 24.04.5 LTS；CUDA 构建版本 12.4；驱动 595.99.02','当前 PyTorch 安装包为 2.6.0+cu124'],
 ],[1.05,2.5,2.45],True)
cap('表 2　软件与硬件环境及其证据来源。区分历史评估元数据与当前主机观测；V2.1 的云端风格路径不能证明其训练 GPU 型号。［S1、S2、S4］')
h2('3.2　共享模型配置')
p('特征投影维度为 256，注意力头数为 8，dropout=0.1。融合块采用双向交叉注意力和零初始化残差门控。解码器共 4 层，输出每步 7 维动作：前 6 维为相对位姿控制，最后 1 维为夹爪 logit。CLIP 原始对比投影层不参与策略前向。［S2、S3］')

# 第四页
page();h('4　训练与闭环评估协议')
table([
 ['设置','检查点与代码中记录的值'],
 ['优化器与学习率计划','AdamW；weight decay=0.001；预热 500 步；余弦衰减至峰值学习率的 10%'],
 ['分组学习率','融合层 / 动作头 1e-4；参与训练的视觉编码器 1e-5'],
 ['训练预算','10,000 次名义训练迭代；每 2,000 步保存检查点'],
 ['微批次与累积','五组为 16×4；V2.1 为 8×8；有效 batch 均为 64；2 个数据加载 worker'],
 ['数值精度与稳定化','FP16 混合精度，参数保持 FP32；梯度缩放；梯度裁剪阈值 1.0'],
 ['EMA / 随机种子','训练模块 EMA 衰减系数 0.999；训练种子 42'],
 ['评估条件','10 任务×每任务 10 回合；seed=42；每回合最多 220 个控制步；先稳定 10 步'],
 ['动作执行方式','一次预测 12 步并依次执行，然后重新观察；每回合重置动作队列'],
 ],[1.25,4.7])
cap('表 3　训练与评估设置。有效 batch 一致并不消除微批次布局造成的数值与采样差异；V2.1 的实际设置以检查点为准。［S1—S3］')
h2('4.1　带掩码的行为克隆损失')
p('每个训练帧对应未来 12 步目标动作。轨迹尾部不足 12 步时进行填充，并用掩码排除填充位置。前 6 个归一化连续动作维度采用 Smooth L1 并取维度均值；夹爪采用二元交叉熵 BCE。两项按有效时间步归一化求和：')
p('L = Σ m · [mean<sub>j=1…6</sub> SmoothL1(a<sub>j</sub>, y<sub>j</sub>) + BCE(g, y<sub>g</sub>)] / Σ m','equation')
p('推理时对连续动作反归一化，将夹爪 logit 以 0 为阈值映射到 -1 或 +1，最后将动作裁剪至 [-1, 1]。训练损失来自演示状态上的局部动作拟合；闭环成功率依赖策略实际访问状态下的整段行为，两者含义不同。［S3］')
h2('4.2　成功定义与配对初始条件')
p('若环境成功判定在控制步上限内成立，则该回合记为成功。每任务均有 10 个回合，因此任务成功率的平均值等于全体回合的成功比例。30 组日志的任务 / 回合初始状态索引完全一致，可进行配对对比。3,000 个回合重复使用了 100 个评估条件，不能视为 3,000 个独立测试条件。')
h2('4.3　结果核验与统计解释')
p('各组 progress.json 均包含 100 个布尔结果，与 log.json 汇总及 100 条唯一回合记录一致。24 组保留完整元数据，可核实 EMA 及上述协议；V1、V1.1 在 6k / 8k / 10k 的 6 组旧记录缺少完整头部，其结果和初始状态索引可核验，但历史权重选择与全部协议无法由这些文件独立复原。Wilson 95% 区间仅作描述；精确 McNemar 检验是未校正多重比较、未考虑任务聚类和训练种子变异的探索性分析。［S1］')

# 第五页
page();h('5　总体闭环结果比较')
rows=[['模型','2k 步','4k 步','6k 步','8k 步','10k 步','观测最优值']]
for v in vs:
 rr=[r for r in a['runs'] if r['version']==v];best=max(rr,key=lambda r:r['rate'])
 rows.append([names[v],*[f"{r['rate']}%" for r in rr],f"{best['rate']}%（{best['step']//1000}k）"])
table(rows,[.75,.65,.65,.65,.65,.65,1.4])
cap('表 4　各训练阶段成功率，每格均为 100 个回合的实际结果。观测最优值是在同一评估集上的事后最大值，不是独立验证结果。主要比较统一使用 10k 检查点。［S1］')
fig('zh_success_curves.png',194)
cap('图 2　五个训练阶段的成功率曲线。同一颜色对应相同融合深度，左图冻结视觉编码器，右图训练视觉编码器。［S1］')
h2('5.1　视觉适配与融合深度存在交互')
p('在 10k 检查点，解冻视觉编码器带来的成功率变化分别为：无融合 84%→92%（+8 个百分点）；两层融合 84%→85%（+1）；六层融合 79%→96%（+17）。最大观测增益出现在六层融合与视觉联合训练的组合。V2.2 相比 V2 高 4 个百分点，相比 V2.1 高 11 个百分点，但这些是本次评估中的观测差异，不能直接外推为总体性能差异。')
h2('5.2　训练阶段与检查点选择')
p('2k 时各组成功率仅为 2%—26%；4k 时 V2 达到 91%，领先 V2.2 的 86%。从 6k 起，V2.2 依次为 95%、91%、96%，均为当阶段最高值。曲线并不单调：V1 在 6k 达到 89% 后，于 10k 降至 84%；V2.1 从 6k 的 88% 降至 10k 的 85%。')
p('后三个检查点的平均成功率，冻结视觉组为 80.3%—84.7%，训练视觉组为 86.3%—94.0%。这些均值来自同一训练过程中的相关阶段，不能替代多训练种子的重复实验。', 'small')

# 第六页
page();h('6　消融对比与结果不确定性')
fig('zh_final_intervals.png',162)
cap('图 3　10k 检查点成功率与描述性 Wilson 95% 区间，每组 n=100。区间描述当前评估条件下的回合结果，不包含独立训练种子之间的波动。［S1］')
rows=[['对比（B 相对 A）','变化','仅 B 成功','仅 A 成功','精确 p 值']]
for av,bv in [('v1','v2'),('v1_1','v2_1'),('v1_2','v2_2'),('v2','v2_2'),('v2_1','v2_2')]:
 r=next(r for r in a['contrasts'] if r['a']==av and r['b']==bv)
 rows.append([f'{names[bv]} / {names[av]}',f"+{r['gain_pp']} 点",r['wins'],r['losses'],f"{r['mcnemar_exact_p']:.5f}" if r['mcnemar_exact_p']<.01 else f"{r['mcnemar_exact_p']:.3f}"])
table(rows,[1.8,.85,1,1,1])
cap('表 5　最终检查点的配对比较。“仅 B 成功”表示相同初始条件下 B 成功而 A 失败。双侧精确 McNemar p 值仅基于不一致结果对，未做多重比较校正，并依赖回合对独立的假设；实际存在共同任务结构，故只作探索性参考。［S1］')
h2('6.1　哪些结论获得较强的记录内支持')
p('V2.2 相对 V1.2 的证据最突出：19 个条件改善、2 个条件退化，净增 17 个百分点，名义 p=0.00022。这支持在当前架构与数据条件下优先研究视觉联合训练，但结论仍受单一训练种子与任务集合限制。V2.2 相对 V2 的 +4 点则来自 7 个改善和 3 个退化，p=0.344；现有样本不足以确立六层融合相对无融合的可靠优势。')
p('V2.2 的描述性区间为 90.2%—98.4%，V2 为 85.0%—95.9%，V1 为 75.6%—89.9%。区间较宽且部分重叠，每任务只有 10 回合，细粒度排名应谨慎解释。V2.1 另有 8×8 微批次布局和分段续训差异；有效 batch 相同表示预算相近，而不是完全受控的实验。')
h2('6.2　统一终点与事后挑选最优点')
p('统一比较 10k 可避免每个模型使用不同训练预算，但报告整体仍是回顾性分析。若各自挑选最佳检查点，V1 为 6k 的 89%，V2.2 为 10k 的 96%，差距为 7 点，小于统一终点时的 12 点。两种选择方式都不能替代在新种子和未使用初始条件上的最终测试；后续应预先规定模型选择规则。')

# 第七页
page();h('7　逐任务表现与剩余失败')
fig('zh_task_heatmap.png',190)
cap('图 4　最终检查点的逐任务成功率。每格只有 10 个回合，故以 10 个百分点为最小变化单位。任务编号与共同回合日志一致。［S1］')
short=['盘子与小碗之间','小碗旁','桌面中央','饼干盒上','木柜上层抽屉内','小碗上','饼干盒旁','炉灶上','盘子旁','木柜上']
rows=[['任务','黑碗初始位置','V1','V2','V2.2']]
for i,t in enumerate(final('v1')['per_task']):rows.append([f'T{i}',short[i],f"{final('v1')['per_task'][t]*100:.0f}%",f"{final('v2')['per_task'][t]*100:.0f}%",f"{final('v2_2')['per_task'][t]*100:.0f}%"])
table(rows,[.6,2.8,.8,.8,.8],True)
cap('表 6　任务索引与部分模型结果。全部任务均要求抓起黑碗并放到盘子上，变化在于初始空间关系；此处“小碗”对应任务名中的 ramekin。［S1、S4］')
h2('7.1　总体提升来自哪些任务')
p('相对 V1，V2.2 在木柜任务由 60% 提升至 100%，在盘子旁任务由 70% 提升至 100%；小碗旁和饼干盒旁保持相同；桌面中央则由 100% 降至 90%。相对 V2，V2.2 在小碗上、饼干盒旁各增加 20 点，小碗旁和抽屉内各增加 10 点，但在桌面中央和炉灶上各降低 10 点。')
p('V2.2 在六个任务上为 10/10，在其余四个任务上为 9/10；四次失败分别出现在 T1、T2、T5、T7。日志没有将失败标注为抓取滑落、碰撞、定位偏差或放置失败，因此未经轨迹或视频审查，不能把这些机制写为已确认的失败原因。', 'small')

# 第八页
page();h('8　训练优化表现与资源开销')
fig('zh_training_loss.png',190)
cap('图 5　训练损失曲线。每个点使用最近 20 个日志观测的滑动均值，约覆盖 400 次训练迭代；虚线冻结视觉编码器，实线训练视觉编码器。训练损失来自当前原始参数，而非评估使用的 EMA 参数。［S2、S3］')
rows=[['模型','末段平均损失','日志峰值显存','日志训练时间']]
for v,t in a['train'].items():
 hours=4.9161849539833335 if v=='v2_1' else t['last_elapsed_seconds']/3600
 rows.append([names[v],f"{t['last20_mean_loss']:.4f}",f"{t['peak_vram_gib']:.2f} GiB",f'{hours:.2f} 小时'+('（三段求和）' if v=='v2_1' else '')])
table(rows,[.8,1.2,1.3,2.0])
cap('表 7　末段损失为最后 20 个唯一步骤日志的均值；显存为日志中的最大已分配 GPU 显存，不是设备总占用。V2.1 将三段续训各自最后的耗时计数相加，其余取最后计数。历史硬件和负载不完整，不能把这些值当作受控速度基准。［S2］')
h2('8.1　更低的训练损失不保证更高的闭环成功率')
p('视觉联合训练的末段损失约为 0.029—0.030，明显低于冻结组的 0.155—0.171。但损失最低的 V2.1（0.0290）仅达 85% 成功率，损失稍高的 V2.2（0.0299）却达 96%。冻结组 V1.2 的拟合损失低于 V1，最终成功率反而较低（79% 对 84%）。因此，训练损失排序不能直接替代闭环表现排序。')
p('损失衡量演示状态上的局部动作误差；成功率是策略实际轨迹上的长时程阈值结果；已记录的评估使用 EMA，而训练曲线来自原始参数。这些目标与测量差异可以解释二者为何不必一致，但不能证明某一次失败一定来自分布偏移。')
h2('8.2　容量与开销')
p('可训练参数从 V1 的 4.62 M 增至 V2.2 的 99.89 M；日志峰值显存从 0.86 GiB 增至 5.53 GiB。V2.2 记录训练 5.30 小时，V2 为 3.28 小时；由于历史 GPU 与系统负载未记录，不能据此宣称硬件速度或成本效率的确定差异。', 'small')

# 第九页
page();h('9　GPU 推理耗时实测')
h2('9.1　测量协议与计时边界')
p('使用工程原有 scripts/measure_inference_time.py，在 RTX 3060 Laptop GPU（6 GiB）上运行；软件为 PyTorch 2.6.0+cu124、CUDA 构建版本 12.4、驱动 595.99.02。六组均使用 10k EMA 检查点、FP32 参数和计算、batch=1、双相机 224×224 输入、8 维状态，以及相同的 21-token 指令。三轮测量串行执行，各轮以固定日程种子 42 打乱模型顺序；每个进程预热 50 次、计时 200 次，最终每组 600 次，共 3,600 次。先导与已替换诊断运行不计入汇总。［S6］')
p('使用 CPU 墙钟计时，并在推理前后同步 CUDA。范围包含视觉编码、完整文本编码、动作头及设备端 12×7 动作块解码。固定合成输入已归一化且驻留 GPU，分词在计时前完成；不包含权重加载、图像预处理、输入传输、动作复制回 CPU 或环境执行。使用 eager 推理，未启用编译或 CUDA Graph。脚本刻意绕过实际闭环中的文本缓存，因此这里是“驻留输入、文本未缓存”的推理微基准，不是端到端闭环耗时。')
rows=[['模型','中位数/ms','均值/ms','P95/ms','三轮中位数范围/ms','动作块/秒']]
for v in vs:
 g=b['groups'][v];rows.append([names[v],f"{g['median_ms']:.2f}",f"{g['mean_ms']:.2f}",f"{g['p95_ms']:.2f}",f"{min(g['round_medians_ms']):.2f}—{max(g['round_medians_ms']):.2f}",f"{g['chunk_predictions_per_second']:.1f}"])
table(rows,[.7,.95,.95,.95,1.65,1.0],True)
cap('表 8　2026 年 11 月 11 日新增的 GPU 推理计时。中位数、均值和 P95 均由每组 600 次同步计时合并计算；范围列为三个独立加载进程的中位数最小值至最大值。动作块/秒=1000÷中位耗时。所有输出均有限，形状为 [1,12,7]。［S6］')
h2('9.2　融合深度、延迟与行为表现')
g=b['groups']['v2_2'];base=b['groups']['v2']
p(f"V2.2 每次完整预测的中位耗时为 {g['median_ms']:.2f} ms，P95 为 {g['p95_ms']:.2f} ms，约为 {g['chunk_predictions_per_second']:.1f} 个动作块/秒。相对无融合的 V2，中位耗时增加 {g['median_ms']-base['median_ms']:.2f} ms（{100*(g['median_ms']/base['median_ms']-1):.1f}%），对应已有闭环结果的 96% 与 92%。两类实验的测量对象不同，不能据此建立可外推的速度—精度前沿。")
p('同一融合深度下，训练时冻结视觉编码器并不会省略推理前向。因此，V1/V2、V1.1/V2.1、V1.2/V2.2 之间的小幅耗时差异，应考虑运行波动，不能直接归因为“冻结带来推理提速”。测量未锁定笔记本温度、GPU 频率和功耗管理，后台显示活动亦未完全控制。')
h2('9.3　动作摊销耗时与响应延迟的区别')
p(f"将 V2.2 的中位耗时摊到 12 个生成动作，约为 {g['amortized_ms_per_action']:.2f} ms/动作，但这只是核算方式：获得一次新观察后的完整预测仍需约 {g['median_ms']:.2f} ms，之后才按队列执行。它既不是单步反应延迟，也不是仿真实际控制频率。部署测量还需覆盖文本缓存路径、预处理、传输、CPU 动作读取和环境执行。",'small')

# 第十页
page();h('10　结果解释、原因假设与实验局限')
h2('10.1　视觉适配为什么可能有效')
p('<b>已观察到：</b>视觉联合训练显著降低行为克隆损失，最终成功率相对对应冻结组增加 8、1、17 个百分点。<b>可能机制：</b>动作监督使 CLIP patch 特征更适合任务相关的空间几何、接触线索和相机外观。V1.2→V2.2 的大幅提升与这一解释相符，但当前并未直接测量特征质量；仍需表征探针、定位指标和多种子消融验证，不能将其写为已证实的因果机制。')
h2('10.2　为什么融合层更多并不总是更好')
p('<b>已观察到：</b>冻结视觉时，0/2/6 层融合的最终成功率为 84%、84%、79%；训练视觉时为 92%、85%、96%。<b>可能机制：</b>融合模块需要视觉特征与控制目标共同适配；视觉固定时，额外容量可能主要改善演示拟合，而未改善闭环决策所需的空间信息。优化敏感性和有限样本波动也是替代解释；不能得出“增加深度必然提升性能”的结论。')
h2('10.3　早期表现弱和后期曲线波动的可能原因')
p('<b>已观察到：</b>2k 时仅 2%—26%，4k 明显提升，后期部分检查点又出现回落。<b>可能机制：</b>早期训练不足与 EMA 滞后共同影响结果。固定衰减 0.999 时，EMA 半衰期约 693 次更新；经过 2,000 次成功更新后，初始化参数副本仍约占 13.5% 权重。这说明滞后具有机制上的可能性，但当前没有 raw/EMA 对比评估，无法隔离其贡献。后期回落还可能来自策略敏感性、演示覆盖不足或小样本噪声，不能仅凭曲线认定过拟合。［S3］')
h2('10.4　因果解释和泛化结论的边界')
p('每组只有一个记录中的训练种子，各检查点重复使用同样 100 个初始条件；回合按任务聚类，回顾性多重比较存在选择偏差。V1/V1.1 的六组旧评估缺完整元数据；V2.1 微批次不同且有三段训练日志，历史输出路径也变化。实验仅覆盖同一 10 任务套件，不能证明未见指令泛化或跨套件、跨 MT50 迁移。文档提到的历史验证集划分变更缺少可匹配归档对照，故本报告不为其赋予数值收益。')
h2('10.5　建议的后续验证')
p('首先，对“视觉是否训练×0/2/6 层融合”的 2×3 设计进行至少三个独立训练种子重复，并统一微批次、预先规定最终模型选择规则。其次，扩大初始条件覆盖并采用考虑任务层次的不确定性估计。第三，在 2k/4k 对比 raw 与 EMA，检验滞后假设。第四，固定同一训练策略，比较执行 1/4/8/12 步后的重新规划效果。最后，用视频标注失败类型，再单独评估八层 V4 与其他套件。这些均为待开展实验，不计作已完成结果。')

# 第十一页
page();h('11　结论与数据可追溯性')
p('当前记录支持优先研究视觉编码适配，并表明其效果受融合深度影响。统一 10k 预算下，V2.2 是观测最优模型，在 100 个 LIBERO-Spatial 回合中达到 96%：比六层冻结组高 17 点，比无融合冻结基线高 12 点，比无融合训练视觉的 V2 高 4 点。最后这一差距相对样本规模较小，尚不能认定为稳健优势。训练损失可用于监控优化，但比较控制性能仍需闭环评估。')
p('新增 GPU 实测中，V2.2 的动作块预测中位耗时为 25.21 ms，P95 为 26.19 ms。该测量包含完整文本编码，排除预处理、传输和仿真，不能替代实际控制链路时延。本报告对已有训练与闭环数据重新统计，并新增推理微基准；没有重新训练或重新运行闭环仿真。')
h2('11.1　项目证据索引')
table([
 ['编号','相对项目根目录的证据','报告用途'],
 ['S1','outputs/eval/libero_spatial_v*/<br/>ckpt{step}/seed42/ 下的<br/>progress.json、log.json、episodes.jsonl','30 组评估；总体和逐任务结果；初始条件配对；元数据覆盖'],
 ['S2','outputs/train/libero_spatial_v*/ 下的<br/>checkpoint_{step}.pt、manifest.json、<br/>metrics.jsonl 或 log.jsonl','模型结构、实际训练设置、演示规模和损失 / 时间 / 显存日志'],
 ['S3','src/myvla/ 下的 model.py、data.py、<br/>preprocess.py、train.py、evaluate.py、ema.py','模型前向、采样、损失、动作解码、闭环协议与 EMA'],
 ['S4','data/libero_spatial/*_demo.hdf5；<br/>pyproject.toml；当前软件环境清单','图像样例结构、依赖规格和当前环境版本'],
 ['S5','configs/train_cfg_v4.yaml；README.md；<br/>项目模型版本记录文档','未评估八层配置与历史说明；不替代原始结果'],
 ['S6','scripts/measure_inference_time.py；<br/>outputs/benchmarks/inference_20261111/','18 份有效计时 JSON、测量清单、summary.json 与 summary.csv'],
 ],[.45,3.8,2.7],True)
cap('表 9　v* 对应 v1、v1_1、v1_2、v2、v2_1、v2_2；step 对应 2000、4000、6000、8000、10000。报告版本用小数点，文件路径用下划线。')
h2('11.2　复现信息与尚未补全的记录')
p('CLIP 固定修订号为 57c216476eefef5ab752ec549e440a49ae4ae5f3。历史云端路径中的“两层可训练视觉 v4”对应当前 V2.1，并非当前八层 V4。六组清单的轨迹身份与长度一致，但路径和时间戳不等同于所有 HDF5 的加密哈希。历史训练 GPU / 驱动与 LIBERO Git 修订号仍缺记录。所有统计与新增测量截止 2026 年 11 月 11 日。','small')
h2('参考文献')
p('［1］Radford A, 等. Learning Transferable Visual Models From Natural Language Supervision. ICML, PMLR 139:8748—8763, 2021. <link href="https://proceedings.mlr.press/v139/radford21a.html" color="#2f7377">proceedings.mlr.press/v139/radford21a.html</link>。','small')
p('［2］Liu B, 等. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. 2023. <link href="https://arxiv.org/abs/2306.03310" color="#2f7377">arXiv:2306.03310</link>。文献用于说明主干与基准背景，本报告未与其公开基线成绩作直接比较。','small')

def decorate(c,doc):
 c.saveState();c.setFont('Hei',8);c.setFillColor(GRAY)
 c.drawString(M,H-28,'myVLA 实验研究｜LIBERO-Spatial 视觉—语言—动作策略')
 c.setStrokeColor(colors.HexColor('#c4cdd1'));c.setLineWidth(.4);c.line(M,H-36,W-M,H-36);c.line(M,36,W-M,36)
 c.drawString(M,22,'数据与实测截止：2026 年 11 月 11 日');c.drawRightString(W-M,22,f'{doc.page} / 11');c.restoreState()
OUT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=49,bottomMargin=49,title='myVLA 实验研究报告：视觉编码适配与多模态融合',author='myVLA 项目',subject='LIBERO-Spatial 实验汇总、模型架构与推理时间实测',pageCompression=1)
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
print(OUT)
