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
 'title':ParagraphStyle('title',fontName='Hei',fontSize=21,leading=28,alignment=TA_CENTER,spaceAfter=12,textColor=INK),
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

# 第 1 页：设置与简洁结构
p('myVLA 实验报告','title')
p('LIBERO-Spatial：视觉适配、融合深度与推理效率｜2026 年 11 月 11 日','subtitle')
p('<b>摘要：</b>汇总六组模型、五个训练阶段的 30 组闭环评估，并补充 GPU 推理实测。10k 步时 V2.2 达到 96% 成功率，预测完整 12 步动作块的中位耗时为 25.21 ms。视觉适配与融合深度存在交互，训练损失不能直接替代闭环评价。','abstract')
story.append(Image(str(ROOT/'assets/v1_2.png'),width=CW,height=CW*1315/3811))
cap('图 1　简洁 VLA 框图，采用工程现有 V1.2 示意图。V2.2 同样使用六层双向融合和四层动作解码器，区别是其视觉编码器参与训练；文本编码器均冻结。状态经 MLP 直接进入多模态 memory，输出为 12×7 动作块。')
h('1　实验设置')
table([
 ['项目','设置'],
 ['数据与评估','10 个任务，500 条演示、62,250 帧，全部用于训练；每检查点 10 任务×10 回合，共 30 组、3,000 回合；相同初始状态索引，seed=42。'],
 ['训练与模型','CLIP ViT-B/16；224×224 双相机 RGB、语言、8 维状态。隐藏维度 256，动作解码器 4 层；AdamW，动作头 LR=1e-4、视觉 LR=1e-5，预热 500 步，训练 10k 步；FP16、EMA=0.999，有效 batch=64。'],
 ['软件与硬件','PyTorch 2.6.0、Transformers 4.57.6、NumPy 1.26.4；robosuite 1.4.1、MuJoCo 2.3.7。当前测量机为 i9-12900H、RTX 3060 Laptop GPU（6 GiB），CUDA 构建版本 12.4；历史训练 GPU 未记录。'],
 ],[.9,5.5],True)
rows=[['模型','V1','V1.1','V1.2','V2','V2.1','V2.2'],['视觉训练','否','否','否','是','是','是'],['融合层数',0,2,6,0,2,6]]
table(rows,[1.2,1,1,1,1,1,1],True)
cap('表 1　六组实验设计。文本始终冻结；V2.1 实际微批次为 8×8 累积，其余为 16×4。当前八层 V4 和 MetaWorld MT50 尚无完成结果，不参与比较。')

# 第 2 页：整体结果
page();h('2　闭环成功率与训练阶段比较')
rows=[['模型','2k 步','4k 步','6k 步','8k 步','10k 步']]
for v in vs:rows.append([names[v],*[f"{r['rate']}%" for r in a['runs'] if r['version']==v]])
table(rows,[1,1,1,1,1,1])
cap('表 2　每格为 100 回合的成功率，均由原始 progress、汇总及回合日志交叉核验。主要比较统一使用 10k 步，不事后为每组挑选不同预算。')
fig('zh_success_curves.png',194)
cap('图 2　成功率随训练步数的变化。左侧冻结视觉，右侧训练视觉；同色对应相同融合深度。')
h2('2.1　主要结果')
p('训练视觉编码器后，无融合、两层、六层模型的 10k 成功率分别从 84%→92%、84%→85%、79%→96%，提高 8、1、17 个百分点。V2.2 最高，但融合深度并非越大越好：冻结组最终为 84%、84%、79%，训练视觉组为 92%、85%、96%。')
p('V2 在 4k 已达到 91%，而 V2.2 从 6k 起领先。后期表现并不单调：V1 从 6k 的 89% 回落至 10k 的 84%，V2.1 从 88% 回落至 85%。因此，训练更久或损失更低均不保证任务成功率提高。')
h2('2.2　差异有多可靠')
p('最终检查点中，V2.2 相对 V1.2 有 19 个配对条件改善、2 个退化（净增 17 点；精确 McNemar 名义 p=0.00022）。相对 V2 则为 7 个改善、3 个退化（+4 点；p=0.344），不足以确立稳健优势。V2.2 的描述性 Wilson 95% 区间为 90.2%—98.4%。检验未校正多重比较，也未考虑任务聚类与训练种子变异。')

# 第 3 页：计算效率与任务结果
page();h('3　推理效率、拟合损失与逐任务结果')
rows=[['模型','末段损失','峰值显存/GiB','推理中位数/ms','推理 P95/ms']]
for v,t in a['train'].items():
 g=b['groups'][v];rows.append([names[v],f"{t['last20_mean_loss']:.4f}",f"{t['peak_vram_gib']:.2f}",f"{g['median_ms']:.2f}",f"{g['p95_ms']:.2f}"])
table(rows,[.8,1.1,1.3,1.4,1.2])
cap('表 3　末段损失取最后 20 个唯一日志观测均值，显存为训练日志中的峰值已分配量。推理采用当前 GPU 上的 FP32、EMA、batch=1；每组 3 轮，每轮预热 50 次、计时 200 次，共 600 个有效计时。')
h2('3.1　推理时间的含义')
p('计时脚本为 scripts/measure_inference_time.py，使用相同的 21-token 指令、驻留 GPU 的合成观察，并在前后同步 CUDA。包含视觉编码、完整文本编码、动作头和设备端动作解码；排除预处理、分词、输入传输、CPU 输出复制和仿真。文本缓存被绕过，未使用编译或 CUDA Graph，因此不是端到端闭环耗时。')
p('V2.2 为 25.21 ms/动作块，P95=26.19 ms，约 39.7 次预测/秒；相比 V2 增加 2.57 ms（11.4%）。除以 12 得到的 2.10 ms/生成动作只是摊销值，不是单步反应延迟。训练时冻结视觉并不省略推理前向，同结构模型的小差异应结合温度、频率及系统负载波动解释。')
fig('zh_task_heatmap.png',174)
cap('图 3　10k 检查点逐任务成功率，每任务仅 10 回合。T0—T9 的黑碗初始位置依次为：盘子与小碗间、小碗旁、桌面中央、饼干盒上、木柜上层抽屉内、小碗上、饼干盒旁、炉灶上、盘子旁、木柜上。任务均要求将黑碗放到盘子上；小碗对应 ramekin。')
p('V2.2 在六个任务上达到 10/10，在 T1、T2、T5、T7 上各为 9/10。其相对 V1 的明显增益出现在木柜上（60%→100%）和盘子旁（70%→100%）；桌面中央则从 100% 降至 90%。日志没有标注具体失败机制，不能据此认定为抓取滑落或碰撞。','small')

# 第 4 页：解释与结论
page();h('4　结果解释与结论')
h2('4.1　视觉适配与融合深度的作用')
p('视觉联合训练将末段损失从约 0.155—0.171 降至 0.029—0.030，并在六层融合组产生最大的成功率提升。合理解释是：动作监督使 CLIP 特征更适合空间几何与控制需求，融合模块也能与视觉特征共同适配。但当前没有特征探针或定位测量，该机制属于待验证假设。')
p('冻结视觉时，加深融合没有带来最终成功率提升；额外容量可能主要改善演示拟合，而未改善闭环决策。V2.1 的损失最低（0.0290）却只有 85% 成功率，V2.2 损失略高（0.0299）却达到 96%，说明不能仅凭训练损失选模型。训练损失来自演示状态和原始参数，闭环结果来自策略实际轨迹与 EMA，两者测量对象不同。')
h2('4.2　训练早期偏弱与后期波动')
p('2k 阶段仅为 2%—26%，可能与训练不足及 EMA 滞后有关。衰减 0.999 时，EMA 半衰期约 693 次更新，2,000 次更新后初始化副本仍约占 13.5%。但缺少 raw/EMA 对照，不能隔离其贡献。后期回落也可能来自策略敏感性、演示覆盖不足和有限回合噪声，不能单凭曲线断定过拟合。')
h2('4.3　结论的适用范围')
p('每组仅一个记录中的训练种子；30 组结果复用了相同 100 个初始条件，不是 3,000 个独立条件。V1/V1.1 的 6k、8k、10k 共六组旧评估缺完整元数据；其结果与索引可核验，但历史权重选择与全部协议不能独立复原。V2.1 微批次及分段续训也与其他组不同。实验仅覆盖同一 10 任务套件，不能证明未见任务或跨数据集泛化。')
p('<b>结论：</b>在现有记录中，V2.2 以 96% 成功率和 25.21 ms 中位动作块预测耗时取得最好观测表现；相对冻结六层组的 +17 点证据较突出，相对 V2 的 +4 点仍需更多重复验证。下一步优先统一微批次，开展多训练种子与更多初始条件评估，并比较 raw/EMA 和不同动作执行长度。')
h2('数据来源与复现')
p('闭环结果：outputs/eval/libero_spatial_v*/ckpt{step}/seed42/ 下的 progress.json、log.json、episodes.jsonl；训练结构与设置：outputs/train/ 下的检查点、清单和训练日志；推理原始记录与汇总：outputs/benchmarks/inference_20261111/。模型按检查点结构识别，历史路径中的两层 v4 对应当前 V2.1，而非八层 V4。本文未重新训练或运行闭环仿真。','small')
h2('参考文献')
p('［1］Radford A, 等. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021. <link href="https://proceedings.mlr.press/v139/radford21a.html" color="#2f7377">PMLR 139:8748—8763</link>。<br/>［2］Liu B, 等. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. 2023. <link href="https://arxiv.org/abs/2306.03310" color="#2f7377">arXiv:2306.03310</link>。','small')

def decorate(c,doc):
 c.saveState();c.setFont('Hei',8);c.setFillColor(GRAY)
 c.drawString(M,H-28,'myVLA 实验报告｜精简版')
 c.setStrokeColor(colors.HexColor('#c4cdd1'));c.setLineWidth(.4);c.line(M,H-36,W-M,H-36);c.line(M,36,W-M,36)
 c.drawString(M,22,'数据与实测截止：2026 年 11 月 11 日');c.drawRightString(W-M,22,f'{doc.page} / 4');c.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=49,bottomMargin=49,title='myVLA 实验报告（精简版）',author='myVLA 项目',subject='LIBERO-Spatial 实验设置、结果分析与推理实测',pageCompression=1)
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
print(OUT)
