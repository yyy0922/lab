"""Publication-style vector architecture figure. Does not run experiments."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MPath

OUT=Path(__file__).resolve().parent
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyhbd.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(27,16))
fig.subplots_adjust(0,0,1,1)
ax.set_xlim(0,2700); ax.set_ylim(1600,0); ax.axis('off')
INK='#263445'; GRAY='#657286'; BORDER='#C5CED8'; BLUE='#EAF2FA'; TEAL='#EAF5F1'; PURPLE='#F0EBF7'; GOLD='#FCF3DF'

def text(x,y,s,size=24,color=INK,bold=False,ha='center',va='center'):
    return ax.text(x,y,s,fontsize=size*.72,color=color,weight='bold' if bold else 'normal',ha=ha,va=va,linespacing=1.5,zorder=5)

def box(x,y,w,h,title,body='',fill='white',edge=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=9',linewidth=1.1,edgecolor=edge,facecolor=fill,zorder=3))
    if body:
        text(x+w/2,y+30,title,25,bold=True)
        text(x+w/2,y+h*.65,body,22)
    else: text(x+w/2,y+h/2,title,25,bold=True)

def panel(x,y,w,h,title):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=14',linewidth=1,edgecolor=BORDER,facecolor='#FCFDFE',zorder=0))
    text(x+25,y+40,title,29,bold=True,ha='left')

def edge(points,color=INK,dashed=False,arrow=True):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>' if arrow else '-',mutation_scale=13,linewidth=1.25,linestyle=(0,(5,4)) if dashed else '-',color=color,zorder=2))

text(1350,55,'基于操作码语义分组门控的 Android 恶意软件检测',36,bold=True)
text(1350,100,'总体框架：方法级位置编码、长度条件融合与 APK 级监督',23,color=GRAY)
panel(40,140,480,1270,'(a) 方法级序列构造')
panel(560,140,1540,1270,'(b) 分组条件编码与多路融合')
panel(2140,140,520,1270,'(c) APK 级聚合与分类')

# (a) Data representation: no sequence crosses a method boundary.
box(120,235,320,80,'输入 APK',fill=BLUE)
box(120,355,320,90,'静态反编译','APKTool → smali',fill=BLUE)
box(120,485,320,90,'按方法提取指令','保持方法边界',fill=BLUE)
box(120,615,320,110,'七类操作码映射','M / R / G / I / T / P / V',fill=BLUE)
box(120,765,320,95,'方法内滑动 5-gram','步长 1；窗口作为位置',fill=BLUE)
box(120,900,320,120,'位置状态 X / G / mask','GA 词表特征 / 中心操作组\n真实位置与补齐位置分离',fill=BLUE)
box(120,1150,320,105,'输入投影','Linear + LayerNorm† → 64 维',fill=BLUE)
for y1,y2 in [(315,355),(445,485),(575,615),(725,765),(860,900),(1020,1150)]: edge([(280,y1),(280,y2)])
text(280,1070,'短方法（映射后长度 < 5）仅统计\n无有效方法的 APK 显式记录',19,color=GRAY)
text(280,1325,'N：当前批次的有效方法数\nT：补齐后位置数；L：有效位置数',20,color=GRAY)

# Common input fan-out. All paths consume the same position features.
edge([(440,1202),(630,1202),(630,560)],arrow=False)
for y in [560,850,1130]: edge([(630,y),(760,y)])
text(585,1260,'H：N × T × 64',24,ha='left')
text(585,1295,'附带 G、mask、L、apk_index',20,ha='left',color=GRAY)

# Group conditioning. One shared projection with three output slices is
# equivalent to three linear projections with a common input.
box(760,250,290,135,'操作语义组嵌入','调用 / 读写\n控制转移 / 移动 / PAD',fill=PURPLE,edge='#786198')
box(1200,250,350,135,'分组条件参数偏置','bΔ / bB / bC\n有界偏置；输出投影零初始化',fill=PURPLE,edge='#786198')
edge([(440,940),(535,940),(535,210),(905,210),(905,250)],color='#786198',dashed=True)
text(490,910,'G',24,color='#786198')
edge([(1050,318),(1200,318)],color='#786198',dashed=True)
edge([(1375,385),(1375,415),(1090,415),(1090,805),(1050,805)],color='#786198',dashed=True)
text(1120,730,'参数调制',20,color='#786198',ha='left')

box(760,500,290,120,'轻量短路径','1 个因果 TCN 块\nLinear → 64 维',fill=TEAL)
box(760,760,290,180,'分组条件 Bi-Mamba','基础 Δ / B / C + 分组偏置\n双向扫描；参数共享\nConcat + Linear → 64 维',fill=PURPLE,edge='#786198')
box(760,1070,290,120,'独立局部 TCN','3 个残差块；d = 1 / 2 / 4\n输出 64 维',fill=TEAL)

box(1200,440,350,105,'长度条件权重 α(L)','由有效位置数 L 决定（F2）',fill=GOLD)
edge([(1375,545),(1375,600)],color='#9D7F3C',dashed=True)
box(1200,600,350,200,'长度条件双分支融合','Hroute = α(L) Hshort\n+ [1 − α(L)] Hlong\n输出 N × T × 64',fill=GOLD)
edge([(1050,560),(1130,560),(1130,650),(1200,650)])
text(1130,538,'Hshort',21)
edge([(1050,875),(1130,875),(1130,750),(1200,750)])
text(1130,910,'Hlong',21)

box(1650,860,330,200,'Top-k 自注意力','拼接特征 → Linear\nQ / K / V（2 头）\n每个查询保留 ki 个有效键\nHattn：N × T × 64',fill=BLUE)
edge([(1550,700),(1605,700),(1605,925),(1650,925)])
edge([(1050,1130),(1605,1130),(1605,1000),(1650,1000)])
text(1340,1103,'Htcn',23)
text(1815,1090,'ki = max(1, ceil(ρLi))，0 < ρ ≤ 1',20,color=GRAY)
text(1815,1130,'先屏蔽 padding，再选择 Top-k',20,color=GRAY)
text(1370,1350,'F0：固定 α = 0.5；F1：全局可学习 α（仅用于消融对照）',21,color=GRAY)

# Three feature outputs enter concatenation at distinct ports.
box(2180,700,400,140,'三路特征拼接','Hroute ‖ Htcn ‖ Hattn\n每个位置 192 维',fill=BLUE)
edge([(1550,650),(2040,650),(2040,730),(2180,730)])
text(1835,623,'Hroute',23)
edge([(1980,965),(2070,965),(2070,770),(2180,770)])
text(2025,990,'Hattn',20)
edge([(1050,1170),(2100,1170),(2100,810),(2180,810)])
text(1815,1198,'Htcn',23)
box(2180,900,400,105,'方法级表示','Masked mean + LayerNorm†\nN × 192',fill=TEAL)
box(2180,1050,400,105,'APK 级表示','按 apk_index 对有效方法取均值\nBapk × 192',fill=TEAL)
box(2180,1200,400,90,'分类头','FC256 → ReLU → Dropout → FC2',fill=BLUE)
box(2180,1330,400,60,'良性 / 恶意',fill=BLUE)
for y1,y2 in [(840,900),(1005,1050),(1155,1200),(1290,1330)]: edge([(2380,y1),(2380,y2)])

# Targets supervise the APK loss only; no method-level targets are invented.
box(2220,250,320,85,'APK 标签 y',fill='#F3F4F6',edge=GRAY)
box(2220,420,320,110,'APK 级交叉熵损失','仅在训练时计算',fill='#F3F4F6',edge=GRAY)
edge([(2380,335),(2380,420)],color=GRAY,dashed=True)
edge([(2580,1245),(2635,1245),(2635,475),(2540,475)],color=GRAY,dashed=True)

# Concise figure notes, outside computational boxes.
edge([(80,1460),(160,1460)])
text(180,1460,'特征流',21,ha='left')
edge([(350,1460),(430,1460)],color='#786198',dashed=True)
text(450,1460,'条件 / 监督信号',21,ha='left')
text(90,1510,'† LayerNorm 为本图所示方案的归一化步骤；mask 同时用于有效位置计算、注意力与方法级池化。',21,color=GRAY,ha='left')
text(90,1550,'注：软融合执行两条路径；Top-k 为关联约束，不代表线性复杂度或恶意代码定位。GA 词表仅由训练集拟合。',21,color=GRAY,ha='left')

for extension in ['svg','pdf','png']:
    fig.savefig(OUT/f'SAS-Droid_framework.{extension}',dpi=200 if extension=='png' else 100,facecolor='white')
fig.savefig(OUT/'SAS-Droid_framework_preview.png',dpi=70,facecolor='white')
print(OUT/'SAS-Droid_framework.svg')
