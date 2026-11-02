# SAS-Droid：基于 LTA 的独立实验实现

Git仓库仅包含源码、文档、框架图与提取工具，不包含APK、方法缓存或模型权重。服务器克隆后请另行传输data/prepared.json和data/methods/，详见SERVER_RUN.md与data/README.md。

服务器运行请优先看 [SERVER_RUN.md](SERVER_RUN.md)。2026-11-02已补齐tools/apktool.jar和跨平台训练入口run_server.py，experiment.py支持--device cpu/cuda/auto。下面的Windows本地运行记录保留为历史说明；CUDA路径仍需服务器硬件实测。

数据默认读取 `D:/remake/apk`：benign=0，malradar=1；new 无确认标签，不参与监督训练。APK 只做静态解析。原 LTA 目录不改动。完整研究规格见 RESEARCH_SPEC.md，实现顺序见 IMPLEMENTATION_PLAN.md。

运行中发现原始 malradar 目录内容发生变化，因此本轮改为先运行 `python freeze_data.py`，复制当前 APK 到 data/apk_snapshot，并记录 data/snapshot_inventory.json；后续自动优先使用这份固定快照。最终样本数以快照和提取清单为准。原始目录不被本项目修改。反编译临时目录在成功保存方法缓存后清理，保留原始 APK、序列和代码位置。

## 已实现

- 原版 84 条 opcode 映射、方法边界及多 DEX；提取结果可断点续用。reference/ 保存原 map.py、GA/SVM 代码及 SHA256。
- 逐位置 5-gram：选中特征用 ID 查表，未选中为 ID 0，其向量为零；真实位置 mask 由长度生成，独立于特征 ID。中心 opcode 映射到调用、读写、控制转移、移动四组。
- 所有有效方法和位置均保留。短方法、不可表征 APK、解析失败分开统计。仅完全相同序列压缩存储并保留次数；训练时相同序列共享一次 dropout 抽样，因此压缩与逐副本独立 dropout 的随机训练过程不同，评估池化等价。
- CPU 可运行的 Mamba-1 显式选择性递推，双向共享参数、分组 Δ/B/C 偏置、TCN、长度条件软融合、按有效长度选 k 的稠密 Top-k 注意力。Top-k 不降低二次复杂度，软融合不跳过支路。
- 方法均值→APK均值→APK标签交叉熵。按方法小批前向池化，计算 APK 梯度后重算各小批并回传；重放随机状态，无需将整个 APK 的训练图放入内存。
- C0–C6、A/B/C/D、D_delta/D_bc、F0/F1/F2 可选。C0 为原 LTA 单时间步设计的 PyTorch 参考移植，非官方数值一致实现；注意力投影和 TCN 实现与 notebook 不完全相同，不能标为严格原代码复现。
- 固定分组划分，GA 仅在训练集上拟合；验证集用于早停、阈值和方案选择。测试集为独立显式命令。

## 环境与运行

本机 Python：`D:/Python312/python.exe`；已装 PyTorch 2.11.0+cpu；Java 位于 `C:/Program Files/Java/jdk-1.8/bin/java.exe`。

在本文件所在目录运行：

```powershell
python -m unittest test_pipeline.py
python -u run_local.py > pipeline.log 2>&1
```

快速真训练检查（与全量试跑的数据划分完全分开，不查看测试集）：

```powershell
python -u quick_check.py > quick_check.log 2>&1
```

此命令选8个已提取、特征指纹不同的较小APK，6个训练/2个验证；GA8×2，五个配置各训练3轮，仍保留每个APK全部方法和位置。输出 quick_results.json。小样本和按大小选样只适合验证训练流程，不能用来判断模型优劣；不会覆盖全量 prepared.json。

run_local 是一次性本地试跑：全部有标签 APK，GA 20×5，每个配置训练 1 轮（C0、C2、B、D、C6），64 维/状态16；只看验证集，不查看测试集。**1轮不是收敛实验，20×5 也不是方案建议的正式 GA 100×20**。每个运行名唯一，已有运行禁止覆盖。

本轮快照成功复制并读取240个APK；另有复制或读取失败项，逐项记录在 snapshot_inventory.json。提取阶段仍逐项记录后续失败；试跑 prepare 显式允许排除这些失败项，并将 excluded_failures 写入 prepared.json，不能把覆盖率算成100%。精确类别数量以快照清单和最终 manifest 为准。

当前运行进度见 pipeline_status.json，详细输出见 pipeline.log。后台是本次一次性计算任务，不是定时或循环自动任务。停止时可针对 pipeline.pid 中本项目的进程及其子进程操作，已保存的提取缓存仍可复用。

正式实验应在新的实验目录重新冻结配置。使用完整 GA 默认参数，并按验证结果选择预算：

```powershell
python experiment.py prepare --population 100 --generations 20
python experiment.py train --run C6_seed42 --variant C6 --seed 42 --epochs 200 --patience 20
python experiment.py train --run B_seed42 --variant B --seed 42 --epochs 200 --patience 20
python experiment.py train --run D_seed42 --variant D --seed 42 --epochs 200 --patience 20
```

prepared.json 已存在时 prepare 会拒绝覆盖。正式 GA 需新建实验目录并保留当前记录，不能把已有选择结果伪装成另一组配置。其他变体与种子43/44同理。正式模型冻结后再执行：

```powershell
python experiment.py test --run C6_seed42 --variant C6 --seed 42 --confirm-frozen
```

测试命令校验模型参数，复用验证工作点，并禁止重复覆盖测试结果。服务器可显式传入--device cuda；本机测试仍为CPU。

## 输出与结论边界

- data/manifest.json：全部 APK 身份、目录标签、提取状态、方法/位置数量、全局特征指纹；data/methods/ 保留方法序列和源码定位。
- data/prepared.json：固定划分、训练选出的 GA 特征、GA轨迹和预算。
- runs/运行名/：config.json、trace.csv、best.pt、best_validation.json；明确最终评估后产生 test.json 和预测。
- smoke_real.json：单 APK 全量前反向链路验证，不能当分类性能。

当前数据只有25个良性样本，无法可靠估计1% FPR；验证阈值按1%约束选取只是工作点规则。目录标签尚未独立复核；长度/来源捷径、近重复和跨时间泛化需要进一步实验。全局5gram指纹相同的APK被放在同一划分，但这不能检测所有近重复/重打包。数据此前已经参与过其他实验，不是完全未使用的新确认集。

本次实现提供可运行系统和试跑记录，不证明新方法优于 LTA，也不声称复现原论文的98.69%。正式效力结论还需要多种子、对齐初始化/预算、长度诊断、近重复审计、配对区间与效率基准；这些统计研究尚未全部自动化。
