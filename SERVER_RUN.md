# 服务器运行说明

将整个 SAS-Droid 文件夹上传到服务器即可。以下命令在其根目录执行。无需同时上传 LTA-main。

若通过GitHub获取项目，仓库仅包含源码、文档和工具，**不包含APK、data/prepared.json、data/methods缓存或模型权重**。克隆后必须从本地另行传输下列训练数据，之后再运行自检：

```bash
git clone https://github.com/yyy0922/lab.git
cd lab
# 另行把本地 SAS-Droid/data/prepared.json 和 data/methods/ 传到这里的 data/ 下
```

上传整个本地文件夹与只git clone仓库不是同一回事。前者可包含本地缓存；后者需要上述额外数据传输。

## 必须保留的文件

- 根目录 Python 源码、requirements.txt、test_pipeline.py。
- reference/：已冻结的原始映射与 GA/SVM 源码。
- data/prepared.json：213 APK 的固定划分及6702个GA特征。
- data/methods/：按APK哈希命名的完整方法序列缓存。
- tools/apktool.jar：仅重新解析 APK 时需要；已补齐到本目录。

缓存训练不读取原始APK，也不重新运行Java、GA。prepared.json里的Windows原始路径只是来源记录，不是训练文件读取路径；实际缓存根据当前项目目录和APK哈希定位。

apk/ 与 data/apk_snapshot/ 是原始数据副本，不需要两个都上传来训练。figures/、runs/、日志、__pycache__/均不是从头训练必需项。不要删除唯一需要保留的APK原始副本；这里只说明上传范围，不执行删除。

## 环境和检查

建议独立Python3.12环境。GPU服务器先按驱动配置安装CUDA可用的PyTorch，再安装依赖；不要把本机Windows CPU环境整体复制到Linux。

```bash
python -m pip install -r requirements.txt
python server_preflight.py --device cuda
python -m unittest test_pipeline.py
```

没有GPU时将 cuda 改为 cpu。auto 会在CUDA不可用时自动回退CPU；若不希望意外跑CPU，务必显式用 --device cuda。

## 继续此前未完成的实验

```bash
mkdir -p logs
nohup python -u run_server.py --device cuda --variants C2 B D C6 --epochs 1 --prefix server_pilot > logs/server_pilot.log 2>&1 &
echo $! > logs/server_pilot.pid
tail -f logs/server_pilot.log
```

这延续原来的“一轮全量试跑”预算，不是200轮正式收敛实验。C0本地已完成；需要同服务器测耗时时可以将C0加入variants。新运行保存到runs/server_pilot_配置_seed42，不覆盖本地旧结果。C2此前未保存中途权重，需要从头训练该组，不能从第60个APK继续。

先只验证完整模型一个配置，也可以运行：

```bash
python -u experiment.py train --run server_check_C6 --variant C6 --epochs 1 --device cuda
```

如显存不足，可以降低 --tokens 512：只减小方法微批的补齐位置预算，不删除方法、不截断位置；极长的单个方法仍可能消耗较大内存。不同微批预算会改变训练时dropout随机过程，因此比较时应统一预算。torch是显式并行扫描实现，不调用官方融合Mamba CUDA内核；不能保证与官方实现速度相同。

## 重新提取（只有更换输入数据时需要）

安装Java，并将APK按apk/benign和apk/malradar放好；不要调用写死Windows路径的旧run_local.py。现有prepared.json被冻结，重新prepare会拒绝覆盖；新数据请用新的实验目录。

```bash
python -c "from data import extract; extract('apk', 'java', workers=2)"
python experiment.py prepare --population 100 --generations 20
```

旧run_local.py、resume_local.py仅保留本地历史流程；服务器请用run_server.py。freeze_data.py也只用于本地历史数据快照。

## 结果与限制

查看runs/运行名/config.json、trace.csv、best_validation.json和best.pt。训练日志不会把测试集用于选择；不要在未冻结模型前执行test命令。

本机只能验证CPU路径；新增CUDA路径需在服务器完成上面的检查和首次试跑后才能确认。归一化结构维持原训练实现，没有为匹配框架图新增LayerNorm。当前数据良性样本少、类别失衡，单轮结果不能证明主贡献。服务器加速不会消除这些研究设计限制。
