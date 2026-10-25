# 复现与排错

记录日期：2026-10-24。以下PowerShell命令从仓库根目录运行。

## 环境与冻结演示

```powershell
py -3.13 -m venv .venv-bc
& '.\.venv-bc\Scripts\python.exe' -m pip install -r '.\robotic-arm-pick-and-place\requirements-bc-cpu.txt'
.\reproduce_learning.ps1 -VerifyOnly
& '.\.venv-bc\Scripts\python.exe' -X utf8 -B '.\robotic-arm-pick-and-place\show_frozen_policy.py' --model aggregated --seed 3001 --viewer
```

冻结模型只读取便携权重与代码，不要求原始datasets/和runs/。去掉 --viewer 可无窗口运行；--model bc 展示原基线。窗口演示包括固定起点准备阶段，准备完成后才进入学习策略控制。

## 完整学习流程

```powershell
.\reproduce_learning.ps1
```

解释器默认依次寻找仓库内、仓库上一级的 .venv-bc。也可显式指定：

```powershell
.\reproduce_learning.ps1 -Python 'C:\path\to\python.exe'
```

脚本串联六个已验证入口，保留每步返回的实际目录：

| 步骤 | 入口 | 正式设置 |
|---|---|---|
| 采集专家轨迹 | collect_reach_grasp_dataset.py | 40条，种子300–339，32/8划分 |
| 训练原BC | train_reach_grasp_bc.py | 100轮，随机种子0，CPU |
| 原BC诊断评估 | evaluate_reach_grasp_bc.py | 种子2000–2019 |
| 策略状态专家补标 | collect_reach_grasp_dagger.py | 40条，种子400–439，32/8划分 |
| 合并数据微调 | train_reach_grasp_dagger.py | 100轮，学习率0.0005 |
| 新位置配对评估 | evaluate_reach_grasp_dagger.py | 两模型各测3000–3019 |

每步失败会停止，并保留已有输出。最后的配对评估目录包含 reproduction.json，可沿其中路径追踪整轮产物。数据路径可含中文，入口会在载入XML前切换到Python项目目录。

从头重训会产生新的模型文件、路径和哈希；不要把新文件哈希当作2026-10-24留存证据的原始哈希。统计结果应按新运行记录如实报告。

## 流程短验

```powershell
.\reproduce_learning.ps1 -Smoke
```

短验用8条专家轨迹、20轮初始BC、8条补标轨迹、2轮微调和2对评估，种子为9000、9100、9200、9300起始的独立区段。它验证读写、补标、参数更新和目录衔接；极小模型可能退步，不能据此声称达到正式结果。

若初始模型从不闭合，则补标数为0，采集器保留记录但不创建训练划分；流水线会停止。这是有效的数据检查，不应伪造标签或把失败轨迹删除后继续。

## 视频导出

只导出视频时安装可选依赖：

```powershell
& '.\.venv-bc\Scripts\python.exe' -m pip install -r '.\robotic-arm-pick-and-place\requirements-video.txt'
```

下例在完成正式流程后，选择最新一轮包含种子3001的20对评估。短验数据不会被选中：

```powershell
$run = Get-ChildItem -LiteralPath '.\robotic-arm-pick-and-place\runs\bc' -Directory -Filter 'reach_grasp_paired_eval_*' | Where-Object {
    $plan = Get-Content -LiteralPath (Join-Path $_.FullName 'plan.json') -Raw | ConvertFrom-Json
    $plan.planned_pairs -eq 20 -and $plan.seeds -contains 3001
} | Sort-Object Name -Descending | Select-Object -First 1
& '.\.venv-bc\Scripts\python.exe' -X utf8 -B '.\robotic-arm-pick-and-place\record_learning_comparison.py' --evaluation $run.FullName --seed 3001 --output 'media/reproduced_comparison.mp4'
```

输出文件已存在时，指定新的 --output 名称。视频按保存的动作重放，逐步核对观测，渲染使用独立MjData，附带JSON校验信息。8秒任务输出201帧、25fps，编码时长8.04秒；没有追加模拟保持时间。默认视频选择一个原BC失败而改进模型成功的已测种子，整体结论仍以全部配对结果为准。

## 常见问题

- Python找不到脚本：检查是否从仓库根目录执行，或使用完整引用路径；不要手输带提示符的命令。
- checkpoint或task source hash mismatch：源代码、模型或数据版本不一致。使用匹配版本；不要删掉校验来使旧结果通过。
- 不同数据格式混用：25→8、6→2、10→4及补标数据采用不同schema。
- 图形窗口或导出失败：先运行无窗口冻结演示确认物理/推理正常，再检查OpenGL驱动。Windows默认不设置MUJOCO_GL=osmesa。
- 原基线的运行失败：失败记录本身是对照数据，不应覆盖或删除。
