# MuJoCo Panda 与 FR3：模仿学习及机械臂迁移实践

**第一阶段发布 · 2026-10-25 · 后续持续更新**

基于开源 MuJoCo 抓放项目，完成两条独立实践：在 Panda 上建立 CPU 行为克隆和专家补标流程；将规则式抓放系统迁移到 FR3 与双指夹爪组合模型。

受当前算力与时间限制，本阶段优先完成可复现的小规模仿真实验。Panda 学习任务止于接近、抓取、抬升和保持；FR3 完成规则式抓放迁移。后续计划扩展学习策略的移动、放置与更广泛鲁棒性评估，[具体范围与计划](docs/phase-one.md)。

## 已验证结果

| 方向 | 第一阶段成果 | 留存证据 |
|---|---|---|
| Panda CPU 模仿学习 | 20 对相同初始状态测试，原 BC 14/20，专家补标微调后 20/20；抬升后横向位移中位数 161.50 → 1.78 mm | [实验结果](panda-pick-place/docs/experiment-results.md)、[逐回合指标与冻结权重](panda-pick-place/experiments/2026-10-24/) |
| FR3 模型与控制适配 | 固定场景 3/3；种子 0～9，10 轮共 30/30 次抓放；XY 误差均值 0.4155 mm | [迁移说明](fr3-adaptation/docs/FR3_MIGRATION.md)、[逐轮记录](fr3-adaptation/experiments/2026-10-25/) |

上述结果均为特定设置下的有限仿真验证。Panda 使用仿真状态、固定起点与向下朝向，通过 IK 执行网络输出；FR3 使用规则控制，未启用重力补偿。两者都未进行真机或双臂验证。FR3 的抓放后原判据与 Panda 的末尾持续保持判据不同，不能直接横向比较成功率。

## 演示

[观看 Panda 同步对比视频](panda-pick-place/robotic-arm-pick-and-place/media/learning_comparison.mp4)

![Panda 同种子原 BC 与补标后策略对比](panda-pick-place/robotic-arm-pick-and-place/media/learning_comparison.png)

视频选取正式评估种子 3001，重放留存动作并逐步核对观测；整体结论使用全部 20 对测试。仓库里的上游 Panda GIF 和图表属于上游示例，不作为 FR3 的演示证据。

## 快速运行（Windows PowerShell）

以下命令从合并仓库根目录执行。两条流程使用独立环境，以保留各自验证过的 NumPy 版本。学习与推理均使用 CPU；图形窗口需要可用的 OpenGL 驱动。

### Panda 冻结策略

```powershell
py -3.13 -m venv .venv-bc
& '.\.venv-bc\Scripts\python.exe' -m pip install -r '.\panda-pick-place\robotic-arm-pick-and-place\requirements-bc-cpu.txt'
& '.\.venv-bc\Scripts\python.exe' -X utf8 -B '.\panda-pick-place\robotic-arm-pick-and-place\show_frozen_policy.py' --model aggregated --seed 3001 --viewer
```

去掉 `--viewer` 无窗口运行；将 `aggregated` 改为 `bc` 查看原模型。同一示例中原模型失败是留存的对照结果。便携权重已包含，无需原始训练目录。

### FR3 随机抓放

```powershell
py -3.13 -m venv .venv-fr3
& '.\.venv-fr3\Scripts\python.exe' -m pip install -r '.\fr3-adaptation\requirements-migration.txt'
Set-Location '.\fr3-adaptation\robotic-arm-pick-and-place'
& '..\..\.venv-fr3\Scripts\python.exe' -X utf8 -B '.\run_fr3_random.py' --seed 0 --headless
```

去掉 `--headless` 可观察窗口；同目录 `run_fr3_batch.py` 重新运行种子 0～9。新结果写入独立时间戳目录。

## 代码与复现

- [Panda 学习流程](panda-pick-place/README.md)：状态与动作接口、专家数据采集、训练、补标、配对评估、视频导出。
- [FR3 迁移流程](fr3-adaptation/README.md)：模型组合、抓取点、控制站点适配和逐轮抓放记录。
- [阶段状态与后续工作](docs/phase-one.md)。
- [第三方来源和许可](THIRD_PARTY_NOTICES.md)。

原始训练集、临时运行和虚拟环境不随仓库发布；完整采集与训练入口、两份便携权重、正式逐回合指标和迁移记录均保留。冻结学习源文件保持原字节，`.gitattributes` 防止换行转换导致哈希校验失配。

## 来源与贡献

上游为 [RafayDon/Mujoco-Robotic-Arm-Pick-and-Place](https://github.com/RafayDon/Mujoco-Robotic-Arm-Pick-and-Place)。模型来源为 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。上游已提供 Panda 场景、DLS IK、规则抓放、随机化和评价功能。

本阶段新增 CPU 学习与专家补标流水线、冻结对照与证据导出，以及 FR3/夹爪模型装配和接口适配。未将上游算法与结果认定为个人原创，也未将这次针对性补标称为全部 DAgger 变体的完整复现。
