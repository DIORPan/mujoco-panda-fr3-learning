# FR3 抓放迁移说明

核对日期：2026-10-25。范围为 FR3 仿真抓放迁移；重力补偿方向已从当前任务中移除。

## 来源与个人贡献

| 部分 | 来源或本次改动 |
|---|---|
| 原 Panda 项目、任务环境、DLS IK、平滑位置目标、抓放状态流程、原成功判据 | RafayDon/Mujoco-Robotic-Arm-Pick-and-Place |
| 随机位置采样、评价、图表及录像能力 | 上游已有；本次不将其认定为个人新增算法或完整评价系统 |
| FR3 模型、Panda 双指夹爪及其网格 | Google DeepMind MuJoCo Menagerie，固定提交见来源清单 |
| FR3 + 夹爪的运行时装配 | 本次使用 MjSpec.attach 组合，处理安装朝向、名称前缀和抓取参考点 |
| 控制器站点接口 | 本次将硬编码 grasp_site 改为可选 site_name，并在名称不存在时明确报错 |
| FR3 场景及实验入口 | 本次创建场景查看、单块抓放、固定三块、指定种子及批量运行脚本，保存逐轮 JSON 和日志 |

代码上游：`0700bff871b28878edca8170754bb08d644fc0ea`。
迁移起点：`defc09ce15e32149c2b0425e03967a90014eda5b`。
上述起点已有可见 grasp_site、IK 日志及 Panda 诊断脚本，属于此前学习基线。

旧 `fr3-pick-place` 仓库未用作本次模型资产来源；Panda 的 BC 工作副本和训练环境独立保留。

## 模型与场景组合

实际代码位于 `robotic-arm-pick-and-place/`：

1. `fr3_model.py` 读取 `assets/franka_fr3/fr3.xml` 和现有 `assets/franka_emika_panda/hand.xml`。
2. 将独立 hand 的朝向设为 `[0.9238795, 0, 0, -0.3826834]`，沿用此前 Panda 内嵌 hand 绕局部 Z 轴 -45° 的安装约定。
3. 在 hand 本体上添加局部位置 `[0, 0, 0.1034]` m 的抓取参考点。该位置沿用 Panda 名义抓取点。
4. hand 挂到 FR3 的 `attachment_site`，为夹爪元素加 `panda_` 前缀。最终抓取点名为 `panda_grasp_site`。
5. 可选接入 `task_environment.xml`。该文件复制自上游 scene.xml，删除 Panda include，保留桌子、方块、目标点和环境设置。
6. 机器人先组装，环境后组装，确保现有控制器的机械臂索引有效。环境使用空前缀，保留 cube_green、target_green 等任务名称。

| 模型 | nq | nv | nu |
|---|---:|---:|---:|
| FR3 裸臂 | 7 | 7 | 7 |
| FR3 + 双指夹爪 | 9 | 9 | 8 |
| FR3 + 夹爪 + 三块任务场景 | 30 | 27 | 8 |

当前模型前 7 个关节 / qpos / dof 均为 FR3 机械臂，其后为两个手指，再后为三个自由方块。控制器仍依赖这一排列，尚未泛化为任意关节顺序。

原始 FR3 XML、hand XML 和许可证未做装配性编辑；组合发生在 Python 内存模型中。正常重力为 `[0, 0, -9.81]` m/s²，未启用重力补偿。保留 Menagerie FR3 的伺服、摩擦等参数；IK 正则项仍为 `0.04`。

## 本机复现

验证版本：Python 3.13.0、MuJoCo 3.16.0、NumPy 2.5.4；Windows CPU 仿真。发布版使用独立 `.venv-fr3`；安装说明见本目录 README。

从仓库根目录进入项目：

```powershell
Set-Location .\robotic-arm-pick-and-place
```

| 目的 | 脚本 |
|---|---|
| 裸臂查看 | view_fr3.py |
| 独立夹爪开合 | view_hand.py |
| 组合模型及抓取点查看 | view_fr3_hand.py |
| 完整任务场景查看 | view_fr3_scene.py |
| IK → FK 几何验证 | inspect_fr3_ik.py |
| 固定绿色方块单次抓放 | run_fr3_green.py |
| 固定三块场景一轮 | run_fr3_fixed.py |
| 指定随机种子 | run_fr3_random.py |
| 种子 0～9 批次 | run_fr3_batch.py |

例如：

```powershell
& '..\..\.venv-fr3\Scripts\python.exe' -B .\run_fr3_random.py --seed 0 --headless --output-dir runs/fr3_migration/reproduction
```

有窗口观察时去掉 `--headless`，等结果输出后再关闭窗口。批次入口使用独立子进程，每个种子重新创建模型、状态和控制器，日志以 UTF-8 保存。

源码和 XML 都使用项目相对路径。中文绝对 XML 路径在本机曾出现加载问题，因此命令需从实际 Python 项目目录执行。

历史 `inspect_fr3_motion.py` 含无补偿与零重力诊断入口，保留作过程记录，不作为当前 FR3 迁移成果的扩展方向。

## 已验证证据

### 几何与固定场景

用户运行的 IK/FK 检查：目标 `[0.5, 0, 0.4]` m，位置误差约 `0.5637102` mm，现场关节最大变化为 0。几何求解与实际运动分开测量。

固定三块场景一轮记录：

[fixed_1792891269902468100.json](../experiments/2026-10-25/fixed_1792891269902468100.json)

| 方块 | XY 误差（mm） | 原判据 |
|---|---:|---|
| 绿 | 0.136211 | 通过 |
| 红 | 0.472529 | 通过 |
| 蓝 | 0.628377 | 通过 |

用户还分别完成了单块绿色抓放、随机 seed=0 的有窗口运行与无窗口一致性检查。这些单轮数据没有重复计入下述批次。

### 固定种子的随机批次

[批次汇总 summary.json](../experiments/2026-10-25/batch_1792914315621804400/summary.json)

- 种子：0～9，10 个不同种子，每轮按绿 → 红 → 蓝执行。
- 每轮初始化：原控制器 randomize=True；X 坐标在 [0.38, 0.55] m、Y 坐标在 [-0.22, 0.22] m 均匀采样，方块初始中心 z=0.25 m，随后落稳 1 秒。
- 每次抓放结束后读取方块位置，沿用 XY 距离 <0.05 m 且中心 z>0.18 m 的原判据。
- 10 份逐轮 JSON、10 份 run.log 与汇总逐一核对，种子、完成状态、计数和重力配置一致。
- 结果：30/30 次方块抓放，10/10 个完整回合通过。

| 范围 | 抓放次数 | 平均 XY 误差（mm） | 最大 XY 误差（mm） |
|---|---:|---:|---:|
| 绿 | 10 | 0.206026 | 0.439064 |
| 红 | 10 | 0.393456 | 0.454782 |
| 蓝 | 10 | 0.647008 | 0.714171 |
| 全部 | 30 | 0.415497 | 0.714171 |

这些误差是方块 XY 到目标的距离，不是夹爪 IK 误差；两个指标不可互换。每个 JSON 还包含落稳后的初始位置、最终位置和方块中心高度。

## 结果边界

- 这是上述初始化范围、种子、顺序与原判据下的有限仿真验证，不能由 30/30 推断总体成功率为 100%。
- 同一回合内三次抓放共享场景，不应当作 30 个独立场景样本。
- 判定发生在每次抓放后；没有额外持续稳定性、速度阈值或回合末重新检验所有已放置方块的机制。
- 未实现新的 IK 或学习算法；上游的评价、录像和图表不算个人新增能力。
- 未做真机或实验室硬件验证。
- 仓库保留的上游 GIF、success_rates.png 和原 README 中的 12 回合结果属于 Panda 上游，不能用作本次 FR3 证据。

## 来源、许可与保存状态

Menagerie 固定提交及 49 个文件的校验结果见 [来源说明](../robotic-arm-pick-and-place/assets/franka_fr3/SOURCES.md) 和 [清单](../robotic-arm-pick-and-place/assets/franka_fr3/source_manifest.json)。FR3 及 Panda 模型许可证文本原样保留；FR3 LICENSE 中另有 BSD 条款文本，也完整保留。

代码上游 README 声明 MIT；该上游提交未提供独立项目代码 LICENSE。发布版保留原声明和模型许可证，来源与适用范围见根目录 THIRD_PARTY_NOTICES.md。

本阶段保持 FR3 与 Panda 学习流程独立。发布版收录固定场景和指定种子批次，JSON 结果保持原字节，日志中的本机输出路径已替换为相对文件名；原始及发布文件哈希见实验目录的 evidence_manifest.json。新运行仍写入 runs/fr3_migration/，不会覆盖留存证据。

## 可用于简历的事实表述

基于开源 MuJoCo 抓放项目完成 FR3 与双指夹爪的模型组合、抓取参考点及控制接口适配，复用 DLS IK 与规则式流程；在固定场景及 10 个固定随机种子下完成仿真验证，保存 30 次抓放的逐轮 JSON、运行日志与汇总。

描述时注明仿真、上游来源、工程适配和有限评估范围，不宣称原创算法或真机经验。
