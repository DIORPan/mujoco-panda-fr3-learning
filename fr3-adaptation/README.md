# MuJoCo FR3 抓放迁移实践

基于 [RafayDon 的 Panda 抓放项目](https://github.com/RafayDon/Mujoco-Robotic-Arm-Pick-and-Place)，将机器人适配为 **Menagerie FR3 + Panda 双指夹爪**。使用上游阻尼最小二乘 IK、位置伺服和规则式抓放流程。

本项目定位为仿真工程适配与复现评估。模型、IK、随机初始化及原成功判据来自上游；本次改动包括运行时模型组合、抓取点和命名适配、控制器站点参数、FR3 运行入口及结果保存。

## 已验证结果

验证日期：2026-10-25。正常重力 `[0, 0, -9.81]`，未启用重力补偿。

| 验证范围 | 结果 |
|---|---|
| 固定场景一轮，绿 → 红 → 蓝 | 3/3 |
| 随机场景，固定种子 0～9，每轮三个方块 | 30/30 次抓放；10/10 个完整回合 |
| 上述随机批次的 XY 误差均值 / 最大值 | 0.415497 / 0.714171 mm |

成功沿用原判据：每次抓放结束后，方块中心到目标的 XY 距离小于 0.05 m，且中心 z 大于 0.18 m。这些是有限场景中的仿真结果，没有持续稳定性检查，不能推广为总体随机成功率或真机能力。

原始证据：[批次汇总](experiments/2026-10-25/batch_1792914315621804400/summary.json)，同目录含逐轮 JSON 和日志。统计仅使用该批次，没有叠加此前的单轮或预检记录。

## 本机运行

验证环境：Windows、Python 3.13.0、MuJoCo 3.16.0、NumPy 2.5.4；发布版使用独立 `.venv-fr3`。从合并仓库根目录先建立独立环境（与 Panda 学习环境分开）：

```powershell
py -3.13 -m venv .venv-fr3
& '.\.venv-fr3\Scripts\python.exe' -m pip install -r '.\fr3-adaptation\requirements-migration.txt'
```

以下命令从 fr3-adaptation 目录执行，先进入 Python 项目：

```powershell
Set-Location .\robotic-arm-pick-and-place
```

查看完整场景：

```powershell
& '..\..\.venv-fr3\Scripts\python.exe' -B .\view_fr3_scene.py
```

固定场景一轮：

```powershell
& '..\..\.venv-fr3\Scripts\python.exe' -B .\run_fr3_fixed.py
```

指定随机种子，无窗口执行：

```powershell
& '..\..\.venv-fr3\Scripts\python.exe' -B .\run_fr3_random.py --seed 0 --headless
```

重新运行完整种子批次会创建新的结果目录：

```powershell
& '..\..\.venv-fr3\Scripts\python.exe' -B .\run_fr3_batch.py
```

XML 使用相对路径，运行时需保持在上述 Python 项目目录。

## 文档与来源

- [迁移实现、贡献边界和结果说明](docs/FR3_MIGRATION.md)
- [模型来源与许可证说明](robotic-arm-pick-and-place/assets/franka_fr3/SOURCES.md)
- [逐文件来源与校验清单](robotic-arm-pick-and-place/assets/franka_fr3/source_manifest.json)
- [原 Panda 项目说明，含上游自己的结果](robotic-arm-pick-and-place/README.md)

代码上游提交为 `0700bff871b28878edca8170754bb08d644fc0ea`；本次迁移起点为学习基线 `defc09ce15e32149c2b0425e03967a90014eda5b`。模型来源固定为 Menagerie `cda5236f98383f6fbaa4820c6b18dba892cd087f`。

模型许可证文件原样保留。代码使用上游 README 的 MIT 声明；该上游提交未提供独立项目代码 LICENSE，未另行补造上游授权文本。详见[第三方来源说明](../THIRD_PARTY_NOTICES.md)。
