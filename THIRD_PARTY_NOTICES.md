# 第三方来源与许可说明

## 抓放项目代码

来源：[RafayDon/Mujoco-Robotic-Arm-Pick-and-Place](https://github.com/RafayDon/Mujoco-Robotic-Arm-Pick-and-Place)，上游代码提交 `0700bff871b28878edca8170754bb08d644fc0ea`。

上游 README 声明代码为 MIT。原文保存在 [Panda 上游说明](panda-pick-place/docs/upstream-readme.md) 和 [FR3 内的上游说明](fr3-adaptation/robotic-arm-pick-and-place/README.md)。该提交未包含独立的项目代码 LICENSE；本发布保留实际声明，未补造上游版权或授权文本。模型资产许可单独保留如下。

上游提供 Panda 场景、DLS IK、规则式抓放、随机化、评价、图表及录像功能。原 GIF、success_rates.png 与原文 12 回合结果均属上游资料。

## 模型资产

Panda 与 FR3 资产来自 [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。

- [Panda 模型许可（学习目录）](panda-pick-place/robotic-arm-pick-and-place/assets/franka_emika_panda/LICENSE)。
- [Panda/夹爪模型许可（迁移目录）](fr3-adaptation/robotic-arm-pick-and-place/assets/franka_emika_panda/LICENSE)。
- [FR3 模型许可](fr3-adaptation/robotic-arm-pick-and-place/assets/franka_fr3/LICENSE)，包含原始 Apache-2.0 及附带 BSD 条款。
- [FR3 资产来源说明](fr3-adaptation/robotic-arm-pick-and-place/assets/franka_fr3/SOURCES.md) 与 [逐文件清单](fr3-adaptation/robotic-arm-pick-and-place/assets/franka_fr3/source_manifest.json)，固定 Menagerie 提交 `cda5236f98383f6fbaa4820c6b18dba892cd087f`。

模型许可证、来源清单和模型文件均保留。FR3 与手爪在运行时组合，不改写原资产 XML。

## 算法参考

行为克隆使用专家动作监督；策略访问状态的专家补标参考 [Ross, Gordon & Bagnell, 2011, DAgger](https://proceedings.mlr.press/v15/ross11a.html)。本项目采用闭合后状态的针对性补标与一次数据聚合，不声称覆盖全部原算法设置。
