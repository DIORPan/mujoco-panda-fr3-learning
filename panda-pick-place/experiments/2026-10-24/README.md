# 2026-10-24 冻结实验

- [paired_results.json](paired_results.json)：脱敏的20对实验结果、源文件哈希、数据规模、环境与逐回合指标。
- [checkpoints/bc.pt](checkpoints/bc.pt)：原BC便携冻结权重。
- [checkpoints/aggregated.pt](checkpoints/aggregated.pt)：闭合后状态补标微调的便携冻结权重。
- [同步视频](../../robotic-arm-pick-and-place/media/learning_comparison.mp4)与[回放校验](../../robotic-arm-pick-and-place/media/learning_comparison.json)。

冻结权重仅包含参数和必要元数据，没有原始训练数据和本机绝对路径。逐参数相等检查通过；便携文件重打包后的文件SHA256与原checkpoint不同，两个哈希均保存在结果JSON中。

通过[仓库README](../../README.md)中的 show_frozen_policy.py 命令可直接推理。训练与微调不在此命令中发生。

14/20与20/20仅对应这批同范围、固定起点、仿真状态输入的接近抓取任务；视频选取其中一个对照种子。完整复现命令见[指南](../../docs/reproduction.md)。
