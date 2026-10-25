# Python代码入口

项目介绍、实测结果和Windows环境安装见[仓库README](../README.md)。

- 查看冻结模型：show_frozen_policy.py
- 新任务专家演示：demo_reach_grasp.py
- 专家数据采集与BC：collect_reach_grasp_dataset.py、train_reach_grasp_bc.py
- 闭合后状态补标与微调：collect_reach_grasp_dagger.py、train_reach_grasp_dagger.py
- 新位置配对评估：evaluate_reach_grasp_dagger.py
- 原动作记录导出视频：record_learning_comparison.py

[复现与排错](../docs/reproduction.md) · [架构与接口](../docs/architecture.md) · [实验结果](../docs/experiment-results.md)

依赖分成 requirements-bc-cpu.txt（学习）与 requirements-video.txt（可选视频导出）。原 requirements.txt、scripts/、eval/ 和 utils/ 仍属于上游规则控制功能。

datasets/ 与 runs/bc/ 为本机生成数据，默认不纳入版本控制。便携冻结模型和脱敏证据保存在仓库 experiments/2026-10-24/，视频保存在 media/。

历史实验：
- collect_bc_episode.py、train_bc.py、rollout_bc.py：原25状态/8关节目标的全流程BC诊断。
- grasp_task.py、collect_grasp_dataset.py、train_grasp_bc.py、rollout_grasp_bc.py、evaluate_grasp_bc.py：预对齐后的6状态/2动作垂直抓取基线。
- reach_grasp_task.py 与 reach_grasp 系列：固定悬停起点的10状态/4动作接近与抓取。

这些数据格式和模型格式不同，加载器会拒绝混用。
