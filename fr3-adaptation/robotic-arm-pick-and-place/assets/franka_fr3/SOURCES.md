# 模型与环境来源

核验日期：2026-10-25。

## FR3 与双指夹爪

来源：[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f)。

固定提交：`cda5236f98383f6fbaa4820c6b18dba892cd087f`。

- 本目录 fr3.xml、README.md、LICENSE 及 XML 引用的 36 个网格来自该提交的 franka_fr3/。
- ../franka_emika_panda/hand.xml、LICENSE 及独立 hand 引用的 8 个网格与该提交内容一致。它们通过原 Panda 项目保留在当前基线中。
- 本次未将旧 fr3-pick-place 仓库视为模型来源。
- 原始模型文件保持原样。运行时装配、hand 朝向和抓取点由 Python 项目中的 fr3_model.py 设置。
- 两份 LICENSE 原样保留。FR3 README 声明 Apache-2.0，FR3 LICENSE 同时附带 BSD 条款文本；完整文本均保留。

[文件清单](source_manifest.json)覆盖 49 个来源文件：44 个网格及 FR3 XML、FR3 README、FR3 LICENSE、hand XML、Panda LICENSE。所有文件均与固定提交的 Git blob 匹配；文本只允许 CRLF/LF 换行差异，二进制 STL 按原始字节比较。清单路径相对上一级 assets/。

清单中的 SHA-256 记录本次 Windows 工作副本字节；跨平台 checkout 可能改变文本换行，来源核验应结合 source_git_blob 与 LF 规范化的 Git blob。上游 README 中的图片、CHANGELOG 等未全部复制，可在固定提交页面查看。

## 任务环境

task_environment.xml 派生自 RafayDon/Mujoco-Robotic-Arm-Pick-and-Place 的 robotic-arm-pick-and-place/assets/franka_emika_panda/scene.xml。迁移时删除 Panda include，并在 Python 中将环境接到 FR3；保留上游桌子、方块、目标点和环境配置。

代码上游提交：`0700bff871b28878edca8170754bb08d644fc0ea`。
学习基线：`defc09ce15e32149c2b0425e03967a90014eda5b`。

项目代码使用上游 README 的 MIT 声明，独立项目代码 LICENSE 未找到；原声明及适用范围见合并仓库根目录 THIRD_PARTY_NOTICES.md。

发布时另从同一固定提交补齐本目录 README 引用的 CHANGELOG.md 和 fr3.png；它们不属于运行模型所需的原 49 文件清单。
