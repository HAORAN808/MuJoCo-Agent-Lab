# 通用机械臂实验智能体 Demo 展示稿

> 本文档用于 Demo 展示环节，展示项目目标、系统架构、运行方式、核心输出、能力边界与后续演进。

---

## 0. 展示目标

**一句话定位**

本项目是一个面向机器人操作实验的实验工作流 Agent 原型。

**展示主线**

```text
自然语言目标
  -> TaskSpec
  -> ExperimentPlan
  -> SkillPlan
  -> MuJoCo 执行
  -> EvaluationReport
  -> RetryPlan / AutoRetry
  -> ExperimentStore
```

**本次展示要证明**

- 实验目标可以被协议化。
- 机械臂实验可以被真实物理仿真执行。
- 结果可以被严格评估和失败归因。
- 系统可以自动做一轮修正实验，并沉淀实验记忆。

---

## 1. 要解决的问题

以这个实验目标为例：

```text
用 fr3 机械臂推动方块移动5cm，不能陷进桌子
```

系统需要同时解决：

- 识别任务类型：push
- 选择机器人和对象：FR3 + cube_5cm
- 抽取目标约束：移动 5cm
- 抽取物理约束：不能陷进桌子
- 生成可执行技能序列
- 执行后判断是否真的成功
- 失败后给出修正并自动重试

---

## 2. 系统架构

```text
User Goal
  |
Agent Orchestrator
  |-- TaskSpec
  |-- ExperimentPlan
  |-- SkillPlan
  |
Instrument Layer
  |-- MuJoCo
  |-- Motion Primitives
  |-- IK Solver
  |
Feedback Layer
  |-- Runs / Metrics / Replay
  |-- EvaluationReport
  |-- RetryPlan / AutoRetry
  |-- ExperimentStore
```

**关键分工**

- Agent：目标理解、实验计划、能力匹配、失败归因、重试决策。
- MuJoCo：可编程虚拟实验仪器。
- SkillPlan / IK：低层动作执行。
- EvaluationReport：可信评估。
- ExperimentStore：实验记忆。

---

## 3. 演示运行方式

**启动命令**

```powershell
cd D:\HuaweiMoveData\Users\hw\Desktop\try
powershell -ExecutionPolicy Bypass -File .\scripts\open_mujoco_web_demo.ps1
```

**健康检查**

```text
http://127.0.0.1:8765/health
```

**演示目标输入**

```text
用 fr3 机械臂推动方块移动5cm，不能陷进桌子
```

**Web Demo 操作**

1. 输入目标。
2. 选择或保持 FR3。
3. 勾选使用 MuJoCo API。
4. 点击运行 NLP 智能体实验。
5. 按页面从上到下讲解结果。

---

## 4. 任务协议输出：TaskSpec

页面中重点看：

```text
task_type = push
robot = franka_fr3
object = cube_5cm
target_displacement_m = 0.05
tolerance = 0.015
table_clearance constraint
```

**这一步证明**

- 系统理解了“推动”。
- 系统理解了“5cm”。
- 系统理解了“不能陷进桌子”。
- 自然语言被转成后端可执行协议。

---

## 5. 智能体闭环输出

页面中会看到：

```text
TaskSpec
ExperimentPlan
SkillPlan
Capability
MuJoCo
Evaluation
RetryPlan
AutoRetry
Memory
```

**每个节点含义**

- TaskSpec：任务协议。
- ExperimentPlan：实验变量和指标。
- SkillPlan：机械臂技能序列。
- Capability：机器人/工具/物体是否支持。
- MuJoCo：物理执行。
- Evaluation：结果评估。
- RetryPlan：失败后的修正计划。
- AutoRetry：自动执行一轮修正实验。
- Memory：实验记录写入历史。

---

## 6. 物理执行输出：MuJoCo

MuJoCo 执行会产生：

- object_displacement
- contact_steps
- max_touch_force
- final_object_pos
- final_ee_pos
- replay frames

**这一步证明**

- 执行不是前端动画。
- 结果来自 MuJoCo 物理仿真。
- 数据可以进入后续评估和报告。

---

## 7. 严格评估输出：EvaluationReport

推动任务的成功标准：

```text
abs(object_displacement - target_displacement_m) <= tolerance
and table_clearance_ok == true
```

**失败类型**

| failure_type | 含义 |
| --- | --- |
| no_contact | 没接触到目标 |
| undershoot | 推得不够 |
| overshoot | 推得太远 |
| table_penetration | 桌面安全失败 |

**这一步证明**

- 移动 30cm 不会被判成功。
- 移动不足不会被判成功。
- 机械臂陷进桌子不会被判成功。
- 成功率有明确验收口径。

---

## 8. 自动重试输出：AutoRetry

当 EvaluationReport 出现失败类型时：

```text
EvaluationReport
  -> dominant failure
  -> RetryPlan
  -> revised task spec
  -> retry runs
  -> before/after comparison
```

示例：

```text
table_penetration
  -> raise approach / enforce clearance
  -> rerun
  -> compare source failure rate
```

**这一步证明**

- 系统不是只给建议。
- 系统能自动执行一轮修正实验。
- 重试结果会和初轮结果对比。

---

## 9. 实验记忆输出：Experiment Memory

每轮 NLP 实验会写入：

```text
results/nlp_experiment_store.jsonl
```

记录内容：

- TaskSpec
- ExperimentPlan
- SkillPlan
- EvaluationReport
- RetryPlan
- RetryExecution
- recorded_at

**这一步证明**

- 实验不是一次性输出。
- 历史结果可以被检索、统计和复用。
- 后续可以做参数推荐和失败边界总结。

---

## 10. 技术原理简述

**分层设计**

```text
自然语言层：Goal
协议层：TaskSpec / ExperimentPlan / SkillPlan
执行层：MotionPlan / IK / MuJoCo
评估层：EvaluationReport / RetryPlan
记忆层：ExperimentStore
```

**IK 执行**

```text
dq = J^T (J J^T + λ²I)^-1 e
```

**安全约束**

- 物体初始高度按几何尺寸放到桌面上方。
- 目标位移写入 success_criteria。
- 执行中记录 table_clearance_ok。
- 桌面安全失败不能算成功。

---

## 11. 当前边界

当前 Demo 已完成：

- 自然语言到 TaskSpec。
- MuJoCo 物理执行。
- 严格成功判定。
- 失败归因。
- 一次自动重试。
- 实验记忆写入。

当前仍未完成：

- 长期多轮自主实验。
- 完整全局运动规划。
- 真实机械臂硬件接入。
- 大规模统计可信评估。
- 跨任务历史经验复用。

---

## 12. 下一步演进

**短期**

- AutoRetry 从一轮扩展为多轮。
- 增加停止条件和收敛判断。
- 强化 ExperimentStore 历史检索。

**中期**

- 接入 MoveIt / OMPL / TrajOpt。
- 引入全身碰撞检测和轨迹优化。
- 增加 strict success / partial success 分层。

**长期**

- 接入真实机械臂和传感器。
- 建立真实实验 Adapter。
- 用历史实验做主动实验设计。

---

## 13. 结尾总结

这个 Demo 的核心价值：

```text
把实验目标变成协议；
把协议变成可执行实验；
把执行结果变成结构化反馈；
把失败反馈变成下一轮行动；
把每轮实验沉淀为记忆。
```

**最后一句**

本项目展示的是一个实验工作流 Agent 的最小闭环：它已经能理解目标、调用实验工具、严格评估结果、自动做一轮修正，并记录实验历史。下一步是把一次闭环扩展为长期多轮自主实验。
