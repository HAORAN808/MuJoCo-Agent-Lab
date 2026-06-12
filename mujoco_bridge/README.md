# MuJoCo 机械臂操作实验智能体后端

## 架构

### NLP 智能管线（推荐）

```text
POST /api/nlp/run
  -> agent.py: run_nlp_agent_round()
    -> resolve_task(goal)                   # 关键词匹配已注册任务
    -> [命中] 已注册任务 runner              # 真实 MuJoCo 场景 + 渲染帧
    -> [未命中]
       -> _local_parse_task(goal, robot_id) # LLM/本地解析为 TaskSpec
       -> robot_registry.get_robot()        # 加载 menagerie 机械臂 RobotSpec
       -> DynamicSceneComposer.compose()    # 动态组装 MuJoCo XML 场景
       -> MotionPlan.from_action_sequence() # 构建运动计划
       -> _run_single_nlp_experiment()      # 真实 MuJoCo 物理仿真
          -> execute_plan()                 # IK + 运动原语 + 渲染回调
          -> _evaluate_run()                # 按任务类型的真实物理阈值判定
    -> summarize_runs()                     # 统计汇总
    -> analyze_runs()                       # LLM 分析 + 下一轮建议
```

### 传统任务管线

```text
POST /api/run_experiments
  -> tasks/registry.py -> 具体任务 runner -> MuJoCo

POST /api/agent/run
  -> agent.py: run_agent_round()
    -> route_task() -> design_experiment() -> 任务 runner -> analyze_runs()
```

## 核心模块

### robot_registry.py — 机器人注册表

解析 menagerie MJCF 文件，提取统一的 RobotSpec：

```python
@dataclass(frozen=True)
class RobotSpec:
    robot_id: str              # "franka_fr3"
    name: str                  # "Franka Emika FR3"
    mjcf_path: str             # 主 XML 路径
    dof: int                   # 6 或 7
    joint_names: List[str]
    actuator_names: List[str]
    actuator_type: str         # "position" 或 "general"
    end_effector_site: str     # "attachment_site"
    has_gripper: bool
    gripper_type: str          # "parallel", "none"
    keyframe_qpos: Optional[List[float]]
    ...
```

支持 8 个机械臂：franka_fr3, franka_emika_panda, universal_robots_ur5e, universal_robots_ur10e, kinova_gen3, kuka_iiwa_14, ufactory_xarm7, ufactory_lite6

### ik_solver.py — Jacobian IK 求解器

阻尼最小二乘 IK：`dq = J^T @ inv(J @ J^T + λ²I) @ error`

- 使用 `mj_jacSite` / `mj_jacBody` 计算 Jacobian
- 奇异处理：自适应增大阻尼
- 关节限位：每步 clamp

### motion_primitives.py — 运动原语

可组合的、机器人无关的运动原语：

- `reach(target_pos)` — IK 求解 + 关节空间插值
- `grasp(force_scale)` — 闭合夹爪
- `lift(height)` — 抬升
- `place(target_pos)` — 移动到目标 + 打开夹爪
- `push(direction, distance)` — 沿方向推动
- `insert(target_pos, approach_axis)` — 沿轴插入

**渲染支持**：`execute_plan()` 接受 `render_callback` 参数，在仿真步进时定期调用 `mujoco.Renderer` 采集渲染帧。`_drive_to_qpos()`、`_do_gripper()`、`_do_wait()` 均会调用渲染回调。

GripperController 按机器人族适配：Franka Hand（tendon）、xArm7（joints）、UR5e+Robotiq（附加 actuator）

### scene_composer.py — 场景组装器

从 LLM 生成的 SceneDescription 自动组装有效 MuJoCo XML：

- 提取机器人原始 `<default>`, `<asset>`, `<actuator>`, `<tendon>`
- 物体加载复用 `mjcf_assets.import_mjcf_model()`
- 工作台高度根据机器人 base 自动计算
- 组装后用 `MjModel.from_xml_string()` 验证

### metrics_engine.py — 指标引擎

任务无关的评估：

- 接触检测（任意两个 body 之间）
- 物体位姿追踪
- 力阈值监控
- 成功标准：object_at_target, contact_achieved, object_displaced, lifted

### experiment_history.py — 多轮迭代

- 存储每轮完整上下文
- 变量空间收窄（一致失败的剔除）
- 收敛检测（最近 N 轮成功率相似）

## 真实成功判定

按任务类型使用物理阈值，而非简单的接触检测：

| 任务类型 | 成功条件 |
|----------|----------|
| `pick_and_place` | lifted_height > 3cm 且 object_displacement > 3cm |
| `push` | object_displacement > 3cm |
| `screwdriving` | final_angle > 180° |
| `tool_use` | object_displacement > 2cm |
| `assembly_insertion` | insertion_depth > 1cm |
| `cloth_folding` | corner_gathered 为真 |

## 已注册任务

| task_id | 描述 |
|---------|------|
| `fr3_arm_primitives` | 通用 FR3 机械臂基础实验 |
| `fr3_pick_place` | FR3 抓取放置 |
| `tabletop_push` | 桌面推动 |
| `screwdriving` | 拧螺丝 |
| `tool_use` | 工具使用 |
| `assembly_insertion` | 装配插入 |
| `cloth_folding` | 布料折叠 |

## API

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/robots` | 列出 8 个可用机器人 |
| GET | `/api/tasks` | 列出已注册任务 |
| GET | `/api/object_library` | 物体模型库 |
| GET | `/api/asset_registry` | 资产注册表 |
| GET | `/api/experiment_space` | 实验变量空间 |
| GET | `/api/arm/skills` | 机械臂技能列表 |
| GET | `/api/runner/status` | Runner 状态 |
| GET | `/api/agent/status` | 智能体配置状态 |
| GET | `/api/nlp/status` | NLP 管线进度 |
| POST | `/api/nlp/run` | NLP 智能管线 |
| POST | `/api/run_experiments` | 传统批量实验 |
| POST | `/api/agent/run` | 智能体闭环 |
| POST | `/api/arm/run_skill` | 单个机械臂技能 |

## 启动

```powershell
cd d:\HuaweiMoveData\Users\hw\Desktop\try

# 真实 MuJoCo 物理仿真（推荐）
.venv\Scripts\python.exe -m mujoco_bridge.server

# fallback 模式
.venv\Scripts\python.exe -m mujoco_bridge.server --fallback

# 指定端口
.venv\Scripts\python.exe -m mujoco_bridge.server --port 9000
```

**注意**：必须使用 `.venv\Scripts\python.exe` 启动，因为项目依赖安装在 `.venv` 虚拟环境中。

## 验证

```powershell
# 检查所有 runner
powershell -ExecutionPolicy Bypass -File .\scripts\verify_runners.ps1

# 检查所有任务
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke_all_tasks.ps1

# 检查机械臂技能
powershell -ExecutionPolicy Bypass -File .\scripts\run_arm_skill_smoke.ps1

# 检查智能体闭环
powershell -ExecutionPolicy Bypass -File .\scripts\verify_agent_loop.ps1
```
