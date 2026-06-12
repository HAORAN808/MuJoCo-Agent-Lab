# 智能体接入与运行说明

## 两条管线

### NLP 智能管线（推荐）

输入任意自然语言任务描述，AI 自动完成全流程，无需手动选择任务：

```text
POST /api/nlp/run
```

执行链路：

1. `resolve_task(goal)` — 关键词匹配，检查是否命中已注册任务
2. **[命中]** 直接委托给已注册任务 runner（真实 MuJoCo 场景 + 渲染帧）
3. **[未命中]** LLM 解析为结构化 TaskSpec → 动态场景组装 → 通用运动原语执行
4. 真实 MuJoCo 物理仿真批量实验（接触、摩擦、重力）
5. 真实 MuJoCo 渲染帧采集（`mujoco.Renderer`，非 canvas 2D）
6. 按任务类型的真实成功判定（物理阈值）
7. LLM 分析结果 + 下一轮实验建议

**任务委托匹配规则：**

| 关键词 | 委托任务 |
|--------|----------|
| screw, screwdriver, bolt, nut, 拧, 螺丝 | `screwdriving` |
| hammer, spatula, tool, 锤子, 铲子 | `tool_use` |
| insert, assembly, 装配, 插入 | `assembly_insertion` |
| cloth, fold, towel, 布, 折叠 | `cloth_folding` |
| push, 推, 滑动 | `tabletop_push` |
| pick, place, grasp, 抓, 放, 取 | `fr3_pick_place` |

支持 8 个 menagerie 机械臂：franka_fr3, franka_emika_panda, universal_robots_ur5e, universal_robots_ur10e, kinova_gen3, kuka_iiwa_14, ufactory_xarm7, ufactory_lite6

请求示例：

```json
{
  "goal": "用 UR5e 抓取桌上的圆柱罐并放到左侧，研究不同摩擦力对成功率的影响",
  "limit": 27,
  "language": "zh",
  "robot_id": "universal_robots_ur5e"
}
```

`robot_id` 可选，留空则自动检测（从自然语言文本中提取机器人名称）。

响应包含：`task_spec`、`robot_id`、`robot_spec`、`runs`、`summary`、`analysis`、`demo_trace`、`agent_trace`

### 传统任务管线

已注册的 7 个硬编码任务，使用 FR3 机械臂：

```text
POST /api/run_experiments    # 不调用模型 API
POST /api/agent/run          # 调用模型 API 做路由、设计和分析
```

请求示例：

```json
{
  "goal": "研究桌面推动物块时摩擦对最终距离的影响",
  "task_id": "tabletop_push",
  "limit": 27,
  "language": "zh"
}
```

## 智能体做什么

NLP 管线的智能体会：

1. 解析用户输入的自然语言任务描述
2. 检查是否匹配已注册任务（优先委托）
3. 选择合适的机器人（从文本自动检测或 LLM 推荐）
4. 确定物体、动作序列和成功标准
5. 定义实验变量空间
6. 运行真实 MuJoCo 物理仿真
7. 使用 `mujoco.Renderer` 采集渲染帧
8. 按任务类型的真实物理阈值判定成功/失败
9. 分析失败原因
10. 给出下一轮实验建议

模型不会直接执行任意代码。它只能在已注册的机器人、物体和动作原语范围内做选择和设计。

## 已注册任务

| task_id | 描述 | 执行器 |
|---------|------|--------|
| `fr3_arm_primitives` | 通用 FR3 机械臂基础实验 | FR3 arm + Franka Hand |
| `fr3_pick_place` | FR3 抓取放置 | FR3 arm + Franka Hand |
| `tabletop_push` | 桌面推动 | 推杆 |
| `screwdriving` | 拧螺丝 | 螺丝刀 carrier |
| `tool_use` | 工具使用 | 工具 carrier |
| `assembly_insertion` | 装配插入 | slide joint |
| `cloth_folding` | 布料折叠 | MuJoCo flexcomp |

## 进度追踪

NLP 管线运行时，前端可通过轮询获取实时进度：

```text
GET /api/nlp/status
```

返回：

```json
{
  "step": "场景组装",
  "detail": "正在组装 FR3 + cube_5cm 场景...",
  "percent": 35
}
```

进度阶段：初始化 → 任务解析 → 场景组装 → 仿真实验 → 结果分析 → 完成

## 模型配置

检查配置状态：

```text
GET /api/agent/status
```

配置文件位于：

```text
configs/model_api.local.json
```

格式：

```json
{
  "api_key": "your-api-key",
  "base_url": "https://api.xiaomimimo.com/v1",
  "model": "MiMo-V2.5-Pro"
}
```

不配置时，NLP 管线会使用本地确定性解析器作为 fallback。

## 环境变量

- `AGENT_FORCE_LOCAL=1`：强制使用本地解析器，不调用模型 API

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
