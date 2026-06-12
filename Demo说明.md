# 面向机器人研发实验的主动实验设计与具身执行闭环智能体 - Demo 说明

## 1. 项目定位

**MuJoCo 机械臂操作通用实验智能体** — 输入任意自然语言任务描述，AI 自动完成任务解析、场景组装、实验设计、真实 MuJoCo 物理仿真、结果分析和下一轮迭代建议。

## 2. 核心能力

### 2.1 NLP 智能管线

输入自然语言任务描述，AI 自动完成全流程：

```text
"用 UR5e 抓取桌上的圆柱罐并放到左侧，研究不同摩擦力对成功率的影响"
  -> LLM 解析为结构化 TaskSpec (robot, objects, actions, success_criteria, experiment_variables)
  -> DynamicSceneComposer 自动组装 MuJoCo XML 场景
  -> MotionPlan 构建运动计划 (reach, grasp, lift, place)
  -> 真实 MuJoCo 物理仿真批量实验
  -> MetricsEngine 评估 (接触检测、物体追踪、力监控)
  -> LLM 分析结果 + 下一轮实验建议
```

**NLP 箿线有两条子路径：**

1. **任务委托路径**：当自然语言输入匹配已注册任务的关键词时（如"拧螺丝"→`screwdriving`），直接委托给对应的已注册任务 runner。这些 runner 有精心设计的真实 MuJoCo 场景、专用执行器和精确的成功判定逻辑。

2. **通用动态路径**：当输入不匹配任何已注册任务时，使用 LLM 解析 + 动态场景组装 + 通用运动原语执行。

**真实 MuJoCo 渲染**：两条路径都使用 `mujoco.Renderer` 生成真实的 MuJoCo 渲染帧（非 canvas 2D 示意图），用于轨迹回放。

**真实成功判定**：按任务类型使用物理阈值判定成功/失败（如 pick_and_place 要求 lifted_height > 3cm）。

支持 8 个 menagerie 机械臂，自动加载真实 MJCF 模型（mesh、关节、执行器、夹爪）。

### 2.2 传统任务管线

7 个已注册硬编码任务，使用 FR3 机械臂：

- `fr3_arm_primitives`：通用 FR3 机械臂基础实验
- `fr3_pick_place`：FR3 抓取放置
- `tabletop_push`：桌面推动
- `screwdriving`：拧螺丝
- `tool_use`：工具使用
- `assembly_insertion`：装配插入
- `cloth_folding`：布料折叠

### 2.3 多轮迭代

- 变量空间收窄：一致失败的变量值被剔除
- 收敛检测：最近 N 轮成功率相似时提示停止
- LLM 分析包含完整历史上下文

## 3. 使用方式

### 3.1 启动

```powershell
cd d:\HuaweiMoveData\Users\hw\Desktop\try

# 真实 MuJoCo 物理仿真（推荐）
.venv\Scripts\python.exe -m mujoco_bridge.server

# fallback 模式（确定性代理 runner）
.venv\Scripts\python.exe -m mujoco_bridge.server --fallback
```

**注意**：必须使用 `.venv\Scripts\python.exe` 启动，因为项目依赖安装在 `.venv` 虚拟环境中。

### 3.2 操作

1. 用浏览器打开 `web_demo/index.html`
2. 在「研发目标」输入任务描述
3. 选择机器人（或留空自动选择）
4. 点击「一键运行 NLP 管线」
5. 查看实验结果、分析和建议
6. 生成并下载 Markdown 报告

### 3.3 API 调用

```powershell
# NLP 管线
curl -X POST http://127.0.0.1:8765/api/nlp/run -H "Content-Type: application/json" -d "{\"goal\":\"用 FR3 抓取方块\",\"limit\":9,\"language\":\"zh\"}"

# 列出可用机器人
curl http://127.0.0.1:8765/api/robots

# 列出已注册任务
curl http://127.0.0.1:8765/api/tasks

# NLP 进度查询
curl http://127.0.0.1:8765/api/nlp/status
```

## 4. 技术架构

```text
mujoco_bridge/
  agent.py             # LLM 智能体 + NLP 管线编排
  robot_registry.py    # 8 个 menagerie 机械臂 RobotSpec 解析
  ik_solver.py         # Jacobian IK 求解器
  motion_primitives.py # 任务空间运动原语 + MuJoCo 渲染回调
  scene_composer.py    # 动态 MuJoCo 场景组装器
  metrics_engine.py    # 任务无关指标引擎
  experiment_history.py# 多轮实验历史追踪
  server.py            # HTTP API 服务器 + 进度追踪
  tasks/*.py           # 7 个已注册任务 runner
  arm_runner.py        # FR3 机械臂技能 runner
  object_library.py    # 物体模型库 (19 个物体)
  asset_library.py     # 资产注册表
  task_render.py       # MuJoCo 帧编码 (JPEG base64)
```

## 5. 实验输出

每轮实验包含：

- **指标**：成功率、失败类型分布、可解释失败占比
- **图表**：失败分布柱状图、变量影响分析图
- **分析**：关键发现（含置信度）、下一轮实验建议
- **日志**：每组实验的详细参数和结果
- **报告**：Markdown 格式，可下载
- **回放**：成功/失败样例轨迹对比（真实 MuJoCo 渲染帧，非示意图）
