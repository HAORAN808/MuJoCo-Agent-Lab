# 机器人研发实验智能体 Web Demo

本地网页控制台，支持两种管线运行机械臂操作实验。

## 使用方式

### 1. 启动后端

```powershell
cd d:\HuaweiMoveData\Users\hw\Desktop\try

# 真实 MuJoCo 物理仿真（推荐）
.venv\Scripts\python.exe -m mujoco_bridge.server

# fallback 模式（无需 MuJoCo 物理引擎）
.venv\Scripts\python.exe -m mujoco_bridge.server --fallback
```

**注意**：必须使用 `.venv\Scripts\python.exe` 启动，因为项目依赖安装在 `.venv` 虚拟环境中。

### 2. 打开网页

直接用浏览器打开 `web_demo/index.html`。

## NLP 智能管线（推荐）

一键完成：输入自然语言 → AI 解析 → 自动场景组装 → MuJoCo 仿真 → 结果分析。

1. 在「研发目标」文本框输入任务描述，例如：
   - "用 UR5e 抓取桌上的圆柱罐并放到左侧"
   - "研究不同摩擦力对 KUKA iiwa 拧螺丝成功率的影响"
   - "机械臂插入装配任务的成功率和失败原因分析"
2. 选择机器人（或留空自动选择）
3. 选择实验规模（9/27/81 组）
4. 点击「一键运行 NLP 管线」
5. 进度条实时显示当前阶段（初始化 → 任务解析 → 场景组装 → 仿真实验 → 结果分析 → 完成）
6. 查看：实验指标、失败分布图、变量影响图、关键发现、下一轮建议
7. 生成并下载 Markdown 报告

**任务委托**：当输入匹配已注册任务的关键词时（如"拧螺丝"→screwdriving），自动委托给对应的专用 runner，获得更精确的实验结果。

**真实渲染**：轨迹回放显示真实 MuJoCo 渲染帧，非 canvas 2D 示意图。

支持 8 个 menagerie 机械臂：FR3、Panda、UR5e、UR10e、Kinova Gen3、KUKA iiwa、xArm7、Lite6。

## 传统任务管线

折叠在页面底部，用于已注册的 7 个硬编码任务：

1. 展开「传统任务管线」
2. 选择任务
3. 勾选 API 模式
4. 点击「运行仿真实验」

## API 调用

```powershell
# NLP 管线
curl -X POST http://127.0.0.1:8765/api/nlp/run -H "Content-Type: application/json" -d "{\"goal\":\"用 FR3 抓取方块\",\"limit\":9,\"language\":\"zh\"}"

# 列出可用机器人
curl http://127.0.0.1:8765/api/robots

# 列出已注册任务
curl http://127.0.0.1:8765/api/tasks

# NLP 进度查询
curl http://127.0.0.1:8765/api/nlp/status

# 传统批量实验
curl -X POST http://127.0.0.1:8765/api/run_experiments -H "Content-Type: application/json" -d "{\"limit\":27}"
```

## 页面结构

1. **输入目标 / 运行** — NLP 管线一键操作 + 实时进度条 + 传统任务管线（折叠）
2. **实验设计与变量空间** — 假设、变量、机器人资产
3. **实验结果** — 指标卡片、失败分布图、变量影响图、轨迹回放（真实 MuJoCo 渲染帧）
4. **失败归因** — 关键发现、下一轮建议
5. **实验报告** — Markdown 生成和下载
