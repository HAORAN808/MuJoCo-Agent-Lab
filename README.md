<div align="center">

# 🤖 MuJoCo Agent Lab

### AI-Driven Autonomous Robot Experimentation Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3+-green.svg)](https://mujoco.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**一个基于LLM驱动的自主机器人实验闭环系统，实现"假设-设计-仿真-分析-迭代"的全自动研发流程**

[English](#english) | [中文](#中文)

</div>

---

## 中文

### ✨ 核心特性

<table>
<tr>
<td width="50%">

**🧠 智能任务解析**
- 自然语言输入（中英文）
- LLM自动解析为结构化任务规格
- 支持8种主流机械臂

</td>
<td width="50%">

**🔧 动态场景组装**
- 自动加载MJCF机器人模型
- 智能匹配物体和工具
- 实时工作台配置

</td>
</tr>
<tr>
<td>

**⚡ 真实物理仿真**
- MuJoCo物理引擎
- Jacobian IK运动规划
- 接触/摩擦/重力仿真

</td>
<td>

**📊 智能实验设计**
- 自动生成变量扫描矩阵
- 假设驱动的实验设计
- 批量并行仿真

</td>
</tr>
<tr>
<td>

**🔍 失败归因分析**
- 多维度失败分类
- 物理阈值判定
- 根因定位

</td>
<td>

**🔄 迭代优化**
- 历史结果追踪
- 变量空间收窄
- 收敛检测

</td>
</tr>
</table>

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    自然语言任务描述输入                            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  🧠 LLM任务解析层                                                │
│  ├─ 任务路由 (route_task)                                        │
│  ├─ 结构化解析 (TaskSpec)                                        │
│  └─ 机器人/物体/动作识别                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔧 场景组装层                                                    │
│  ├─ DynamicSceneComposer                                         │
│  ├─ Menagerie MJCF加载                                           │
│  └─ 物体/工具/工作台配置                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  ⚡ 运动规划层                                                    │
│  ├─ JacobianIKSolver (阻尼最小二乘)                              │
│  ├─ 任务空间原语 (reach, grasp, lift, place...)                  │
│  └─ GripperController                                            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  🎯 MuJoCo物理仿真层                                             │
│  ├─ 批量实验执行                                                  │
│  ├─ 接触检测 & 力监控                                             │
│  └─ 渲染帧采集                                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  📊 分析与迭代层                                                  │
│  ├─ 指标引擎评估                                                  │
│  ├─ 失败归因分析                                                  │
│  ├─ LLM结果解读                                                   │
│  └─ 下一轮实验建议                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 🤖 支持的机器人

| 机器人 | 型号 | 自由度 | 特点 |
|--------|------|--------|------|
| <img src="https://img.shields.io/badge/FR3-7DOF-orange" /> | Franka FR3 | 7 | 高精度力控 |
| <img src="https://img.shields.io/badge/Panda-7DOF-orange" /> | Franka Emika Panda | 7 | 研究级标准 |
| <img src="https://img.shields.io/badge/UR5e-6DOF-blue" /> | Universal Robots UR5e | 6 | 工业协作 |
| <img src="https://img.shields.io/badge/UR10e-6DOF-blue" /> | Universal Robots UR10e | 6 | 大负载 |
| <img src="https://img.shields.io/badge/Gen3-7DOF-purple" /> | Kinova Gen3 | 7 | 轻量化 |
| <img src="https://img.shields.io/badge/iiwa-7DOF-red" /> | KUKA iiwa 14 | 7 | 高刚度 |
| <img src="https://img.shields.io/badge/xArm7-7DOF-green" /> | UFactory xArm7 | 7 | 性价比 |
| <img src="https://img.shields.io/badge/Lite6-6DOF-cyan" /> | UFactory Lite6 | 6 | 轻量协作 |

### 🎯 支持的任务类型

```python
# 📦 抓取放置
"用FR3抓取桌上的方块并放到左侧"

# 🔩 装配插入
"使用UR5e完成轴孔装配任务"

# 🔧 工具操作
"用机械臂握持螺丝刀拧紧螺丝"

# 📐 桌面推动
"研究不同摩擦力对推动距离的影响"

# 👕 柔性操作
"折叠桌上的毛巾"

# 🔘 按钮操作
"按压控制面板上的按钮"
```

### 🚀 快速开始

#### 环境要求

- Python 3.10+
- Windows/Linux/macOS

#### 安装

```bash
# 克隆仓库
git clone https://github.com/HANRAN808/MuJoCo-Agent-Lab.git
cd MuJoCo-Agent-Lab

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements-mujoco.txt
```

#### 启动服务

```bash
# 启动HTTP服务器 (默认端口8765)
python -m mujoco_bridge.server

# 或指定端口
python -m mujoco_bridge.server --port 9000

# Fallback模式 (无需MuJoCo)
python -m mujoco_bridge.server --fallback
```

#### 配置LLM (可选)

创建 `configs/model_api.local.json`:

```json
{
  "api_key": "your-api-key",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4"
}
```

支持 OpenAI / Anthropic / 小米MiMo 等兼容API。

### 📡 API接口

#### 1. NLP智能管线 (推荐)

```bash
POST /api/nlp/run
Content-Type: application/json

{
  "goal": "用FR3抓取桌上的方块并放到左侧，研究不同摩擦力对成功率的影响",
  "limit": 27,
  "language": "zh",
  "robot_id": "franka_fr3"  # 可选，自动检测
}
```

**响应包含：**
- `task_spec` - 结构化任务规格
- `runs` - 实验运行结果
- `summary` - 统计摘要
- `analysis` - LLM分析报告
- `demo_trace` - 渲染帧序列

#### 2. 传统任务管线

```bash
POST /api/run_experiments
{
  "task_id": "tabletop_push",
  "limit": 27
}
```

#### 3. 实时进度

```bash
GET /api/nlp/status

# 响应
{
  "step": "场景组装",
  "detail": "正在组装 FR3 + cube_5cm 场景...",
  "percent": 35
}
```

### 📁 项目结构

```
MuJoCo-Agent-Lab/
├── 📂 mujoco_bridge/           # 核心源码
│   ├── 📄 server.py            # HTTP服务器 & API
│   ├── 📄 agent.py             # LLM智能体
│   ├── 📄 runner.py            # 仿真运行器
│   ├── 📄 ik_solver.py         # IK求解器
│   ├── 📄 motion_primitives.py # 运动原语
│   ├── 📂 tasks/               # 任务定义
│   ├── 📂 skills/              # 技能库
│   └── 📂 planners/            # 规划器
├── 📂 capabilities/            # 机器人/物体能力定义
├── 📂 configs/                 # 配置文件
├── 📂 schemas/                 # JSON Schema
├── 📂 web_demo/                # 前端界面
├── 📂 external/                # 第三方资源
├── 📄 requirements-mujoco.txt  # Python依赖
└── 📄 README.md               # 本文件
```

### 🎨 Web界面

启动服务后访问 `http://127.0.0.1:8765`，提供：

- ✅ 自然语言任务输入
- ✅ 实时进度追踪
- ✅ MuJoCo渲染回放
- ✅ 实验报告可视化
- ✅ 失败案例分析

### 🔬 技术亮点

| 技术 | 说明 |
|------|------|
| **MuJoCo 3.3+** | 高性能物理仿真引擎，支持接触、摩擦、柔性体 |
| **Jacobian IK** | 阻尼最小二乘逆运动学，自适应阻尼参数 |
| **LLM集成** | OpenAI/Anthropic兼容API，本地fallback |
| **声明式能力** | JSON配置机器人/物体/工具，易于扩展 |
| **实验记忆** | JSONL格式存储，支持历史追踪和收敛检测 |

### 🛠️ 开发指南

#### 添加新机器人

1. 在 `capabilities/robots.json` 添加机器人定义
2. 在 `external/mujoco_menagerie/` 放置MJCF模型
3. 在 `robot_registry.py` 注册机器人规格

#### 添加新任务

1. 在 `tasks/` 目录创建任务模块
2. 实现 `TaskSpec` 和执行逻辑
3. 在 `tasks/registry.py` 注册任务

#### 添加新技能

1. 在 `skills/library.py` 定义技能原语
2. 实现运动规划逻辑
3. 在 `motion_primitives.py` 集成

### 📊 实验报告示例

```
═══════════════════════════════════════════════════════
实验报告 - FR3抓取放置任务
═══════════════════════════════════════════════════════
总运行次数: 27
成功率: 40.7% (11/27)

失败分析:
├─ 滑落失败: 8次 (29.6%)
├─ 未接触: 5次 (18.5%)
└─ 位移不足: 3次 (11.1%)

下一轮建议:
→ 增大抓取高度补偿 (+2cm)
→ 降低摩擦系数至 low
→ 调整物体偏移至 medium
═══════════════════════════════════════════════════════
```

### 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 了解详情

### 🙏 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真引擎
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - 机器人模型库
- [Deep Potential](https://deepmodeling.org/) - 追光计划支持

---

## English

### ✨ Key Features

- **🧠 Intelligent Task Parsing**: Natural language input (Chinese/English), LLM-powered structured task specification
- **🔧 Dynamic Scene Assembly**: Automatic MJCF model loading, intelligent object/tool matching
- **⚡ Real Physics Simulation**: MuJoCo engine, Jacobian IK motion planning, contact/friction/gravity
- **📊 Smart Experiment Design**: Auto-generated variable sweep matrices, hypothesis-driven design
- **🔍 Failure Attribution**: Multi-dimensional failure classification, physics-based criteria
- **🔄 Iterative Optimization**: Historical tracking, variable space narrowing, convergence detection

### 🚀 Quick Start

```bash
git clone https://github.com/HANRAN808/MuJoCo-Agent-Lab.git
cd MuJoCo-Agent-Lab
pip install -r requirements-mujoco.txt
python -m mujoco_bridge.server
```

Visit `http://127.0.0.1:8765` for the web interface.

### 📡 API

```bash
# NLP Pipeline (Recommended)
POST /api/nlp/run
{
  "goal": "Use FR3 to pick up the block and place it on the left",
  "limit": 27
}
```

---

<div align="center">

**Built with ❤️ for the Pursuit of Light Plan (追光计划)**

[![Deep Potential](https://img.shields.io/badge/Deep%20Potential-追光计划-blue)](https://deepmodeling.org/)

</div>
