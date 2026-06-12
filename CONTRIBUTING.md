# 贡献指南

感谢你对 MuJoCo Agent Lab 的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告 Bug

1. 在 [Issues](https://github.com/HANRAN808/MuJoCo-Agent-Lab/issues) 中创建新issue
2. 使用bug报告模板
3. 提供详细的复现步骤和环境信息

### 提交新功能

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 添加新机器人

1. 在 `capabilities/robots.json` 中添加机器人定义
2. 将MJCF模型放入 `external/mujoco_menagerie/`
3. 在 `robot_registry.py` 中注册机器人规格
4. 更新README中的支持机器人列表

### 添加新任务

1. 在 `tasks/` 目录创建新模块
2. 实现 `TaskSpec` 数据类和执行逻辑
3. 在 `tasks/registry.py` 中注册任务
4. 添加相应的测试

### 添加新技能

1. 在 `skills/library.py` 中定义技能原语
2. 实现运动规划逻辑
3. 在 `motion_primitives.py` 中集成

## 代码规范

- 遵循 PEP 8 风格指南
- 使用类型注解
- 为所有公共函数添加文档字符串
- 保持代码简洁明了

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` Bug修复
- `docs:` 文档更新
- `style:` 代码格式（不影响功能）
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具相关

## 许可证

贡献即表示你同意你的代码在 MIT 许可证下发布。

## 问题反馈

如有任何问题，请通过以下方式联系：

- GitHub Issues: [创建Issue](https://github.com/HANRAN808/MuJoCo-Agent-Lab/issues)
- Email: 1120231688@bit.edu.cn

感谢你的贡献！🎉
