---
tags:
  - claude-code
  - skills
  - tools
date: 2026-06-22
---

# Claude Code 已安装 Skills 列表

> 记录时间: 2026-06-22

## Obsidian Skills（kepano/obsidian-skills）

| Skill | 功能描述 |
|-------|----------|
| **obsidian-markdown** | 创建/编辑 Obsidian 风格 Markdown（wikilinks、embeds、callouts、properties 等） |
| **obsidian-cli** | 通过 CLI 与运行的 Obsidian 实例交互（创建/读取/搜索笔记、属性管理、插件开发） |
| **obsidian-bases** | 创建 .base 文件，支持 filters、formulas、table/cards/list/map 视图 |
| **json-canvas** | 创建/编辑 .canvas 文件（JSON Canvas Spec 1.0），支持 text/file/link/group 节点和连线 |
| **defuddle** | 从网页提取干净的 Markdown 内容，去除导航/广告等干扰 |

## oh-my-claudecode Skills

### 工作流

| Skill | 功能描述 |
|-------|----------|
| oh-my-claudecode:autopilot | 自动驾驶模式，自动完成多步骤任务 |
| oh-my-claudecode:ultrawork | 超级工作模式 |
| oh-my-claudecode:ultraqa | 超级 QA 测试模式 |
| oh-my-claudecode:ultragoal | 超级目标追踪模式 |
| oh-my-claudecode:ralph | Ralph 循环执行模式 |
| oh-my-claudecode:ralplan | Ralph 计划模式 |
| oh-my-claudecode:team | 多 Agent 团队协作 |
| oh-my-claudecode:omc-teams | OMC 团队管理 |
| oh-my-claudecode:ccg | CCG 模式 |

### 研究与分析

| Skill | 功能描述 |
|-------|----------|
| oh-my-claudecode:deep-research | 深度研究——多源搜索、事实核查、生成引用报告 |
| oh-my-claudecode:deep-dive | 代码库深度分析 |
| oh-my-claudecode:deep-interview | 深度访谈模式 |
| oh-my-claudecode:autoresearch | 自动研究模式 |
| oh-my-claudecode:sciomc | 并行科学家 Agent 综合分析 |
| oh-my-claudecode:external-context | 外部文档搜索和查找 |

### 开发与调试

| Skill | 功能描述 |
|-------|----------|
| oh-my-claudecode:debug | 调试模式 |
| oh-my-claudecode:plan | 计划模式 |
| oh-my-claudecode:verify | 验证模式 |
| oh-my-claudecode:code-review | 代码审查 |
| oh-my-claudecode:ai-slop-cleaner | AI 代码清理 |
| oh-my-claudecode:self-improve | 自我改进模式 |
| oh-my-claudecode:learner | 学习模式 |
| oh-my-claudecode:skillify | 将工作流转换为 Skill |
| oh-my-claudecode:trace | 证据驱动的因果追踪 |

### 项目管理

| Skill | 功能描述 |
|-------|----------|
| oh-my-claudecode:release | 发布助手 |
| oh-my-claudecode:wiki | LLM Wiki——持久化知识库 |
| oh-my-claudecode:project-session-manager | 项目会话管理 |
| oh-my-claudecode:writer-memory | 写作记忆 |
| oh-my-claudecode:remember | 记忆管理 |
| oh-my-claudecode:visual-verdict | 可视化判断 |

### 配置与工具

| Skill | 功能描述 |
|-------|----------|
| oh-my-claudecode:omc-setup | OMC 安装/刷新 |
| oh-my-claudecode:setup | OMC 设置 |
| oh-my-claudecode:mcp-setup | MCP 服务器配置 |
| oh-my-claudecode:configure-notifications | 通知集成配置（Telegram、Discord、Slack） |
| oh-my-claudecode:hud | HUD 显示配置 |
| oh-my-claudecode:deepinit | 深度初始化 |
| oh-my-claudecode:omc-doctor | OMC 诊断 |
| oh-my-claudecode:omc-reference | OMC Agent 目录和参考 |
| oh-my-claudecode:skill | Skill 管理 |
| oh-my-claudecode:ask | 问答模式 |
| oh-my-claudecode:cancel | 取消所有活跃 OMC 模式 |
| oh-my-claudecode:local-build-reminder | 本地构建提醒 |

## 内置 Skills

| Skill | 功能描述 |
|-------|----------|
| update-config | 配置 Claude Code settings.json |
| keybindings-help | 键盘快捷键自定义 |
| verify | 验证代码变更 |
| code-review | 代码审查 |
| simplify | 代码简化 |
| fewer-permission-prompts | 减少权限提示 |
| loop | 定时循环执行任务 |
| claude-api | Claude API / Anthropic SDK 开发 |
| run | 启动应用验证变更 |
| init | 项目初始化 |
| review | Pull Request 审查 |
| security-review | 安全审查 |

## 安装来源

- **Obsidian Skills**: 来自 https://github.com/kepano/obsidian-skills，通过 
 ERROR  Missing required argument: source

  Usage:
    npx skills add <source> [options]

  Example:
    npx skills add vercel-labs/agent-skills 全局安装
- **oh-my-claudecode Skills**: 随 oh-my-claudecode 插件安装
- **内置 Skills**: Claude Code 自带