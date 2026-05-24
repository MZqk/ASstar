## Codex Rules

- 默认不使用 Superpower/superpowers/brainstorming/TDD/完整开发流程；只有用户明确要求才用。
- 小改动、配置、命令、日志/排障解释、README 小修：禁止启用 Superpower。
- 默认最小改动；不重构无关模块；不修改 `release/`、`packages/` 第三方二进制/安装包，除非用户明确要求。

## Project Guardrails

- 离线运行链路优先：可回归、可打包、可运行优先于风格优化。
- 变更前按需读取文档：用户/使用行为看 `README.md`；打包/runtime/验证看 `INTEGRATION_README.md`；pipeline 算法看 `pipeline/AGENTS.md`。
- 影响范围：`pipeline/` 算法，`gui/` 运行与预检，`build/` 打包，`resources/` 模板/说明。
- Siril/StarNet 相关必须保持兼容：`SIRIL_PYTHON_CLI`、runtime home、`SirilPythonSeed` 恢复、StarNet 架构/可执行校验。
- 调整 `pipeline/seestar_Superimpose.py` stage 1-10 顺序时，回复中必须说明行为变化并同步用户文档。

## Docs

- `README.md`：用户说明、使用方式、常见构建命令。
- `INTEGRATION_README.md`：内部集成、打包、runtime、插件、验证矩阵。
- `AGENTS.md`：全仓约束。
- `pipeline/AGENTS.md`：pipeline/Stage11 约束。
- 避免复制长段事实；更新对应单一来源，其他文档只留摘要或引用。

## Validation

- Python 改动：`python3 -m py_compile <changed_py_files>`。
- GUI/pipeline 调度：检查路径拼接、preflight 覆盖、失败日志。
- build 脚本：`bash -n build/build_macos_app.sh`。
- 无法完整验证时，回复写明未验证项、原因、风险。

## Reply

- 说明改动文件、核心行为变化、已验证项、剩余风险/下一步。
