# Ravelin

**让执行流持续演进，让产品始终属于你。**

Ravelin 为产品所需的执行流添加可独立采用的生产机制，同时让执行流和产品语义继续留在
应用中。它们让运行时资源可移植，让用户重新接入仍在进行的工作而永久产品历史由应用
持有，让选定工作可恢复，并让待处理输入跨越 Worker 生命周期。

根据你面对的边界选择机制：

- **Assets**：让序列化后的组件保存逻辑引用，而不是在线 Provider client、endpoint
  和凭证。每个环境把引用解析成 chat 或 embedding 模型服务、向量数据库或工具服务。
  只要替代资源满足相同的组件契约，组件连接和控制流就可以保持不变。
- **Streaming 与应用自有 Journal**：区分临时的实时传输、有限窗口内的重连、进行中
  生成的 catch-up，以及应用选择长期保存的产品历史。
- **持久化执行与外部介入**：把显式声明的组件和步骤边界接入 Restate，在恢复时复用
  已记录的结果，并让工作在不保留原进程的情况下等待人工或外部系统。

每种机制都可以在对应的边界出现时单独采用：

| 当产品发生变化…… | 随之出现的边界…… | 按需接入…… |
| --- | --- | --- |
| 同一执行流进入另一个环境、租户或 Provider | 在线部署资源不应成为序列化执行流结构的一部分 | Assets |
| 执行在初始请求结束后继续 | 重连客户端可能落到另一个副本 | 共享 Streaming 与 catch-up |
| 已完成的模型或工具调用不应重复付费 | 恢复需要声明可复用的执行单元 | Restate-backed components 或显式 steps |
| 人或外部系统稍后才给出答案 | 待处理的 continuation 必须跨越 Worker 生命周期 | 持久化等待与 resolve |
| 结果成为产品记录、分支或审计历史 | 实时传输与恢复状态不应决定长期数据模型 | 应用自有 Journal hooks |

组合使用时，每种机制仍然承担不同角色：Assets 提供运行时依赖和 Provider 行为；
Restate 保护显式声明的工作；Streaming 与 Journal hooks 暴露生命周期和活动。应用仍然
决定 run、branch、join、decision 和长期记录各自意味着什么。

你可以在 [从原型到生产](docs/user-guide/from-prototype-to-production.rst) 中沿着 Provider
切换、并行工作、浏览器重连、Worker 替换和延迟审批走完这些变化。

同一组边界也可以服务于不同产品：小型 tool loop、带分支的 review workflow 和并行
research flow 可以共享资源解析、传输、局部恢复和持久化输入，而不必共享同一种 thread、
checkpoint 或产品历史模型。

**当前集成。** Ravelin 通过 Haystack 实现这些接合点；Haystack 的组件和图提供执行
脚手架，同时把执行流语义和服务生命周期留给应用。当前面向组件的 API、扩展数据模型
和持久化 Pipeline adapter 都针对 Haystack。每增加一种执行底座，都需要完成专门集成。

## 安装

Ravelin 需要 Python 3.10 或更高版本。

```bash
pip install ravelin
```

仅在需要相应集成时安装可选依赖：

```bash
pip install "ravelin[milvus]"
pip install "ravelin[redis]"
pip install "ravelin[restate]"
```

## 快速开始：可移植的运行时资源

Assets 是采用 Ravelin 最小的独立路径。这个例子不需要 Streaming 后端或
持久化执行器：它只展示序列化组件如何保留逻辑资源身份，同时由每个环境提供
实际运行时实现。

先把当前部署使用的模型服务写入 `assets/primary-chat-model.json`。凭证仍然是环境
变量引用，而不是明文值：

```json
{
  "name": "primary-chat-model",
  "type": "model-service",
  "configs": {
    "inference_api_format": "openai-chat",
    "base_url": "https://api.openai.com"
  },
  "secrets": {
    "api_key": {
      "type": "env_var",
      "env_vars": ["OPENAI_API_KEY"],
      "strict": true
    }
  }
}
```

再把这项资源加载到普通的 Haystack 组件中：

```python
from ravelin.assets.asset_factory import ModelServiceFactory
from ravelin.components.generators import ChatGenerator
from ravelin.dataclasses import ChatMessage
from ravelin.utils.asset_storage import set_local_asset_storage_path


set_local_asset_storage_path("./assets")
model_service = ModelServiceFactory.load_from_local("primary-chat-model")
generator = ChatGenerator(model_service=model_service, model="gpt-5.6-terra")

# 持久化的是结构，而不是在线 client 或已解析的凭证。
serialized = generator.to_dict()

# 反序列化时，当前环境会重新解析这项逻辑资源。
restored = ChatGenerator.from_dict(serialized)
result = restored.run([ChatMessage.from_user("为什么要让运行时资源可移植？")])
print(serialized["init_parameters"]["model_service"])
print(result["replies"][0].text)
```

组件保存的是 `AssetReference`，因此序列化后的 Pipeline 结构不会吸收在线服务对象或
凭证。每个环境可以为同一名称提供不同的本地定义。若要从其他配置系统加载，序列化的
引用需指向已注册的 resolver source。

基础流程见[Assets 与运行时服务](docs/user-guide/assets.rst)，完整协议见
[Asset 解析与 Secret 契约](docs/user-guide/asset-resolution.rst)。

## 当执行离开单次请求

更难的路径始于同一次逻辑执行跨越多个生命周期：浏览器断开时生成仍在继续；
Worker 可能在完成昂贵步骤后消失；人工回复可能在原进程退出后才到达。实时传输、
执行恢复、待决人工等待和永久产品历史因此需要关联身份，却不能被合并成同一个
存储或生命周期。

[从原型到生产](docs/user-guide/from-prototype-to-production.rst)按边界走过重连、进程故障和
延迟审批。这些机制仍可以独立采用：

| 当你的应用需要…… | Ravelin 提供…… | 你继续控制…… |
| --- | --- | --- |
| 让同一个图运行在不同 Provider 或环境中 | [Asset 引用、resolver、Secret 与运行时实现](docs/user-guide/asset-resolution.rst) | 资源选择与发布策略 |
| 在执行已与客户端请求解耦后，让实时 UI 跨越刷新和多副本重新接入 | [共享 Streaming 后端、cursor replay、进行中生成的 catch-up 和 relay API](docs/user-guide/streaming.rst) | 运行生命周期、产品呈现和 canonical state |
| 保存产品真正需要的历史 | [位于明确执行边界上的 Journal event hooks](docs/user-guide/streaming.rst#the-application-journal) | Schema、保留周期、授权和查询 |
| 在故障或进程替换后恢复显式声明的工作 | [Restate 组件调用和可选的持久化步骤](docs/user-guide/durability.rst) | 工作流含义、commit 策略和外部副作用安全 |
| 等待人工或外部输入 | [持久化等待、resolution primitive 和 notifier hook](docs/user-guide/durability.rst#human-and-external-intervention) | 谁能响应、决定意味着什么、下一步做什么 |

这些是彼此独立的采用路径，而不是成熟度阶段。本地 Pipeline 可以只使用 Assets；应用
服务可以使用共享 Streaming 而不接入 Restate；持久化工作流也可以完全不发布实时事件。

### 当前契约覆盖什么

Ravelin 目前仍属于 beta 阶段。它公开的生产契约边界明确、经过测试：

- **Assets：**序列化组件只保留逻辑 `AssetReference`；新进程会重新解析并验证在线
  资源。环境变量 Secret descriptor 可以跨越序列化边界；解析后的 token 值和 Provider
  client 不可以。
- **Streaming 与 Journal：**仍在保留窗口内的 cursor 会从该事件开始做 inclusive
  replay。实时窗口过期后，订阅会追上仍在进行的 generation snapshot 并继续接收实时
  事件；已完成历史由应用存储接入。在可恢复执行下，Journal 写入还需要幂等 sink 或
  Restate 保护。
- **持久化与外部介入：**进程故障后可以复用已完成的组件调用和显式持久化子步骤。
  未完成组件内部没有受保护的工作可能再次执行；外部副作用仍需要应用级安全契约。

代码库测试了支撑这些陈述的契约边界。Streaming 测试覆盖 late join、
inclusive cursor replay 和进行中 generation catch-up；Restate 测试区分成功或终止完成与
awakeable suspension 或可重试故障，后两者不得关闭当前运行的实时流。参见
[Streaming relay 测试](tests/streaming/relay_test.py)、
[Redis 后端测试](tests/streaming/backends/redis_test.py) 和
[Restate 应用测试](tests/core/restate/app_test.py)。

完整的 Provider 切换、浏览器重连、Worker 故障和审批等待过程，见
[从原型到生产](docs/user-guide/from-prototype-to-production.rst)。

Ravelin 面向将执行流语义和永久产品历史保留在应用中的团队。

每种机制只引入与自身契约相匹配的基础设施：多副本可重连 Streaming 使用 Redis 等共享
后端；永久产品历史保留在应用自有存储中；需要持久恢复或长期等待时再接入 Restate。
[为什么选择 Ravelin](docs/user-guide/why-ravelin.rst)解释了这一选择背后的架构逻辑。

## 文档

- [为什么选择 Ravelin](docs/user-guide/why-ravelin.rst)：在集成式 Runtime、有界的应用
  编排与 Ravelin 可组合机制之间，如何判断生命周期归属。
- [如何选择 Runtime 边界](docs/user-guide/comparison.rst)：当前集成式框架如何回答同样的
  设计问题，以及各自执行契约有何不同。
- [架构与职责边界](docs/user-guide/architecture-boundaries.rst)：应用、执行底座和
  Ravelin 各自负责什么。
- [Assets 与运行时服务](docs/user-guide/assets.rst)：第一个 Asset、引用和 resolver
  的实践流程。
- [Streaming](docs/user-guide/streaming.rst)：实时事件、重连和后端职责。
- [持久化执行](docs/user-guide/durability.rst)：基于 Restate 的恢复和人工介入流程。
- [从原型到生产](docs/user-guide/from-prototype-to-production.rst)：一条组合后的服务路径、
  各存储的职责和故障注入检查。
- [Asset 解析与 Secret 契约](docs/user-guide/asset-resolution.rst)：resolver 注册、
  信任边界和序列化行为。

在本地构建完整文档：

```bash
hatch run docs:build
```

## 开发

项目使用 Hatch 管理开发环境。在仓库根目录安装开发工具并创建环境：

```bash
pip install hatch pre-commit
hatch env create

# 可选：在每次提交前运行已配置的检查
pre-commit install
```

已配置的 hooks 包括完整的测试、覆盖率、类型和代码检查，以及发布日志校验。
Black hook 需要本机提供 Python 3.10。Hatch 创建环境时会安装开发依赖，
并下载英文和中文 spaCy 模型。

```bash
# 单元测试和总覆盖率
hatch run test:unit

# 变更行覆盖率、类型检查和代码检查
hatch run test:diff-cov
hatch run test:types
hatch run test:lint

# 完整验证
hatch run test:all

# 构建源码包和 wheel
hatch build
```

`test:unit` 和 `test:all` 不包含集成测试。使用 `hatch run test:integration`
单独运行 Restate harness 测试；它们需要 Docker 和所需容器镜像的访问条件。
该命令不会覆盖单元测试的覆盖率数据。

先运行单元测试，再检查变更行覆盖率，确保 `coverage.xml` 是最新的。
两项覆盖率检查均要求至少 90%。变更行覆盖率默认与 `origin/master` 比较；
如需使用其他已有的基准 ref，可运行 `hatch run test:diff-cov origin/main`。

代码、注释和贡献讨论使用英文。文档默认使用英文，可同时维护翻译版本。
添加类型注解，并遵循 `pyproject.toml` 中的 Black 格式配置（每行 120 字符）。
使用 `hatch run format` 格式化代码。

## 参与贡献

保持修改范围聚焦，说明修改目的、可观察的行为变化和验证方式。
修复缺陷时添加回归测试，行为变化时更新相关文档。
使用 [Issue 模板](.github/ISSUE_TEMPLATE/)和
[PR 模板](.github/pull_request_template.md)组织报告和评审内容。

Commit 信息遵循 Conventional Commits：
`<type>[optional scope]: <description>`，例如 `fix(streaming): ...`。
不兼容变更使用 `!` 标记，例如 `feat!: ...`，并说明影响。

涉及用户可见变化时，在 `releasenotes/notes/` 中添加发布日志：

```bash
hatch run reno new <change-description>
hatch run reno lint
```

先填写生成的日志，再运行校验命令。仅文档或内部维护的修改，除非影响用户，
否则不要求发布日志。

## 许可证

Ravelin 使用 [Apache License 2.0](LICENSE.txt)。
