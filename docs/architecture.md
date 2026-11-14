# 架构与分层规则

v2 把 CodePlan 里的模块调用约束收口成一份**机器可读的分层清单**
[`docs/architecture/layers.yaml`](architecture/layers.yaml) 和一个基于 AST 的静态检查器
[`tools/check_layering.py`](../tools/check_layering.py)。本文件说明 R1-R3 的含义、
清单结构、检查器用法、例外登记流程与当前基线。

> 检查器只读源码，不导入 torch，也不执行被检查的模块，秒级完成。
> `tests/test_architecture` 已把"仓库零违规"纳入 pytest 门禁；命令行检查器建议在提交前单独执行。

相关文档：[CodePlan 第四章 R1-R7](../CodePlan.md)、[DVX 标准数据集](dataset_spec.md)。

---

## 1. 层次编号

`level` **数字越小越靠上层**，数字越大越底层；依赖方向只能自上而下。

| level | name | 模块 | 职责 |
|---|---|---|---|
| 0 | `assembly` | `models/detectors`、`models/classifiers`、`models/segmenters` | 组合层：把骨干/颈部/头部组装成可用模型 |
| 1 | `component` | `models/heads`、`models/losses`、`models/backbones`、`models/necks` | 组件层：头、损失、骨干、颈部 |
| 2 | `foundation` | `models/layers`、`models/postprocess` | 基础算子与后处理原语 |
| 3 | `shared` | `models/base` | 共享基类/契约 |
| 4 | `infrastructure` | `registry`、`errors`、`utils` | 注册表、异常族、通用工具 |

只有**能映射到清单层级**的模块才参与规则判定；`data/`、`training/`、`config/`、
`metrics/`、`export/` 等不在清单里，检查器会跳过（见 §7）。

---

## 2. 三条规则（R1-R3）

### R1：依赖单向，只允许 import 同级或更底层

目标模块的 `level` 必须 **≥** 源模块的 `level`；`target.level < source.level` 即"向上依赖"，
属于违规。

```text
allowed:     detectors(0) -> heads(1) -> layers(2) -> registry(4)
violation:   layers(2) -> heads(1)          # foundation 反向依赖组件层
```

例外：**目标为 level 3/4（`shared` / `infrastructure`）时 R1 始终放行**。这两个层是所有层
公共底座，level 4 的模块 import level 3 不算"向上依赖"；但同层、不同 unit 之间仍受 R2 约束。

### R2：同层、不同组件包之间禁止互相 import

两个模块 `level` 相同时，若它们属于清单中的**不同 unit**（例如 `heads` 与 `losses`、
`backbones` 与 `necks`），则禁止互相 import。

```text
violation:   heads/xxx.py -> losses/yyy.py          # 组件层内部跨包
violation:   detectors/a.py -> classifiers/b.py     # 组合层内部跨包
deferred:    heads/xxx.py -> heads/yyy.py           # 同包，转由 R3 判定（默认违规，需例外）
```

规则意图是强制"同级隔离"：共享实现要么下沉到 `layers/` / `postprocess.py`，
要么通过 Registry 注入，而不是横向直接依赖。

### R3：组件包内部的兄弟 import 必须登记例外

在 `component` 层的**包**内部，非 `__init__.py` 的模块互相 import 属于违规，
除非命中 `layers.yaml` 的 `exceptions`。

- 只作用于 `component` 层（`head` / `losses` / `backbones` / `necks`）；
  `assembly` / `foundation` / `shared` / `infrastructure` 不适用。
- `__init__.py` 豁免 R3：聚合导出不算兄弟模块互相依赖（R1/R2 仍会判定）。
- 规则只约束"非 `__init__.py` 的源文件"；目标是 `__init__.py` 时仍会按普通目标判定。
- 合理的例外必须来自真实代码：共享基类、子类继承、子包内公共支撑模块等；
  不允许为了"让检查通过"而编造例外。

当前登记的例外见 §5。

### R1-R3 与 CodePlan R1-R7 的对应关系

CodePlan 第四章有七条规则（R1-R7），本检查器编号的是其中可静态判定的三条，
**两套编号并不一一对应**，读代码评审意见时请注意：

| 检查器规则 | 大致对应 CodePlan | 说明 |
|---|---|---|
| R1 | CodePlan R1（依赖单向） | 机器可判定 |
| R2 | CodePlan R2（同级隔离）的前半段 | 机器可判定 |
| R3 | CodePlan R2 的"子包内支撑模块例外"要求 | 把"先登记再复用"变成门禁 |
| — | CodePlan R3（职责边界） | 语义约束，仍需人工评审 |
| — | CodePlan R4-R7（decode / Registry / 新增组件 / 组合器子包） | 人工评审 + 专项测试 |

不要因为"R1-R3 通过了"就认为 R4-R7 也满足。

---

## 3. `layers.yaml` 结构

分层的**成员归属与例外**以清单为唯一依据；规则语义由检查器实现（`_evaluate_rules`），
`rules` 段只是人类可读说明，不会被解析。完整内容见
[`docs/architecture/layers.yaml`](architecture/layers.yaml)，结构如下：

```yaml
version: 1
description: "DVisionix 分层与 import 规则（R1-R3 的机器可读版本）"

layers:
  - level: 0
    name: assembly
    packages: [dvisionix.models.detectors, dvisionix.models.classifiers, dvisionix.models.segmenters]
  - level: 1
    name: component
    packages: [dvisionix.models.heads, dvisionix.models.losses,
               dvisionix.models.backbones, dvisionix.models.necks]
  - level: 2
    name: foundation
    packages: [dvisionix.models.layers]
    modules: [dvisionix.models.postprocess]
  - level: 3
    name: shared
    modules: [dvisionix.models.base]
  - level: 4
    name: infrastructure
    modules: [dvisionix.registry, dvisionix.errors, dvisionix.utils]

rules:
  R1: "只允许 import 同级或更底层（目标 level >= 自身 level）"
  R2: "同层、不同组件包之间禁止互相 import"
  R3: "组件包内部，非 __init__.py 的兄弟模块互相 import 必须登记例外"

exceptions:
  - from_glob: "dvisionix/models/backbones/*.py"
    to: "dvisionix/models/backbones/feature.py"
    rule: R3
    reason: "共享基类 FeatureBackboneBase（CodePlan R2 例外）"
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `version` | 是 | 固定为 `1`，否则检查器以退出码 2 失败 |
| `layers[*].level` | 是 | 整数；0 最上层，数字越大越底层 |
| `layers[*].name` | 是 | 层语义名；`component` 这个名字会启用 R3 |
| `layers[*].packages` | 二选一 | 包名列表；匹配"包本身及其任意子模块" |
| `layers[*].modules` | 二选一 | 模块名列表；匹配"模块本身及其子模块"（`dvisionix.utils` 实际是包，也可放这里） |
| `rules` | 否 | 人类可读的规则说明；检查器不解析其内容（规则语义在代码中实现） |
| `exceptions[*].from` / `from_glob` | 二选一 | 源文件路径（相对仓库根，posix，含 `.py`）或通配 |
| `exceptions[*].to` / `to_glob` | 二选一 | 目标文件路径或通配 |
| `exceptions[*].rule` | 是 | `R1` / `R2` / `R3` |
| `exceptions[*].reason` | 建议 | 为什么允许；应当能指向 CodePlan 的既有例外或说明架构原因（检查器允许为空，但评审要求非空） |

匹配细节：

- 一个模块可以匹配多个 `packages`/`modules` 条目时，取 **unit 字符串最长**（最具体）的那条。
- 清单里同一个 unit 重复登记会直接报错（退出码 2）。
- 例外路径用 `fnmatch` 语义：`**` 会先替换为 `*`；`*` 可以跨越 `/`。
- 一条 import 只命中**第一条**匹配的例外，其余不重复计数。
- 例外的 `hits` 会计入 `--json` 的 `exceptions_used`；没有命中的例外不会报错，
  但说明它可能已经过期，应清理。

---

## 4. 检查器用法

```bash
python tools/check_layering.py [--root DIR] [--manifest FILE] [--json] [--no-fail]
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--root DIR` | 脚本上两级（仓库根） | 待检查仓库根目录；`dvisionix/` 必须在其下 |
| `--manifest FILE` | `<root>/docs/architecture/layers.yaml` | 分层清单路径 |
| `--json` | 关闭 | 输出机器可读 JSON（stdout 只有 JSON） |
| `--no-fail` | 关闭 | 有违规也返回 0，仅用于报告/统计，**不要**放进 CI 门禁 |

### 4.1 人类可读输出

```bash
$ python tools/check_layering.py
分层检查：/path/to/DVisionix/dvisionix
分层清单：/path/to/DVisionix/docs/architecture/layers.yaml
扫描文件：188 个
命中登记例外：4 条 / 共 20 次
[通过] 未发现分层违规。
```

有违规时按「文件:行 → 目标模块 → 规则」逐条列出，并在末尾给出失败总数与清单路径：

```text
文件:行                                                         目标模块                                                           规则
----------------------------------------------------------------------------------------------------
dvisionix/models/layers/foo.py:3                                dvisionix.models.heads.bar                                         R1
    说明：向上依赖：dvisionix.models.layers（level 2） -> dvisionix.models.heads（level 1）
```

### 4.2 JSON 输出

`--json` 的顶层字段固定为 `violations` / `checked_files` / `exceptions_used`：

```json
{
  "violations": [
    {
      "file": "dvisionix/models/layers/foo.py",
      "line": 3,
      "target": "dvisionix.models.heads.bar",
      "target_file": "dvisionix/models/heads/bar.py",
      "rule": "R1",
      "message": "向上依赖：dvisionix.models.layers（level 2） -> dvisionix.models.heads（level 1）"
    }
  ],
  "checked_files": 188,
  "exceptions_used": [
    {
      "rule": "R3",
      "from_glob": "dvisionix/models/backbones/*.py",
      "to": "dvisionix/models/backbones/feature.py",
      "reason": "共享基类 FeatureBackboneBase（CodePlan R2 例外）",
      "hits": 10
    }
  ]
}
```

注意：`--json` **不会**把退出码改成 0；CI 里应同时检查退出码。

### 4.3 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 无违规；或指定了 `--no-fail` |
| `1` | 存在违规（且未指定 `--no-fail`） |
| `2` | 清单不存在/格式非法、待检查的包目录不存在，或命令行参数错误 |

### 4.4 门禁测试

```bash
python -m pytest tests/test_architecture -q
```

[`tests/test_architecture/test_layering.py`](../tests/test_architecture/test_layering.py) 包含：

- 对真实仓库跑检查器，要求 `violations == []`；
- 在 `tmp_path` 造迷你包，自测 R1 / R2 / R3 能被抓到；
- 登记例外后放行、相对导入被正确绝对化、CLI 的 `--json` / `--no-fail` / 退出码符合契约。

建议本地提交前执行：

```bash
python tools/check_layering.py && python -m pytest tests/test_architecture -q
```

---

## 5. 当前基线

2026-11-14 实测（`python tools/check_layering.py --json`）：

| 指标 | 数值 |
|---|---|
| 扫描文件 | **188**（早期记录值 181；以本次实测为准） |
| 违规 | **0** |
| 登记例外 | **4 条 / 共命中 20 次** |

例外明细（全部为 R3）：

| from | to | 命中 | 原因 |
|---|---|---|---|
| `dvisionix/models/backbones/*.py` | `dvisionix/models/backbones/feature.py` | 10 | 共享基类 `FeatureBackboneBase`（CodePlan R2 例外） |
| `dvisionix/models/heads/detection/nmsfree_yolo.py` | `dvisionix/models/heads/detection/yolo.py` | 1 | `NMSFreeYOLOHead` 继承 `YOLOHead`（CodePlan R2 例外） |
| `dvisionix/models/losses/**/*.py` | `dvisionix/models/losses/**/*.py` | 3 | 损失子包内共享 `base` / `box_loss` / `matcher` / `assigner` |
| `dvisionix/models/losses/*.py` | `dvisionix/models/losses/*.py` | 6 | 顶层损失模块共享 `base` / `classification` / `segmentation` |

> 关于 `188`：`checked_files` 会随代码文件增加而变化；早期记录曾写作 `181`。
> 本文以当前仓库 **实测值 188** 为准（违规 0、例外命中 20 未变），可随时用
> `--json` 的 `checked_files` 复查。

---

## 6. 例外登记流程

**先登记，再改代码**。新增例外必须与引入它的代码在同一个变更里，评审时同时看
`layers.yaml` 的 diff 与实现。

1. **先判断能否通过下沉解决**：优先把共享实现下沉到 `layers/` / `postprocess.py`，或改为
   Registry 注入。例外是兜底，不是常规手段。
2. **在 `layers.yaml` 的 `exceptions` 中登记**：
   - `from` / `from_glob` 二选一，`to` / `to_glob` 二选一，路径相对仓库根；
   - `rule` 写实际违反的规则；
   - `reason` 写清"为什么必须横向依赖"，能引用 CodePlan 的既有例外就引用；
   - 通配范围尽量小：能用精确路径就不要用 `*.py`，更不要写 `**/*.py` 级别的宽通配。
3. **跑检查器**：确认 `violations == []`，且新例外的 `hits > 0`（否则说明路径写错或已过期）。
4. **跑门禁测试**：`python -m pytest tests/test_architecture -q`。
5. **在变更说明/CHANGELOG 中记录**（若例外改变了架构约定）。

补充约定：

- 新增 **R2 例外**（跨组件包直接依赖）必须尤其谨慎：R2 的本意就是禁止横向依赖，
  登记时必须在 `reason` 中说明为什么 Registry / 下沉 / 接口抽象都不可行。
  当前清单中没有 R2 例外。
- 新增 **R1 例外**（向上依赖）原则上不允许；目标为 level 3/4 的共享层已由规则天然放行，
  不应再登记。
- 删除代码后记得清理失效例外；检查器不会主动报"未使用例外"。

---

## 7. 检查范围与限制

- 只扫描 `dvisionix/**/*.py`；`tools/`、`tests/` 不参与分层判定。
- 只检查**能映射到 `layers.yaml`** 的模块。`data/`、`training/`、`config/`、`metrics/`、
  `export/`、`models/__init__.py` 等当前不在清单中，因此完全跳过。
- 只做 AST 静态解析：`importlib` / `__import__` 等动态导入、条件导入不会被发现。
- 相对导入会先解析为绝对模块名再判定；无法解析的越界相对导入会被跳过（Python 本身也会报错）。
- 源文件读取失败或语法错误只打印警告并跳过，不会让检查失败——语法错误应由
  `ruff` / `black` / 测试单独拦截。
- 不检查 CodePlan R4-R7 的语义约束（decode 策略、Registry 构建、新增组件流程、组合器目录），
  这些仍依赖评审与对应测试。
- 不检查例外是否仍必要；`exceptions_used` 只列出命中的例外，未命中的需人工清理。
- 检查器自身不 import torch，也不依赖项目运行环境（只用 `ast` + `PyYAML`），可放在 CI 最前；
  当前尚未单独接入 CI 脚本，零违规由 `tests/test_architecture` 在 pytest 中兜底。
