# DVisionix v2 —— P0/P1 实现契约（Lead 制定，实现方必须遵守）

> 状态：**生效**。本文件是 P0/P1 阶段所有实现的唯一接口事实来源。
> 实现方不得擅自改动本文件定义的公开接口；如需变更，先报告 Lead。
> 环境：统一使用 `/data/lzy/Env/condaEnvs/dvisionix/bin/python`（下称 `$PY`）。
> 代码风格：UTF-8 无 BOM；文件头沿用现有风格（`# -*- coding: utf-8 -*-` + 作者/用途）；中文注释与 docstring；
> ruff line-length 100（`E,F,W,I`）、black 100。提交前必须跑 `$PY -m ruff check` 与 `$PY -m black --check`。

---

## 0. 全局约束

1. **不得修改**本契约未列出的既有公开 API 行为；既有 531 条测试必须保持全绿。
2. **不得新增第三方依赖**：只允许已安装的 torch / numpy / opencv / pyyaml / pillow（及项目已有的 extras）。
3. 所有异常必须继承 `dvisionix.errors.DVisionixError`（P0 建立），不要直接 `raise ValueError` 表示领域错误。
4. 所有路径在内存中统一为**绝对路径字符串**（`os.path.abspath`），对外展示用 posix 风格的相对路径。
5. 任何"自动生成/自动推断"的行为必须**显式打印**，不得静默修改磁盘。
6. 每个模块必须有对应单元测试；测试不得依赖 `project/data`（用 `tmp_path` 造夹具）；真实数据验收测试单独放并 `skipif`。

---

# P0 基线

## P0-1 `dvisionix/errors.py`（新增）

```python
# -*- coding: utf-8 -*-
"""DVisionix 统一异常族。

所有领域错误都继承 DVisionixError，便于调用方统一捕获；
每条消息应包含"发生了什么 + 应该怎么办"（hint）。
"""

class DVisionixError(Exception):
    def __init__(self, message: str, hint: str | None = None) -> None: ...
    message: str
    hint: str | None
    # __str__: 有 hint 时返回 f"{message}\n提示：{hint}"

class ConfigError(DVisionixError): ...
class RegistryError(DVisionixError): ...
class DatasetError(DVisionixError): ...
class DatasetSpecError(DatasetError): ...
class AnnotationError(DatasetError): ...
class SplitError(DatasetError): ...
class TaskModelMismatchError(DVisionixError): ...
class TaskDataMismatchError(DVisionixError): ...
class ModelOutputMismatchError(DVisionixError): ...
class CheckpointError(DVisionixError): ...
class ExportError(DVisionixError): ...
```

要求：`__all__` 完整；每个类一行 docstring；`DVisionixError.__str__` 带 hint；不得依赖 torch。
导出方式：不需要改 `dvisionix/__init__.py`（按子模块直接 import）。

## P0-2 `docs/architecture/layers.yaml`（新增）

分层清单是 R1-R7 的**单一事实来源**。格式：

```yaml
version: 1
description: "DVisionix 分层与 import 规则（R1-R3 的机器可读版本）"

layers:
  - level: 0
    name: assembly            # 组合层
    packages: [dvisionix.models.detectors, dvisionix.models.classifiers, dvisionix.models.segmenters]
  - level: 1
    name: component           # 组件层
    packages: [dvisionix.models.heads, dvisionix.models.losses, dvisionix.models.backbones, dvisionix.models.necks]
  - level: 2
    name: foundation          # 最底层
    packages: [dvisionix.models.layers]
    modules: [dvisionix.models.postprocess]
  - level: 3
    name: shared              # 共享基类/契约
    modules: [dvisionix.models.base]
  - level: 4
    name: infrastructure
    modules: [dvisionix.registry, dvisionix.errors, dvisionix.utils]

rules:
  R1: "只允许 import 同级或更底层（目标 level >= 自身 level）"
  R2: "同层、不同组件包之间禁止互相 import"
  R3: "组件包内部，非 __init__.py 的兄弟模块互相 import 必须登记例外"

exceptions:                   # 必须来自对现有代码的实测，不允许编造
  - from_glob: "dvisionix/models/backbones/*.py"
    to: "dvisionix/models/backbones/feature.py"
    rule: R3
    reason: "共享基类 FeatureBackboneBase（CodePlan R2 例外）"
  - from: "dvisionix/models/heads/detection/nmsfree_yolo.py"
    to: "dvisionix/models/heads/detection/yolo.py"
    rule: R3
    reason: "NMSFreeYOLOHead 继承 YOLOHead（CodePlan R2 例外）"
  - from_glob: "dvisionix/models/losses/**/*.py"
    to_glob: "dvisionix/models/losses/**/*.py"
    rule: R3
    reason: "损失子包内共享 base/box_loss/matcher/assigner（CodePlan R2 例外）"
```

> 上述例外条目已由 Lead 实测确认；实现时用检查器跑一遍，若有遗漏再补并报告。

## P0-3 `tools/check_layering.py`（新增）

CLI：

```
$PY tools/check_layering.py [--root DIR] [--manifest FILE] [--json] [--no-fail]
```

行为：
- `--root` 默认仓库根（脚本上两级）；`--manifest` 默认 `<root>/docs/architecture/layers.yaml`。
- 用 `ast` 解析 `dvisionix/**/*.py` 的 `Import` / `ImportFrom`；相对导入必须正确解析为绝对模块名。
- 只检查**能映射到 manifest 层级**的模块；其余（`dvisionix.data/training/config/metrics/export/models.__init__` 等）跳过。
- 三条规则：
  - R1：`target_level < source_level` → 违规（向上依赖；**level 数值越小越靠上层**）。
    目标为 level 3/4 的共享层/基础设施层时始终允许（它们的数值更大，天然满足）。
    > 勘误：本契约初版此处误写为 `source_level < target_level`，方向写反；以本节为准。
  - R2：`source_level == target_level` 且二者不同"组件包"（component 层的 heads/losses/backbones/necks 互为不同组件包）→ 违规。
  - R3：同一组件包内，非 `__init__.py` 的兄弟模块互相 import → 违规，除非命中 exceptions。
- 例外匹配：`from` / `from_glob` 与 `to` / `to_glob` 同时支持；`**` 与 `*` 用 `fnmatch` 语义（`**` 先替换为 `*`）。
- 输出：人类可读表格（文件:行 → 目标 → 违反规则），`--json` 输出 `{"violations": [...], "checked_files": N, "exceptions_used": [...]}`。
- 退出码：有违规且非 `--no-fail` → 1；否则 0。
- 脚本自身不得 import torch（保持秒级）。

## P0-4 `tests/test_architecture/test_layering.py`（新增）

- `test_repository_has_no_layering_violation`：对真实仓库跑检查器，`violations == []`。
- 至少 4 条**检查器自测**（在 `tmp_path` 造迷你包）：向上依赖被抓、同层跨包被抓、兄弟 import 未登记被抓、命中例外时放行。
- 目录需有 `__init__.py`；marker 用 `@pytest.mark.unit`。

## P0-5 `.gitignore`（修改）

追加（若不存在）：`.ruff_cache/`、`.mypy_cache/`、`.cache/`（已有）、`htmlcov/`、`.tox/`（已有）。

**验收**：`$PY -m pytest tests/test_architecture -q` 通过；`$PY tools/check_layering.py` 退出码 0；既有全量测试仍全绿。

---

# P1 DVX 标准数据集

## P1-0 设计原则（再次强调）

- **一个数据集 = 一个根目录 + 递归扫描**；多级目录只影响路径，不携带语义。
- **所有标注共用一个类别表**（`det` 用 5 个子目录的 20 类并集）。
- **切分非破坏性**：只在内存中构建索引列表，绝不移动/改名原始文件。
- 数据规格描述"有什么标注"，训练目标由训练配置决定（P2 再接线）。

## P1-1 目录与模块

```
dvisionix/data/spec/__init__.py     # 只做 re-export
dvisionix/data/spec/schema.py       # DVXDatasetSpec + load_spec
dvisionix/data/spec/classes.py      # ClassTable
dvisionix/data/spec/scan.py         # SampleRef + scan_samples 调度
dvisionix/data/spec/split.py        # SplitResult + build_splits
dvisionix/data/parsers/__init__.py
dvisionix/data/parsers/folder.py    # folder_per_class 扫描
dvisionix/data/parsers/labelme.py   # labelme 扫描/解析
dvisionix/data/parsers/rasterize.py # polygon -> mask
dvisionix/data/dvx_dataset.py       # DVXDataset
```

## P1-2 `dvx_dataset.yaml` 规范（`schema.py`）

```yaml
version: 1
data_type: classification     # classification | multi_task_classification | detection | segmentation
layout: folder_per_class      # folder_per_class | labelme
root: .                       # 相对 yaml 所在目录；支持绝对路径

# 单标签空间（classification / detection / segmentation 用）
classes: [cat, dog]

# 多任务（multi_task_classification 用，二选一）
tasks:
  - {name: licence, path: T15C_1CCD6_licence, classes: [无, 短牌照, 长牌照]}
  - {name: logo,    path: T15C_1CCD6_logo,    classes: [一体奇瑞标, 嵌入奇瑞标]}

splits:
  mode: ratio                 # ratio | list | dir
  ratio: {train: 0.8, val: 0.2}
  seed: 42
  stratify: class             # class | none；默认 class（分类）/ none（检测、分割）
  list_file: null             # mode=list 时必填（JSON：{"train": ["rel/path"], ...}）
  dir_map: null               # mode=dir 时必填（{"train": "train_data/train", ...}，相对每个 task.path 或 root）

parse:
  recursive: true
  image_ext: [.jpg, .jpeg, .png, .bmp]
  labelme:
    ignore_labels: [__ignore__, __background__]
    image_data: ignore        # ignore | error
    shape_types: {rectangle: detection, polygon: segmentation}

seg_mode: semantic            # semantic | query（本期只实现 semantic）
ignore_index: 255
```

`DVXDatasetSpec` 公共接口（**不得改名**）：

```python
@dataclasses.dataclass
class TaskSpec:
    name: str
    path: str
    classes: list[str]

@dataclasses.dataclass
class SplitConfig:
    mode: str = "ratio"
    ratio: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})
    seed: int = 42
    stratify: str | None = None
    list_file: str | None = None
    dir_map: dict[str, str] | None = None

@dataclasses.dataclass
class ParseConfig:
    recursive: bool = True
    image_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    labelme_ignore_labels: tuple[str, ...] = ("__ignore__", "__background__")
    labelme_image_data: str = "ignore"
    labelme_shape_types: dict[str, str] = ...

@dataclasses.dataclass
class DVXDatasetSpec:
    version: int
    data_type: str
    layout: str
    root: str                       # 绝对路径
    classes: list[str] | None
    tasks: list[TaskSpec] | None
    splits: SplitConfig | None
    parse: ParseConfig
    seg_mode: str = "semantic"
    ignore_index: int = 255
    spec_path: str = ""             # yaml 绝对路径

    @classmethod
    def from_yaml(cls, path: str) -> "DVXDatasetSpec": ...
    def validate(self) -> None: ...           # 失败抛 DatasetSpecError
    def class_names(self, task: str | None = None) -> list[str]: ...
    def num_classes(self, task: str | None = None) -> int: ...
    def resolve_root(self) -> str: ...

def load_spec(path: str) -> DVXDatasetSpec: ...
```

校验规则（失败均抛 `DatasetSpecError`，消息含 hint）：
1. `version == 1`；`data_type` / `layout` 在枚举内；`data_type` 与 `layout` 组合合法：
   - `classification`/`multi_task_classification` ↔ `folder_per_class`
   - `detection`/`segmentation` ↔ `labelme`
2. `multi_task_classification` 必须给 `tasks`（≥1），且每个 task 有 `name/path/classes`；不得同时给顶层 `classes`。
3. 非多任务必须给非空 `classes`；`classes` 不得重复。
4. `seg_mode` 仅 segmentation 可设，且 ∈ {semantic, query}。
5. `splits.mode` ∈ {ratio, list, dir}；ratio 之和 ≈ 1.0（容差 1e-6），键 ⊆ {train, val, test}，train 必填；list 必须给 `list_file`；dir 必须给 `dir_map`。
6. `root` 解析后必须存在且是目录。

## P1-3 `classes.py` `ClassTable`

```python
class ClassTable:
    def __init__(self, names: Sequence[str]) -> None
    def __len__(self) -> int
    def index(self, name: str) -> int          # 未登记 -> AnnotationError
    def name(self, index: int) -> str
    def to_list(self) -> list[str]
    def has(self, name: str) -> bool
```
要求：保持传入顺序（即 label id 顺序）；拒绝重名（`DatasetSpecError`）。

## P1-4 `scan.py` `SampleRef` 与 `scan_samples`

```python
@dataclasses.dataclass
class SampleRef:
    image_path: str                                  # 绝对路径
    rel_path: str                                    # 相对 spec.root，posix 风格
    task: str | None = None                          # 多任务分类
    label: int | None = None                         # 分类
    boxes: "np.ndarray | None" = None                # (N,4) float32 xyxy，绝对像素
    labels: "np.ndarray | None" = None               # (N,) int64
    segments: "list[np.ndarray] | None" = None       # 分割：每个多边形 (P,2) float32
    segment_labels: "np.ndarray | None" = None       # (M,) int64
    ann_path: str | None = None                      # labelme json 绝对路径
    image_size: tuple[int, int] | None = None        # (H, W)，来自 labelme
    meta: dict = field(default_factory=dict)

def scan_samples(spec: DVXDatasetSpec) -> list[SampleRef]:
    """按 spec 扫描全部样本（与 split 无关）。必须确定性排序（按 rel_path）。"""
```
规则：
- 递归扫描时跳过隐藏目录（`.` 开头）与 `dvx_dataset.yaml`。
- 图片按 `parse.image_ext` 匹配（小写比较）；**同一 stem 的 `.jpg`/`.png` 同时存在时视为冲突** → `AnnotationError`。
- labelme：图片必须存在**同名** `.json`（`<stem>.json`）；**缺失 → 跳过该图片并发一次 `UserWarning`（含数量与最多 10 个示例路径），不报错**；JSON 存在但解析失败 → `AnnotationError` 含 JSON 路径与原因；`shapes` 为空视为负样本保留。
- labelme 形状：`rectangle` → `[x1,y1,x2,y2]`（取四个点的 min/max，兼容旋转矩形）；`polygon` → `(P,2)`；其余形状跳过并计入告警计数。
- 类别：`shape["label"]` 必须能在 `ClassTable` 中查到；`ignore_labels` 中的直接丢弃；未知标签 → `AnnotationError`（列出未知标签集合，最多 5 个样例）。
- 空 shapes 允许（负样本），但 `check_dataset` 会统计。
- 多任务：按 `TaskSpec.path` 分别扫描；样本带 `task=name`。

## P1-5 `split.py` `SplitResult` 与 `build_splits`

```python
@dataclasses.dataclass
class SplitResult:
    splits: dict[str, list[int]]          # split 名 -> sample 下标（升序）
    stats: dict                           # 见下
    mode: str
    seed: int | None
    def save(self, path: str) -> None: ...        # JSON
    @classmethod
    def load(cls, path: str) -> "SplitResult": ...
    def summary(self) -> str: ...                  # 人类可读多行
```

`stats` 结构：
```python
{
  "total": int,
  "splits": {"train": {"count": int, "per_class": {"name": int}}, ...},
  "warnings": [str, ...],
}
```

```python
def build_splits(refs: list[SampleRef], config, spec: DVXDatasetSpec) -> SplitResult: ...
```
算法要求：
1. **ratio 模式**：
   - 以 `sorted(ref.rel_path)` 的稳定顺序为基准，用 `random.Random(seed)` 打乱（或用确定性的 `rng.sample`）。
   - `stratify == "class"` 且样本有 `label` 时按 label 分层；多任务按 `(task, label)` 分层；检测/分割无 stratify 时按样本整体。
   - 分配后保证：`train` 非空；每个 split 名都在结果里（即使为空也要出现，附 warning）。
   - 当样本数 < split 数时抛 `SplitError`。
   - 若某类别在 val/test 中缺失 → 追加 warning（不报错）。
2. **dir 模式**：`dir_map` 相对 `spec.root`（多任务时相对每个 `task.path`）推导每个样本的 split；未被任何目录覆盖的样本 → `SplitError`（列出前 5 个）。
3. **list 模式**：读 `list_file`（JSON/YAML），值为相对路径列表；校验所有路径都存在且每个样本恰好属于一个 split；重叠 → `SplitError`。
4. **通用校验**：任意两个 split 下标交集必须为空；下标必须覆盖 `[0, len(refs))` 全部（缺失 → `SplitError` 报告数量）。
5. **确定性**：同一输入多次调用结果完全一致（有测试）。

## P1-6 `parsers/`（folder / labelme / rasterize）

```python
# folder.py
def scan_folder_per_class(root: str, spec: DVXDatasetSpec, task: TaskSpec | None = None,
                          subdir: str | None = None) -> list[SampleRef]: ...
# 布局：<root>[/<task.path>][/<subdir>]/<class>/*.jpg
# 只把"目录名 ∈ 该类表"的目录当类别目录；其余目录（如 train_data）忽略。

# labelme.py
def scan_labelme(root: str, spec: DVXDatasetSpec, class_table: ClassTable) -> list[SampleRef]: ...
def parse_labelme_file(json_path: str, class_table: ClassTable,
                       ignore_labels: Sequence[str], allowed_shape_types: Sequence[str]
                       ) -> tuple[list[np.ndarray], list[int], list[int], int]:
    """返回 (多边形列表, 多边形类别 id 列表, 矩形框类别 id 列表, 跳过的形状数)。
    注意 boxes 与 segments 分开返回，由调用方决定用哪个。"""
```
说明：`scan_labelme` 对 detection 只保留 rectangle；对 segmentation 只保留 polygon（由 `spec.data_type` 决定），另一类形状计入跳过。

```python
# rasterize.py
def polygons_to_mask(polygons: Sequence[np.ndarray], labels: Sequence[int],
                     height: int, width: int, background: int = 0) -> np.ndarray:
    """返回 (H,W) int64；背景填 background；用 cv2.fillPoly 逐多边形填充，
    后写的多边形覆盖先写的（与标注顺序一致）。多边形坐标可为 float，内部取 round。"""
```
要求：坐标裁剪到图像范围；测试覆盖"重叠多边形后者覆盖前者""越界坐标""空多边形列表"。

## P1-7 `dvx_dataset.py` `DVXDataset`

```python
class DVXDataset(BaseDataset):
    def __init__(self, spec: "str | Path | DVXDatasetSpec", split: str = "train",
                 transforms=None, task: str | None = None,
                 split_result: SplitResult | None = None) -> None: ...

    # 只读属性
    spec: DVXDatasetSpec
    split: str
    data_type: str
    classes: list[str] | None            # 单任务类别表
    task_names: list[str] | None         # 多任务
    num_classes: int | None
    refs: list[SampleRef]                # 当前 split 的样本
    split_result: SplitResult

    def __len__(self) -> int
    def __getitem__(self, index: int) -> dict
```
行为：
- 构造时：`scan_samples(spec)` → `build_splits(...)`（若传入 `split_result` 则直接用）→ 取当前 split 下标。
- `task` 参数：仅 `multi_task_classification` 允许；给定时只保留该任务样本。
- `__getitem__` 返回**符合 `Sample` 契约**的 dict：
  - classification：`{"image": path, "label": int}`
  - multi_task：`{"image": path, "label": int, "task": str}`
  - detection：`{"image": path, "boxes": (N,4) float32, "labels": (N,) int64}`
  - segmentation：`{"image": path, "mask": (H,W) int64}`（**取样本时**按 `image_size` 栅格化、不缓存，内存 O(1)；不在 `__init__` 预生成整份 split）
  - 之后**必须复用** `BaseDataset` 的加载与变换逻辑：读图（cv2）→ `self._check_sample_keys` → `self.transforms(sample)`。
- 需要同时修改 `dvisionix/data/sample.py`：把 `"task"` 加入 `Sample.EXTENDED_KEYS`。
- 需要在 `dvisionix/data/__init__.py` 增加 `DVXDataset`、`DVXDatasetSpec`、`load_spec` 的导出（**只加不改**）。

## P1-8 `tools/init_dataset.py`（新增）

```
$PY tools/init_dataset.py --root DIR --data-type TYPE [--layout LAYOUT] [--out FILE] [--force]
```
- 扫描目录，推断 layout（有同名 json → labelme；否则 folder_per_class）。
- folder_per_class：类别 = 一级子目录名（多任务时 `--task-dir` 层）；labelme：类别 = 所有 shape label 的并集（排序后写入）。
- 输出 yaml 草稿；`--out` 未给时打印到 stdout；`--out` 给定时写文件，已存在且无 `--force` → 报错。
- **不得**自动写入数据目录（除非显式 `--out`）。
- 输出末尾必须打印"请人工确认 classes 顺序（顺序即 label id）"。

## P1-9 `tools/check_dataset.py`（新增）

```
$PY tools/check_dataset.py --spec FILE [--json] [--split-preview]
```
输出：
- 基本信息：data_type / layout / root / classes 数 / 样本总数 / 每类样本数。
- 标注质量：segmentation 的像素覆盖率（按多边形面积估算）、detection 的框数分布与越界框数、空 shapes 数、跳过的形状数。
- `--split-preview`：按 spec.splits 计算并打印每个 split 的样本数、每类分布、warning。
- `--json`：上述内容的 JSON。
- 发现严重问题（未知类别、缺 json、类别表为空、切分产生空 train）→ 退出码 1。

## P1-10 四个真实数据集的 spec（写入数据根目录）

文件位置：`project/data/{cls,det,multicls,seg}/dvx_dataset.yaml`（**不移动、不修改任何图片/JSON**）。

- `cls`：`data_type: classification`，`layout: folder_per_class`，5 类（中文，顺序：幻影灰/新卡其白/新碳晶黑/科技灰/纳斯达克银），`splits: ratio 0.8/0.2 seed 42 stratify class`。
- `det`：`data_type: detection`，`layout: labelme`，**20 类并集**（顺序由 init 工具生成后人工确认），`splits: ratio 0.8/0.2 seed 42 stratify none`。
- `multicls`：`data_type: multi_task_classification`，`layout: folder_per_class`，标准结构 `<任务>/<类别>/*.jpg`，`tasks` 两任务；`splits: mode ratio {train: 0.8, val: 0.2} seed 42 stratify class`（**不使用 dir_map**）。
- `seg`：`data_type: segmentation`，`layout: labelme`，`seg_mode: semantic`，类表 `[__background__, <17 类按现有 label 排序>]`，`ignore_index: 255`，`splits: ratio 0.8/0.2 seed 42 stratify none`。

> 注意：`cls` / `det` / `seg` / `multicls` 四个数据集都**没有预先划分的 split 目录**，统一用 `ratio` 模式在读取时构建切分（非破坏性）。

## P1-11 测试清单（`tests/`）

必须新增：
- `tests/test_architecture/test_layering.py`（P0-4）
- `tests/test_data/test_dvx_spec.py`：合法/非法 yaml、组合校验、root 解析、多任务校验。
- `tests/test_data/test_dvx_classes.py`：ClassTable 顺序/重名/未知。
- `tests/test_data/test_dvx_split.py`：ratio 确定性、分层、dir、list、重叠/缺失报错、train 非空。
- `tests/test_data/test_dvx_parsers.py`：folder 扫描（含中文类名、递归、忽略非类目录）；labelme 扫描（rectangle/polygon、未知标签、缺 json、冲突扩展名）；rasterize（覆盖顺序、越界、空）。
- `tests/test_data/test_dvx_dataset.py`：四种 data_type 各构造一个迷你数据集，`__getitem__` 字段/dtype/shape 正确，DataLoader 能取 batch。
- `tests/test_data/test_acceptance_real_data.py`：对 `project/data/*/dvx_dataset.yaml` 逐个构建 dataset 与切分；`skipif` 路径不存在。

全部使用 `tmp_path` 造夹具（用 numpy + cv2 生成 8×8 图片、手写 labelme JSON）。
**不得**把 `project/data` 的图片复制进仓库。

## P1-12 验收命令

```bash
cd /data/lzy/DVisionix/DVisionix
$PY -m pytest tests/test_architecture -q
$PY -m pytest tests/test_data -q
$PY -m pytest tests -q -m "not slow"          # 既有 531 条必须继续全绿
$PY tools/check_layering.py
$PY tools/check_dataset.py --spec /data/lzy/DVisionix/project/data/cls/dvx_dataset.yaml --split-preview
$PY tools/check_dataset.py --spec /data/lzy/DVisionix/project/data/det/dvx_dataset.yaml --split-preview
$PY tools/check_dataset.py --spec /data/lzy/DVisionix/project/data/multicls/dvx_dataset.yaml --split-preview
$PY tools/check_dataset.py --spec /data/lzy/DVisionix/project/data/seg/dvx_dataset.yaml --split-preview
$PY -m ruff check dvisionix tools tests
$PY -m black --check dvisionix tools tests
```

---

## 附：文件所有权（避免并发冲突）

| 归属 | 文件 |
|---|---|
| P0 实现 | `dvisionix/errors.py`、`docs/architecture/layers.yaml`、`tools/check_layering.py`、`tests/test_architecture/*`、`.gitignore` |
| P1-A 实现 | `dvisionix/data/spec/*`、`tests/test_data/test_dvx_spec.py`、`test_dvx_classes.py`、`test_dvx_split.py` |
| P1-B 实现 | `dvisionix/data/parsers/*`、`tests/test_data/test_dvx_parsers.py` |
| P1-C 实现 | `dvisionix/data/dvx_dataset.py`、`dvisionix/data/sample.py`(仅加 task 键)、`dvisionix/data/__init__.py`(仅加导出)、`tools/init_dataset.py`、`tools/check_dataset.py`、`project/data/*/dvx_dataset.yaml`、`tests/test_data/test_dvx_dataset.py`、`test_acceptance_real_data.py` |
| 文档 | `docs/dataset_spec.md`、`docs/README.md`、`README.md`、`CHANGELOG.md` |
| Lead | 本契约、审查、验收、git 提交 |

任何实现方**不得**修改他人所有权范围内的文件。

---

## 附二：P1 实现裁决记录（Lead 确认，2026-11-14）

以下偏差由实现方在报告中提出，Lead 逐条裁决为**接受**，契约以本节为准：

1. **`parse_labelme_file` 返回值**：按契约字面实现为
   `(polygons, polygon_labels, box_label_ids, skipped_count)`——第三个返回值是**矩形类别 id**，不返回框坐标；
   矩形几何统一由 `scan_labelme → SampleRef.boxes` 提供（共用内部解析，无重复读盘）。
   契约 P1-6 中"注意 boxes 与 segments 分开返回"一句仅指 `SampleRef` 字段，不指该函数返回值。
2. **`scan_samples(spec, subdir=None)`**：`subdir` 仅作为**显式可选参数**保留（显式传入才扫描该子目录层）；
   **不再**根据 `dir_map` 自动推断公共子目录，也不再向 stderr 打印自动收敛信息。
   （原自动推断逻辑是为旧 multicls 的 `train_data/` 重复目录而加，用户已清理该目录，故移除，见附三。）
3. **同 stem 多格式冲突**：`folder` 与 `labelme` 两种布局都做检测，冲突抛 `AnnotationError`。
4. **异常类型划分**：扫描期数据问题（目录缺失/缺 JSON/未知标签/扩展名冲突/重复 rel_path）、
   rasterize 参数问题 → `AnnotationError`；空类别表、labelme 被用于非 detection/segmentation、
   `shape_types` 无匹配 → `DatasetSpecError`。
5. **`recursive=False` 语义**：labelme 只看 `spec.root` 本层；folder 只看 `<base>/<class>/*` 一层。
6. **`image_size` 兜底**：labelme JSON 缺 `imageWidth/imageHeight` 时用 cv2 读图尺寸并 `UserWarning`。
7. **空标注语义**：detection 空 shapes → `boxes (0,4) float32` + `labels (0,) int64`；
   segmentation → `segments == []` + `segment_labels (0,) int64`（不用 None）。
8. **`SampleRef.meta` 自由字段**：folder 写 `class_name`；labelme 写 `skipped_shapes` / `shape_count` / `image_data_present`。

---

## 附三：P1 用户反馈修正（2026-11-15）

1. **缺标注改为跳过**：图片没有同名 labelme JSON 时不再报错，跳过并对全部跳过项发一次 `UserWarning`
   （含数量 + 最多 10 个示例相对路径）；`shapes` 为空的 JSON 仍作为**负样本**保留；
   损坏/非法 JSON、未知标签仍抛 `AnnotationError`。
2. **多任务标准结构**：`<任务>/<类别>/*.jpg`。`multicls` 规格改用 `ratio` 切分，不再使用 `dir_map`；
   `scan.py` 移除"自动推断公共一级子目录"的魔法，`subdir` 仅作为显式可选参数保留。
   `splits.mode: dir` 仍是通用能力（适用于数据本身已按目录划分的用户）。
