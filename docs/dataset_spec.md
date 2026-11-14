# DVX 标准数据集

DVX 是 v2 引入的数据集描述格式：**一个数据集 = 一个根目录 + 一份 `dvx_dataset.yaml`**。
规格文件声明"数据在哪里、有什么标注、怎么切分"，`DVXDataset` 按声明把样本读成符合
`Sample` 契约的 dict，可直接交给 `DataLoader`（检测/分割需使用专用 collate，见 §6.3）。

> 适用范围：v2 开发中（P1 已完成，尚未发布）。训练侧的配置接线在 P2，
> 当前 `tools/train.py` **还不能**通过 YAML 配置直接构建 `DVXDataset`，见文末「边界说明」。
> 运行示例前请先激活 conda 环境 `dvisionix`。

相关文档：[数据模块](data.md)、[快速开始](quick_start.md)、
[架构与分层规则](architecture.md)。

---

## 1. 设计原则

1. **一个数据集 = 一个根目录 + 一份 `dvx_dataset.yaml`**。
   规格里的 `root` 指向根目录，扫描器只读它下面的图片与标注。
2. **多级目录只影响路径，不携带语义**。
   目录层级本身不表示任务或 split；`rel_path`（相对 `root` 的 posix 路径）只用来稳定地
   标识样本。唯一的例外是 `folder_per_class`：**目录名命中 `classes` 的目录才被当作类别目录**，
   其余目录（如 `train_data/`、`images/`）只是路径的一部分。
3. **单任务数据集的所有标注共用一张类别表**。
   `classes` 的顺序就是 label id（0、1、2……），检测、分割、分类都不例外；多任务分类是
   例外，每个任务各有一张独立类别表（`tasks[*].classes`）。顺序一旦用于训练就不要随意调整，
   否则旧标注/旧 checkpoint 的 id 含义会变。
4. **切分是非破坏性的**。
   `ratio` / `dir` / `list` 三种模式都只在内存中生成"样本下标 → split"的索引，
   绝不移动、改名或删除原始文件；同一份数据 + 同一个 seed 得到完全相同的切分。
5. **`data_type` 描述"有什么标注"，不决定"训练什么"**。
   `data_type: detection` 表示样本带检测框，具体训练目标（分类/检测/蒸馏/检索……）
   由训练任务决定。本期只完成数据层，训练目标接线在 P2。

---

## 2. 两种目录布局

### 2.1 `folder_per_class`（分类 / 多任务分类）

每个类别一个目录，图片放在类别目录下（可再嵌套子目录）：

```text
my_cls/                         # 数据集根目录（root）
├── dvx_dataset.yaml
├── cat/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 2024/                   # 允许再嵌套：最近的类别祖先是 cat
│       └── 003.jpg
└── dog/
    ├── 001.jpg
    └── 002.png
```

```yaml
version: 1
data_type: classification
layout: folder_per_class
root: .
classes: [cat, dog]
```

规则：

- 扫描器从图片所在目录**向上找最近的、目录名在 `classes` 中的祖先目录**作为类别；
  `<class>/x.jpg` 与 `<class>/sub/x.jpg` 都属于该类。
- 不在任何类别目录下的图片（例如直接放在 `root/` 或 `root/images/` 下）不会被收录。
- 类别目录的**名字必须与 `classes` 中的字符串完全一致**（区分大小写）。
- 隐藏目录/隐藏文件（`.` 开头）跳过；图片扩展名按 `parse.image_ext` 做**小写比较**。
- 同一目录下同一 stem 同时存在多种格式（如 `x.jpg` 与 `x.png`）→ `AnnotationError`。

多任务分类是同样的布局，只是每个任务有自己的子目录和独立类别表：

```text
my_multi/
├── dvx_dataset.yaml
├── licence/
│   ├── 无/
│   ├── 短牌照/
│   └── 长牌照/
├── logo/
│   ├── 一体奇瑞标/
│   └── 嵌入奇瑞标/
└── train_data/                 # 可选：已有的切片目录（dir 模式使用）
    ├── train/...
    └── val/...
```

```yaml
version: 1
data_type: multi_task_classification
layout: folder_per_class
root: .
tasks:
  - {name: licence, path: licence, classes: [无, 短牌照, 长牌照]}
  - {name: logo, path: logo, classes: [一体奇瑞标, 嵌入奇瑞标]}
```

每个任务的类别 id 从 0 开始独立编号；样本同时带 `task` 名与 `label`。
多任务数据集**不允许**再写顶层 `classes`。

### 2.2 `labelme`（检测 / 分割）

每张图片旁边放一份**同目录、同 stem** 的 `.json`：

```text
my_det/
├── dvx_dataset.yaml
├── batch_01/
│   ├── a.jpg
│   ├── a.json
│   ├── b.jpg
│   └── b.json
└── batch_02/
    └── ...
```

```yaml
version: 1
data_type: detection
layout: labelme
root: .
classes: [person, car]
```

规则：

- 图片与 `<stem>.json` 必须同目录同名；缺 JSON → `AnnotationError`（空标注要写 `"shapes": []`，
  这样是合法的负样本，不是"没有标注"）。
- 默认映射下，`detection` 只保留 `rectangle`：取四个角点的 min/max 得到 `[x1, y1, x2, y2]`
  （兼容旋转矩形）；`segmentation` 只保留 `polygon`（可由 `parse.labelme.shape_types` 调整）。
- 其他形状（`circle` / `line` / `point` / `linestrip` / `mask`……）跳过，并计入
  `SampleRef.meta["skipped_shapes"]`（`check_dataset` 会统计）。
- 形状的 `label` 必须能在类别表中查到；`parse.labelme.ignore_labels` 中的标签直接丢弃
  （不计入未知标签，也不计入跳过数）。
- `imageWidth` / `imageHeight` 用于确定分割掩码尺寸；缺失时用 `cv2` 读图兜底并告警。
- 分割类表通常写成 `[__background__, 类1, 类2, ...]`，见 §8.1。

---

## 3. 字段总表

最小规格只有 `version` / `data_type` / `layout` / `root` / `classes`（多任务为 `tasks`）。
其余字段都有默认值。

### 3.1 顶层字段

| 字段 | 类型 | 默认值 | 约束与说明 |
|---|---|---|---|
| `version` | int | 无（必填） | 当前只支持 `1`；其他值报 `DatasetSpecError` |
| `data_type` | str | 无（必填） | `classification` / `multi_task_classification` / `detection` / `segmentation` |
| `layout` | str | 无（必填） | `folder_per_class` / `labelme`；与 `data_type` 的组合必须合法（见下表） |
| `root` | str | `"."` | 相对**规格文件所在目录**解析，也支持绝对路径；解析后必须存在且是目录 |
| `classes` | list[str] | 无 | 非多任务数据集必填；不能为空、不能重名；**顺序即 label id** |
| `tasks` | list[task] | 无 | 仅 `multi_task_classification` 必填；每项为 `{name, path, classes}` |
| `splits` | mapping | 无（不配置切分） | 切分配置；不写时 `DVXDataset` 无法自动切分（除非传 `split_result`，见 §4） |
| `parse` | mapping | 见 §3.3 | 扫描/解析行为 |
| `seg_mode` | str | `semantic` | `semantic` / `query`；仅 `segmentation` 可设置非默认值；**本期只实现 `semantic`** |
| `ignore_index` | int | `255` | 分割忽略像素值；**当前只做校验与展示，`DVXDataset` 尚未消费**（见 §8.1） |

`data_type` 与 `layout` 的合法组合：

| `data_type` | 允许的 `layout` | 样本形态 |
|---|---|---|
| `classification` | `folder_per_class` | 图片 + 单标签 |
| `multi_task_classification` | `folder_per_class` | 图片 + 每任务一个标签 |
| `detection` | `labelme` | 图片 + 矩形框 |
| `segmentation` | `labelme` | 图片 + 多边形（取样本时栅格化为掩码） |

### 3.2 `tasks[*]`（仅多任务分类）

| 字段 | 类型 | 约束与说明 |
|---|---|---|
| `name` | str | 任务名，规格内唯一；加载时用 `task="..."` 过滤 |
| `path` | str | 相对 `root` 的任务子目录；多任务之间不能重叠到同一批图片 |
| `classes` | list[str] | 该任务独立的类别表；非空、不重名；顺序即该任务的 label id |

多任务规格**不得**同时提供顶层 `classes`；反过来，非多任务规格提供 `tasks` 也会报错。
`path` 指向的目录必须存在，否则扫描时抛 `AnnotationError`。

### 3.3 `parse`

| 字段 | 类型 | 默认值 | 约束与说明 |
|---|---|---|---|
| `recursive` | bool | `true` | 是否递归扫描。`false` 的语义见 §8 的排查表 |
| `image_ext` | list[str] | `[.jpg, .jpeg, .png, .bmp]` | 非空；小写比较；缺前导点会自动补全并告警 |
| `labelme.ignore_labels` | list[str] | `[__ignore__, __background__]` | 直接丢弃的 label（在未知标签检查之前） |
| `labelme.image_data` | str | `ignore` | `ignore`：容忍内嵌 `imageData` 并计数/告警；`error`：直接报错 |
| `labelme.shape_types` | mapping | `{rectangle: detection, polygon: segmentation}` | 形状类型 → 目标任务；值只能是 `detection` / `segmentation`；必须至少有一种形状匹配当前 `data_type`，否则报 `DatasetSpecError` |

### 3.4 `splits`

| 字段 | 类型 | 默认值 | 约束与说明 |
|---|---|---|---|
| `mode` | str | `ratio` | `ratio` / `dir` / `list` |
| `ratio` | mapping | `{train: 0.8, val: 0.2}` | 仅 `ratio` 模式使用；键 ⊆ `{train, val, test}`；必须含 `train`；非负；总和 = 1（容差 `1e-6`） |
| `seed` | int | `42` | 仅 `ratio` 模式使用；决定切分可复现性 |
| `stratify` | str | `class`（分类/多任务）/ `none`（检测、分割） | 仅 `ratio` 模式使用；`class` 按类别分层，`none` 不分层 |
| `list_file` | str | 无 | 仅 `list` 模式必填；JSON/YAML 清单路径，相对**规格文件所在目录**（也支持绝对路径） |
| `dir_map` | mapping | 无 | 仅 `dir` 模式必填；`{split: 相对目录}`，相对 `root`（多任务相对每个 `task.path`），不能写绝对路径 |

> 未知字段只会触发 `UserWarning` 并被忽略（例如把 `splits` 误写成 `split`），
> 看到告警时请对照上表确认字段名。
>
> 类别名是**字符串**：纯数字（如 `7`）、布尔、日期等会被 YAML 自动推断成非字符串，
> 必须加引号写成 `'7'`；含 `:` / `#` 等特殊字符的类名也建议加引号。真实 `det` 的
> `classes` 第一条 `'7'` 就是这种情况。

---

## 4. 三种切分模式

三种模式的共同保证：

- **非破坏性**：只生成下标列表（`SplitResult.splits`），不触碰磁盘上的图片/JSON。
- **确定性**：`ratio` 模式固定 `seed` 且先按 `rel_path` 排序再打乱，输入顺序不影响结果；
  `dir` / `list` 模式按路径判定，本身确定。
- **覆盖与不重叠**：所有 split 的下标两两不相交，并完整覆盖 `[0, 样本数)`；
  否则 `SplitError`（不会被静默吞掉）。
- `train` 必须存在且非空；`train` 为空时会从样本最多的其他 split 移动一个样本并记 warning，
  若没有可移动的样本则直接报错。
- 某个类别在 `val` / `test` 中没有样本时只追加 warning，不中断。
- 样本数少于 split 数量（例如 1 个样本要切 train/val）时抛 `SplitError`。

### 4.1 `ratio`：按比例随机切分

```yaml
splits:
  mode: ratio
  ratio: {train: 0.8, val: 0.2}
  seed: 42
  stratify: class          # 分类/多任务默认 class；检测/分割默认 none
```

算法要点：

1. 以 `sorted(ref.rel_path)` 为稳定基准；
2. 用 `random.Random(seed)` 打乱；
3. `stratify: class` 时按 `label` 分组（多任务按 `(task, label)`），各组内部分配后再合并；
4. 用最大余数法把每组样本数分配到各 split，保证总数守恒。

`stratify` 缺省时由 `data_type` 决定：分类/多任务为 `class`，检测/分割为 `none`。

### 4.2 `dir`：按已有目录切分

```yaml
splits:
  mode: dir
  dir_map:
    train: train_data/train
    val: train_data/val
```

- 每个样本的 `rel_path` 必须落在**至少一个** `dir_map` 目录下；未被任何目录覆盖 → `SplitError`
  （列出前 5 个样例）。
- 目录可以嵌套：同时命中多个时按**最长目录**归属，并记 warning（建议显式避免）。
- 多任务时 `dir_map` 相对每个 `task.path` 解析（例如 `train_data/train` 实际是
  `<root>/<task.path>/train_data/train`）。
- 已经有 `train_data/{train,val}` 切片、同时根目录又有扁平类目录时，同一批图片会以两个
  不同 `rel_path` 被扫描两次。`scan_samples` 在 `dir` 模式下会自动收敛到 `dir_map` 的
  **公共一级子目录**（`train_data/train` + `train_data/val` → `train_data`），并向 **stderr**
  打印一行说明；也可以编程式调用 `scan_samples(spec, subdir="train_data")` 显式指定。
  真实 `multicls` 数据集就属于这种情况。

### 4.3 `list`：按清单文件切分

```yaml
splits:
  mode: list
  list_file: split.json
```

清单文件相对规格文件所在目录解析（也可写绝对路径），JSON 或 YAML 均可：

```json
{
  "train": ["cat/001.jpg", "cat/002.jpg", "dog/001.jpg"],
  "val": ["cat/003.jpg", "dog/002.jpg"]
}
```

- 路径必须是相对 `root` 的 posix 风格 `rel_path`；
- 每个样本必须**恰好出现一次**：同一 split 内重复、跨 split 重叠都报 `SplitError`；
- 清单里出现数据集中不存在的路径 → `SplitError`（列出前 5 个）；
- 数据集中有样本没被清单覆盖 → `SplitError`（列出前 5 个）；
- 必须包含 `train`。

### 4.4 保存与复用切分结果

```python
from dvisionix.data import SplitResult, load_spec
from dvisionix.data.spec import build_splits, scan_samples

spec = load_spec("project/data/cls/dvx_dataset.yaml")
refs = scan_samples(spec)
result = build_splits(refs, spec.splits, spec)
print(result.summary())
result.save("work/cls_split.json")

# 之后直接复用，不再重新计算（下标必须与本次扫描结果一致）
saved = SplitResult.load("work/cls_split.json")
```

`DVXDataset(..., split_result=saved)` 会直接使用传入的切分结果。**数据集内容变化后要重新生成**，
否则下标可能越界或指向别的样本，`DVXDataset` 会抛 `SplitError` 而不是猜测。

---

## 5. 三步上手

以下命令都在仓库根目录执行；`python` 需指向已安装依赖的 `dvisionix` 环境
（本机验证用 `/data/lzy/Env/condaEnvs/dvisionix/bin/python`）。

### 5.1 第一步：生成规格草稿

`tools/init_dataset.py` 只做**只读扫描**，不会移动/改名/删除任何图片或 JSON。
它按"图片是否存在同名 JSON"推断布局，并把类别写成排序后的草稿：

```bash
# 分类：扫描 <root>/<class>/*.jpg，类别取一级子目录名
python tools/init_dataset.py \
  --root ./data/my_cls \
  --data-type classification \
  --out ./data/my_cls/dvx_dataset.yaml

# 检测：类别取所有 shape label 的并集（排序）
python tools/init_dataset.py \
  --root ./data/my_det \
  --data-type detection \
  --out ./data/my_det/dvx_dataset.yaml

# 分割：同上，并自动写入 seg_mode: semantic / ignore_index: 255
python tools/init_dataset.py \
  --root ./data/my_seg \
  --data-type segmentation \
  --out ./data/my_seg/dvx_dataset.yaml

# 多任务：建议显式给出每个任务目录（--task-dir 可重复）
python tools/init_dataset.py \
  --root ./data/my_multi \
  --data-type multi_task_classification \
  --task-dir licence --task-dir logo \
  --out ./data/my_multi/dvx_dataset.yaml
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--root DIR` | 数据根目录（必填，只读扫描） |
| `--data-type TYPE` | 四种类型之一（必填） |
| `--layout {folder_per_class,labelme}` | 缺省按同名 JSON 自动推断；与推断不一致时会提示 |
| `--task-dir REL_DIR` | 多任务的任务子目录，可重复；不写时自动发现一级任务目录 |
| `--out FILE` | 草稿输出路径；**不写则只打印到 stdout**，绝不会自动写数据目录 |
| `--force` | 允许覆盖已存在的 `--out` 文件 |

`--out` 给定时，草稿中的 `root` 会写成**相对输出文件所在目录**的路径，方便规格随数据目录一起移动。
命令末尾会提示"请人工确认 classes 顺序（顺序即 label id）"。

### 5.2 第二步：人工确认类别顺序并体检

打开生成的 `dvx_dataset.yaml`，先确认两件事：

1. **`classes` 是否齐全、顺序是否正确**。初始化工具按字符串排序输出（中文按 Unicode 排序），
   这个顺序是任意的，但一旦用于训练就会固定为 label id；请按你的约定调整。
2. **`layout` / `root` 是否符合预期**。`root` 相对规格文件解析，通常写成 `root: .`。

确认后运行体检（只读）：

```bash
python tools/check_dataset.py \
  --spec ./data/my_cls/dvx_dataset.yaml \
  --split-preview
```

输出包含类别数、样本数、每类分布、标注质量（检测框/多边形/跳过形状等）以及切分预览。
常见问题（未知标签、缺 JSON、类别表为空、`train` 为空等）会以退出码 `1` 结束；
只有 warning 时退出码仍为 `0`。需要机器可读结果时加 `--json`（stdout 只输出 JSON）。

### 5.3 第三步：用 `DVXDataset` 加载

```python
from torch.utils.data import DataLoader
from dvisionix.data import DVXDataset
from dvisionix.data.transforms import ClassificationTransforms

train_ds = DVXDataset(
    "./data/my_cls/dvx_dataset.yaml",
    split="train",
    transforms=ClassificationTransforms(train=True, image_size=224),
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

sample = train_ds[0]
print(sample["image"].shape, sample["label"])   # (3, 224, 224) tensor(...)
```

三种数据类型的完整写法见下一节。

---

## 6. 编程用法

### 6.1 构造

```python
from dvisionix.data import DVXDataset, DVXDatasetSpec, SplitResult, load_spec

# 方式一：直接传规格路径（相对当前工作目录）
ds = DVXDataset("project/data/cls/dvx_dataset.yaml", split="train")

# 方式二：先加载、检查，再传规格对象
spec = load_spec("project/data/cls/dvx_dataset.yaml")
ds = DVXDataset(spec, split="val", transforms=None)

# 方式三：复用已保存的切分结果
saved = SplitResult.load("work/cls_split.json")
ds = DVXDataset(spec, split="val", split_result=saved)
```

构造参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `spec` | `str` / `PathLike` / `DVXDatasetSpec` | 必填 | 规格路径或已加载的规格对象 |
| `split` | str | `"train"` | 要取的切分名；不在 `split_result.splits` 中会报 `SplitError` |
| `transforms` | pipeline / callable / None | `None` | `None` 时只把图片读成 RGB `uint8 ndarray` |
| `task` | str / None | `None` | 仅多任务分类可用；给定后只保留该任务的样本 |
| `split_result` | `SplitResult` / None | `None` | 给定时直接使用，否则按 `spec.splits` 现场构建 |

构造期会执行：`spec.validate()` → `scan_samples()` → `build_splits()`（或使用传入的
`split_result`）→ 取当前 split 的样本引用。`scan_samples` 会读取全部标注，因此大规模数据集构造一次即可复用 `DVXDataset` 对象；
分割掩码**不在构造期生成**。

### 6.2 只读属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `spec` | `DVXDatasetSpec` | 构建所用的规格对象 |
| `split` | str | 当前切分名 |
| `data_type` | str | 四种类型之一 |
| `classes` | `list[str]` / `None` | 单任务类别表；多任务分类为 `None` |
| `task_names` | `list[str]` / `None` | 多任务分类的任务名；其他类型为 `None` |
| `num_classes` | int / None | 单任务类别数；多任务分类为 `None` |
| `refs` | `list[SampleRef]` | 当前 split（及 `task` 过滤后）的样本引用 |
| `split_result` | `SplitResult` | 实际使用的切分结果 |
| `collate_fn` | callable / None | 检测/分割为专用 collate；分类为 `None`（用默认 collate） |

多任务分类没有统一类别表：请用 `ds.spec.num_classes("licence")`、`ds.spec.class_names("licence")`
取某个任务的类别信息。

### 6.3 `__getitem__` 返回结构

| `data_type` | 字段 | 类型 / 形状 | 说明 |
|---|---|---|---|
| 所有 | `image` | `uint8 (H, W, 3)` | BGR→RGB 后的 numpy；有 `transforms` 时可能是 Tensor |
| `classification` | `label` | `int` | 类别 id |
| `multi_task_classification` | `label` + `task` | `int` + `str` | 任务名与在该任务类表内的 id |
| `detection` | `boxes` | `float32 (N, 4)` | `[x1, y1, x2, y2]` 绝对像素 xyxy |
| `detection` | `labels` | `int64 (N,)` | 与 `boxes` 一一对应；负样本为 `(0, 4)` / `(0,)` |
| `segmentation` | `mask` | `int64 (H, W)` | 取样本时按 `image_size` 栅格化；背景为 `0` |

分割掩码在 `__getitem__` 中按需生成（不缓存），因此内存占用与 batch 大小相关，
不会在构造期把整个 split 的掩码同时放进内存。

`DVXDataset` 默认注入的 `collate_fn`：

- `detection` → `detection_collate`：`image` 堆叠为 `(B, C, H, W)`，`boxes` / `labels` 保持 list；
- `segmentation` → `segmentation_collate`：`image` 与 `mask` 一起堆叠；
- 其他 → `None`，交给 PyTorch 默认 collate。

> **`DataLoader` 不会自动使用 `dataset.collate_fn`**。检测/分割要显式写
> `DataLoader(ds, ..., collate_fn=ds.collate_fn)`，否则默认 collate 无法处理变长框/掩码。
> 专用 collate 会对 `image` 做 `torch.stack`，所以 DataLoader 场景下必须使用带 ToTensor 的
> transforms（`DetectionTransforms` / `SegmentationTransforms` 已包含）；只做原始数据检查时
> 可以用 `transforms=None` 逐样本访问。

### 6.4 分类

```python
from torch.utils.data import DataLoader
from dvisionix.data import DVXDataset
from dvisionix.data.transforms import ClassificationTransforms

train_ds = DVXDataset(
    "project/data/cls/dvx_dataset.yaml",
    split="train",
    transforms=ClassificationTransforms(train=True, image_size=224),
)
val_ds = DVXDataset(
    "project/data/cls/dvx_dataset.yaml",
    split="val",
    transforms=ClassificationTransforms(train=False, image_size=224),
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

print(train_ds.classes, train_ds.num_classes)   # 5 类，顺序即 label id
sample = train_ds[0]                            # {"image": Tensor, "label": Tensor}
batch = next(iter(train_loader))
print(batch["image"].shape, batch["label"].shape)
```

### 6.5 检测（labelme）

```python
from torch.utils.data import DataLoader
from dvisionix.data import DVXDataset
from dvisionix.data.transforms import DetectionTransforms

train_ds = DVXDataset(
    "project/data/det/dvx_dataset.yaml",
    split="train",
    transforms=DetectionTransforms(train=True, image_size=640),
)
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=train_ds.collate_fn,   # 必须是 dataset 自带的 collate
)

sample = train_ds[0]
print(sample["boxes"].shape, sample["boxes"].dtype)   # (N, 4) float32
print(sample["labels"].shape, sample["labels"].dtype) # (N,) int64

batch = next(iter(train_loader))
print(batch["image"].shape, len(batch["boxes"]), batch["boxes"][0].shape)
```

检测允许空标注：`boxes` 为 `(0, 4) float32`、`labels` 为 `(0,) int64`，训练时需按任务约定处理负样本。

### 6.6 分割（labelme）

```python
from torch.utils.data import DataLoader
from dvisionix.data import DVXDataset
from dvisionix.data.transforms import SegmentationTransforms

train_ds = DVXDataset(
    "project/data/seg/dvx_dataset.yaml",
    split="train",
    transforms=SegmentationTransforms(train=True, image_size=512),
)
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=train_ds.collate_fn,
)

sample = train_ds[0]
print(sample["mask"].shape, sample["mask"].dtype)   # (H, W) int64
print(sorted(set(sample["mask"].flatten().tolist())))  # 0 为背景，其余为类别 id
```

不传 `transforms` 时可以直接检查原始尺寸的掩码：

```python
ds = DVXDataset("project/data/seg/dvx_dataset.yaml", split="val", transforms=None)
sample = ds[0]
print(sample["image"].shape, sample["mask"].shape)
```

### 6.7 多任务分类

```python
from torch.utils.data import DataLoader
from dvisionix.data import DVXDataset
from dvisionix.data.transforms import ClassificationTransforms

ds = DVXDataset(
    "project/data/multicls/dvx_dataset.yaml",
    split="train",
    transforms=ClassificationTransforms(train=True, image_size=224),
)
print(ds.classes)        # None：多任务没有统一类别表
print(ds.task_names)     # ['licence', 'logo']
print(ds.spec.num_classes("licence"))   # 3

# 只取某一个任务的样本（可选）
licence_ds = DVXDataset(
    "project/data/multicls/dvx_dataset.yaml",
    split="train",
    task="licence",
    transforms=ClassificationTransforms(train=True, image_size=224),
)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
batch = next(iter(loader))
print(batch["image"].shape, batch["task"][:2], batch["label"][:2])
```

多任务分类的 `ratio` 切分按 `(task, label)` 分层；`dir` 切分的目录相对每个 `task.path` 解析。

---

## 7. 四套真实数据实例

本仓库配套的验收数据位于 `/data/lzy/DVisionix/project/data/`
（与代码仓库同级、**不入库**；`tests/test_data/test_acceptance_real_data.py` 按
`DVX_DATA_ROOT` → `<repo>/project/data` → `<repo>/../project/data` 的顺序查找）。
四套 spec 都由 `tools/init_dataset.py` 生成草稿后人工确认类别顺序。

| 数据集 | 规格 | 类型 / 布局 | 类别数 | 样本数 | 默认切分（train / val） |
|---|---|---|---|---|---|
| `cls` | `project/data/cls/dvx_dataset.yaml` | classification / folder_per_class | 5 | 2210 | 1768 / 442，ratio 0.8/0.2，stratify class |
| `det` | `project/data/det/dvx_dataset.yaml` | detection / labelme | 20 | 1646 | 1317 / 329，ratio 0.8/0.2，stratify none |
| `multicls` | `project/data/multicls/dvx_dataset.yaml` | multi_task_classification / folder_per_class | 5（licence 3 + logo 2） | 2458 | 1964 / 494，dir 模式 |
| `seg` | `project/data/seg/dvx_dataset.yaml` | segmentation / labelme | 18（`__background__` + 17） | 766 | 613 / 153，ratio 0.8/0.2，stratify none |

实测统计（`check_dataset --split-preview --json`，2026-11-14）：

- **cls**：幻影灰 530 / 新卡其白 601 / 新碳晶黑 609 / 科技灰 193 / 纳斯达克银 277；
  分层切分后 train 每类 424/481/487/154/222，val 106/120/122/39/55；无 warning。
- **det**：20 类，共 10014 个框；每样本框数 min/median/mean/max = 4/6/6.08/10；
  越界框、退化框、缺失尺寸均为 0；2 个 JSON 内嵌 `imageData`（按默认 `ignore` 处理并告警）；
  80/20 切分后 train 1317 / val 329。
- **multicls**：licence 1076（无 442 / 短牌照 30 / 长牌照 604）、logo 1382
  （一体奇瑞标 856 / 嵌入奇瑞标 526），合计 2458；`dir` 模式自动收敛到 `train_data`
  公共子目录，train 1964 / val 494，无重复样本。
- **seg**：18 类（`__background__` 目标数为 0），4348 个多边形；每样本 3~9 个多边形
  （中位数 5）；按多边形面积估算的像素覆盖率 mean 0.161 / min 0.039 / max 0.230；
  越界顶点、空 polygons、跳过形状均为 0；80/20 切分后 train 613 / val 153。

一次检查四套数据：

```bash
for name in cls det multicls seg; do
  python tools/check_dataset.py \
    --spec "/data/lzy/DVisionix/project/data/$name/dvx_dataset.yaml" \
    --split-preview
done
```

四套当前结论均为「通过」。可直接加载验证：

```python
from dvisionix.data import DVXDataset

for name in ("cls", "det", "multicls", "seg"):
    ds = DVXDataset(f"/data/lzy/DVisionix/project/data/{name}/dvx_dataset.yaml", split="train")
    print(name, len(ds), ds.data_type, ds.num_classes)
```

> 真实数据根目录不在仓库内；请把上面的绝对路径换成自己的数据位置。
> 仓库内另附四份示例模板（它们是 DVX 规格，不是 `tools/train.py` 的训练配置）：
> [分类](../configs/datasets/classification_folder.example.yaml)、
> [检测](../configs/datasets/detection_labelme.example.yaml)、
> [分割](../configs/datasets/segmentation_labelme.example.yaml)、
> [多任务](../configs/datasets/multi_task_classification.example.yaml)。

---

## 8. 常见错误与排查

所有领域错误都继承 `DVisionixError`，消息为「发生了什么 + 提示：应该怎么办」。
数据问题通常落在 `DatasetSpecError` / `AnnotationError` / `SplitError` 三类。

| 现象 / 报错 | 原因 | 处理 |
|---|---|---|
| `AnnotationError: labelme 标注含未登记类别` | 标注 label 不在 `classes` 中，或拼写/大小写不一致 | 把 label 加入 `classes`（注意插入位置会改变后续 id）；或修正标注 |
| `AnnotationError: labelme 标注缺失` | 图片旁没有同名 `<stem>.json` | 用 labelme 补标注；空标注写 `"shapes": []`；确实无标注的图片移出数据集 |
| `AnnotationError: 同一 stem 的图片格式冲突` | 同一目录存在 `x.jpg` 与 `x.png` | 只保留一种格式；扫描器不猜测哪个是样本 |
| 图片完全没被扫描到 | 扩展名不在 `parse.image_ext` 中（大小写不敏感），或目录被隐藏 | 补充扩展名（带前导点）；检查图片是否在隐藏目录/隐藏文件中 |
| 扫不到样本 / `SplitError: 没有可切分的样本` | `folder_per_class` 的目录名与 `classes` 不完全一致，或图片扩展名不在白名单 | 核对目录名与 `classes` 字符串（区分大小写）及 `parse.image_ext` |
| `AnnotationError: 扫描得到重复的 rel_path` | 多任务 `task.path` 互相重叠，同一图片被两个任务扫到 | 让各任务目录互不重叠；同一图片只能属于一个任务 |
| `SplitError: N 个样本未被 dir_map 覆盖` | 扁平类目录与 `train_data` 切片目录同时存在，同一图片被扫成两个 `rel_path` | 让 `dir_map` 覆盖全部扫描结果，或只保留切片目录；`dir` 模式下 `scan_samples` 会自动收敛公共子目录并在 stderr 提示 |
| `SplitError: 样本同时出现在 ...` | `list` 清单中同一路径写了两次或跨 split 重复 | 删除重复条目；每个样本只能出现一次 |
| `SplitError: 缺少切分配置` | 规格没写 `splits` | 补 `splits`，或向 `DVXDataset` 传 `split_result` |
| `check_dataset` 报「切分产生的 train 为空」 | `train` 比例为 0 / 样本太少 | 提高 `train` 比例或补充样本 |
| 分割掩码只有 0 | 标注是 `rectangle` 而非 `polygon`，或所有多边形都在 `ignore_labels` 中 | 分割必须用 `polygon`；`__background__` 只用于保留 0 号类，不要用它标目标 |
| 内嵌 `imageData` 报错 | `parse.labelme.image_data: error` | 在 labelme 中关闭"保存图像数据"并重新导出；或改为 `ignore` |
| `recursive: false` 扫不到深层图片 | 语义是"只扫一层"：folder 只看 `<base>/<class>/*`，labelme 只看 `root` 本层 | 需要递归时写 `true`（默认）；多任务的 `<base>` 是 `<root>/<task.path>` |
| 未知字段告警（如 `split:`） | 字段名拼错，配置被忽略 | 对照 §3 修正；告警不会中断解析，但字段不会生效 |

### 8.1 分割的背景类与 `ignore_index`

真实 `seg` 数据集的约定：

```yaml
classes:
  - __background__        # 0 号类
  - ABar
  # ...
seg_mode: semantic
ignore_index: 255
parse:
  labelme:
    ignore_labels: [__ignore__, __background__]
```

- `__background__` 固定为 **0 号类**；`rasterize` 用 `background=0` 填充未被任何多边形覆盖的像素，
  因此"背景"在掩码里就是 0。
- `__background__` 同时出现在 `ignore_labels` 中：标注文件里即使有 `label: __background__`
  的形状也会被丢弃，不会变成目标 id 0。
- `ignore_index: 255` 是**给训练侧**的忽略像素值（`CrossEntropy` / mIoU 等），
  当前 `DVXDataset` 的栅格化固定背景为 0，**尚未消费**该字段；P2 接线训练时需让训练配置的
  `ignore_index` 与规格保持一致，否则背景会被当作前景参与 loss/指标。
- `seg_mode: query` 目前只是 schema 接受的保留值，`DVXDataset` 只实现 `semantic` 栅格化。

---

## 9. 边界说明

- **`data_type` ≠ 训练目标**。规格只描述标注形态；同一份 `detection` 数据可以用于检测，
  也可以用于其他消费 `boxes`/`labels` 的任务，具体由训练任务（P2）决定。
- **尚未接入 `tools/train.py`**。`DVXDataset` 已导出且可编程式使用，但还没有注册到
  `DATASETS`，`build_dataset({...})` 与 `tools/train.py --config` 目前无法按配置构建它；
  训练接线（含 transforms/collate/task 预设的匹配）属于 P2。
- **检索任务尚未实现**。本期没有 retrieval / re-identification 的 spec 类型、样本字段与评估；
  后续如需支持，应新增 `data_type` 而不是复用分类字段。
- **`query` 分割模式尚未实现**。schema 接受 `seg_mode: query` 以便向前兼容，但数据层没有
  对应的 query 输出。
- **`ignore_index` 尚未参与掩码生成**。见 §8.1。
- **切分结果与数据内容绑定**。`SplitResult` 存的是下标，数据集增删样本后必须重新生成；
  `list` 模式则依赖 `rel_path`，增删文件后同样要更新清单。
- **不做格式转换**。`DVXDataset` 不移动、不复制、不重写原始文件；labelme JSON 保持原样，
  分割掩码只在取样本时于内存中生成。
