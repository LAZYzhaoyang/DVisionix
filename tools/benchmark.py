# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 性能基准工具（固定输入 + 可复现，用于「先测后改」的前后对比）。
"""DVisionix 性能基准工具。

设计原则（CodePlan 7.7 要求「先建立 benchmark，再修改实现」）：

- **固定输入**：所有场景使用固定随机种子与固定规模，结果可复现；
- **同时输出正确性指标**：优化前后的数值必须在容差内一致，否则提速无意义；
- **内存需要隔离测量**：``resource.getrusage().ru_maxrss`` 是**进程级峰值**且只增不减，
  因此 ``--isolate`` 会为每个场景单独起一个子进程，得到干净的峰值内存。

用法::

    python tools/benchmark.py --scenario all                     # 进程内计时
    python tools/benchmark.py --scenario all --isolate           # 每场景独立进程（含峰值内存）
    python tools/benchmark.py --scenario pq --isolate --json out.json
    python tools/benchmark.py --list
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

#: name -> (说明, 运行函数)
SCENARIOS: Dict[str, Callable[[], Dict[str, Any]]] = {}
_DESCRIPTIONS: Dict[str, str] = {}


def scenario(name: str, description: str):
    """把一个函数注册为基准场景。"""

    def deco(fn):
        SCENARIOS[name] = fn
        _DESCRIPTIONS[name] = description
        return fn

    return deco


def _peak_rss_mb() -> float:
    """当前进程的峰值常驻内存（MB），跨平台。

    - Unix：``resource.getrusage``（Linux 单位 KB，macOS 单位字节）
    - Windows：``resource`` 不可用，改用 psapi 的 ``PeakWorkingSetSize``
    """
    if sys.platform == "win32":  # pragma: no cover - 平台相关
        import ctypes
        from ctypes import wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 必须声明签名：句柄是 64 位，默认按 c_int 传会截断伪句柄导致调用失败
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        func = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if func is None:  # 旧系统回退到 psapi
            func = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        func.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        func.restype = wintypes.BOOL

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not func(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0)


def _time_call(fn: Callable[[], Any], repeat: int = 3) -> Dict[str, float]:
    """跑 ``repeat`` 次，返回耗时统计（毫秒）与最后一次的返回值。"""
    times: List[float] = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "ms_min": min(times),
        "ms_median": sorted(times)[len(times) // 2],
        "repeat": repeat,
        "_result": result,
    }


class _FixedImageDataset(torch.utils.data.Dataset):
    """固定张量数据集。

    必须在**模块级**定义：``num_workers > 0`` 时 DataLoader 会把数据集对象
    序列化给子进程（Windows 用 spawn），局部类无法 pickle。
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {"image": self.images[index].clone(), "label": self.labels[index]}


# ---------------------------------------------------------------------------
# 场景 1：EMA 回调开销
# ---------------------------------------------------------------------------
@scenario("ema", "EMA 回调相对训练步的额外开销（目标 < 3%）")
def bench_ema() -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from dvisionix.models import SimpleCNN
    from dvisionix.training import ClassificationTask, Trainer
    from dvisionix.training.callbacks import EMA

    steps = 20
    dataset = _FixedImageDataset(torch.zeros(steps * 8, 3, 32, 32), torch.arange(steps * 8) % 3)

    def run(with_ema: bool) -> float:
        torch.manual_seed(0)
        loader = DataLoader(dataset, batch_size=8)
        callbacks = [EMA(decay=0.999)] if with_ema else []
        trainer = Trainer(
            ClassificationTask(num_classes=3),
            loader,
            None,
            callbacks=callbacks,
            max_epochs=1,
            seed=0,
            log_interval=10**9,
        )
        model = SimpleCNN(num_classes=3, in_channels=3)
        start = time.perf_counter()
        trainer.fit(model)
        return (time.perf_counter() - start) * 1000.0

    plain = min(run(False) for _ in range(2))
    with_ema = min(run(True) for _ in range(2))
    overhead = (with_ema - plain) / plain * 100.0 if plain > 0 else float("nan")
    return {
        "metrics": {
            "ms_without_ema": round(plain, 2),
            "ms_with_ema": round(with_ema, 2),
            "ema_overhead_pct": round(overhead, 2),
        }
    }


# ---------------------------------------------------------------------------
# 场景 2：全景质量（PQ）
# ---------------------------------------------------------------------------
def _panoptic_pair(size: int, grid: int, id_scale: int = 1000, shift: int = 3):
    """构造固定的一张全景 GT / 预测（grid×grid 个块状实例）。"""
    ids_gt = np.zeros((size, size), dtype=np.int64)
    ids_pred = np.zeros((size, size), dtype=np.int64)
    step = size // grid
    counter = 0
    for gy in range(grid):
        for gx in range(grid):
            counter += 1
            y0, x0 = gy * step, gx * step
            y1, x1 = y0 + step, x0 + step
            inst = id_scale + counter  # category 1
            ids_gt[y0:y1, x0:x1] = inst
            # 预测：轻微偏移，IoU 仍 > 0.5，可被匹配上
            py0 = min(y0 + shift, size - 1)
            px0 = min(x0 + shift, size - 1)
            ids_pred[py0 : min(y1 + shift, size), px0 : min(x1 + shift, size)] = inst
    return torch.from_numpy(ids_pred), torch.from_numpy(ids_gt)


@scenario("pq", "全景质量 PQ：固定 512×512 / 25 实例的耗时与峰值内存")
def bench_pq() -> Dict[str, Any]:
    from dvisionix.metrics import PanopticQuality

    size, grid = 512, 5
    pred, gt = _panoptic_pair(size, grid)

    def run():
        metric = PanopticQuality(num_categories=3)
        for _ in range(3):
            metric.update(pred, gt)
        return metric.compute()

    timings = _time_call(run, repeat=3)
    return {
        "timings": timings,
        "metrics": {k: round(float(v), 6) for k, v in timings.pop("_result").items()},
    }


# ---------------------------------------------------------------------------
# 场景 3：检测 mAP
# ---------------------------------------------------------------------------
@scenario("map", "检测 mAP：固定 100 图 / 每图 20 框的耗时")
def bench_map() -> Dict[str, Any]:
    from dvisionix.metrics import MeanAveragePrecision

    rng = np.random.default_rng(0)
    num_images, per_image, num_classes = 100, 20, 5
    boxes, scores, labels, gt_boxes, gt_labels = [], [], [], [], []
    for _ in range(num_images):
        base = rng.uniform(0, 400, size=(per_image, 2))
        wh = rng.uniform(10, 80, size=(per_image, 2))
        b = np.concatenate([base, base + wh], axis=1).astype(np.float32)
        s = rng.uniform(0.05, 1.0, size=(per_image,)).astype(np.float32)
        lb = rng.integers(1, num_classes + 1, size=(per_image,)).astype(np.int64)
        boxes.append(torch.from_numpy(b))
        scores.append(torch.from_numpy(s))
        labels.append(torch.from_numpy(lb))
        gt_boxes.append(torch.from_numpy(b[: per_image // 2].copy()))
        gt_labels.append(torch.from_numpy(lb[: per_image // 2].copy()))

    def run():
        metric = MeanAveragePrecision(num_classes=num_classes)
        metric.update(boxes, scores, labels, gt_boxes, gt_labels)
        return metric.compute()

    timings = _time_call(run, repeat=3)
    result = timings.pop("_result")
    return {
        "timings": timings,
        "metrics": {k: round(float(v), 6) for k, v in result.items()},
    }


# ---------------------------------------------------------------------------
# 场景 4：匈牙利匹配器
# ---------------------------------------------------------------------------
@scenario("matcher", "匈牙利匹配：固定 300 查询 × 30 GT 的耗时")
def bench_matcher() -> Dict[str, Any]:
    from dvisionix.models.losses.detection.matcher import HungarianMatcher

    torch.manual_seed(0)
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    pred_logits = torch.randn(300, 6)
    pred_boxes = torch.rand(300, 4)
    gt_boxes = torch.rand(30, 4)
    gt_labels = torch.randint(0, 5, (30,))

    timings = _time_call(lambda: matcher(pred_logits, pred_boxes, gt_boxes, gt_labels), repeat=5)
    result = timings.pop("_result")
    return {
        "timings": timings,
        "metrics": {"matched_pairs": int(len(result[0]))},
    }


# ---------------------------------------------------------------------------
# 场景 5：TaskAligned 分配器
# ---------------------------------------------------------------------------
@scenario("assigner", "TaskAligned 分配：固定 3 层 / 每层 20×20 的耗时")
def bench_assigner() -> Dict[str, Any]:
    from dvisionix.models.losses.detection.assigner import TaskAlignedAssigner

    torch.manual_seed(0)
    assigner = TaskAlignedAssigner(num_classes=5, topk=13)
    strides = [8, 16, 32]
    shapes = [(20, 20), (10, 10), (5, 5)]
    pred_boxes = [torch.rand(h * w, 4) * 100 for h, w in shapes]
    pred_scores = [torch.rand(h * w, 5) for h, w in shapes]
    centers = []
    for (h, w), s in zip(shapes, strides):
        ys = (torch.arange(h) + 0.5) * s
        xs = (torch.arange(w) + 0.5) * s
        cx, cy = torch.meshgrid(xs, ys, indexing="xy")
        centers.append(torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1))
    gt_boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0], [80.0, 80.0, 150.0, 150.0]])
    gt_labels = torch.tensor([1, 2])

    timings = _time_call(
        lambda: assigner.assign(pred_boxes, pred_scores, centers, strides, gt_boxes, gt_labels),
        repeat=5,
    )
    labels, _ = timings.pop("_result")
    return {
        "timings": timings,
        "metrics": {"num_positives": int(sum(int((lb > 0).sum()) for lb in labels))},
    }


# ---------------------------------------------------------------------------
# 场景 6：数据加载吞吐
# ---------------------------------------------------------------------------
@scenario("dataloader", "数据加载：固定 256 张 128×128 的吞吐（num_workers 对比）")
def bench_dataloader() -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    n, size = 256, 128
    dataset = _FixedImageDataset(torch.zeros(n, 3, size, size), torch.arange(n) % 4)

    out: Dict[str, Any] = {"metrics": {}}
    for workers in (0, 2):
        loader = DataLoader(dataset, batch_size=16, num_workers=workers)
        start = time.perf_counter()
        batches = sum(1 for _ in loader)
        elapsed = (time.perf_counter() - start) * 1000.0
        out["metrics"][f"ms_num_workers_{workers}"] = round(elapsed, 2)
        out["metrics"]["batches"] = batches
    return out


@scenario("import", "导入开销：config-only 与完整视觉栈（越短越好）")
def bench_import() -> Dict[str, Any]:
    """在**全新解释器**里测量导入耗时（进程内测量没有意义）。"""
    cases = {
        "config_only": "from dvisionix.config import Config",
        "models": "import dvisionix.models",
        "torch_only": "import torch",
    }
    out: Dict[str, Any] = {"metrics": {}}
    for name, code in cases.items():
        times = []
        for _ in range(3):
            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, timeout=600, cwd=os.getcwd()
            )
            if proc.returncode != 0:
                out["metrics"][f"ms_{name}"] = f"ERROR: {proc.stderr.decode()[-120:]}"
                break
            times.append((time.perf_counter() - start) * 1000.0)
        else:
            out["metrics"][f"ms_{name}"] = round(min(times), 1)
    return out


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
def run_scenario(name: str, repeat: int) -> Dict[str, Any]:
    """在独立进程里跑一个场景，返回含峰值内存的结果。"""
    cmd = [sys.executable, os.path.abspath(__file__), "--scenario", name, "--json-stdout"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return {"scenario": name, "error": proc.stderr.strip()[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description="DVisionix 性能基准")
    parser.add_argument("--scenario", default="all", help="场景名或 all")
    parser.add_argument("--repeat", type=int, default=3, help="每个场景的重复次数")
    parser.add_argument("--isolate", action="store_true", help="每个场景独立子进程（含峰值内存）")
    parser.add_argument("--json", default=None, help="把结果写入 JSON 文件")
    parser.add_argument("--json-stdout", action="store_true", help="仅输出 JSON（内部使用）")
    parser.add_argument("--list", action="store_true", help="列出所有场景")
    args = parser.parse_args()

    if args.list:
        for name, desc in _DESCRIPTIONS.items():
            print(f"{name:12s} {desc}")
        return 0

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"未知场景 {unknown}；可选：{list(SCENARIOS)}")

    results: List[Dict[str, Any]] = []
    for name in names:
        if args.isolate:
            # 子进程只跑这一个场景，其内部的 _peak_rss_mb() 就是该场景的进程峰值
            result = run_scenario(name, args.repeat)
        else:
            payload = SCENARIOS[name]()
            if len(names) == 1:
                # 只跑单个场景时，进程峰值才有归因意义
                payload["peak_rss_mb"] = round(_peak_rss_mb(), 1)
            result = {"scenario": name, **payload}
        result.setdefault("scenario", name)
        results.append(result)

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "isolated": bool(args.isolate),
        "results": results,
    }

    if args.json_stdout:
        print(json.dumps(report))
        return 0

    print(
        f"# DVisionix benchmark  python={report['python']} torch={report['torch']} "
        f"device={report['device']} isolated={report['isolated']}\n"
    )
    for result in results:
        name = result.get("scenario")
        if "error" in result:
            print(f"[{name}] ERROR: {result['error']}")
            continue
        metrics = result.get("metrics", {})
        timings = result.get("timings", {})
        extra = ""
        if "ms_median" in timings:
            extra = f" median={timings['ms_median']:.1f}ms min={timings['ms_min']:.1f}ms"
        if "peak_rss_mb" in result:
            extra += f" peak_rss={result['peak_rss_mb']:.1f}MB（含解释器+torch 基线）"
        print(f"[{name}]{extra}")
        for k, v in metrics.items():
            print(f"    {k} = {v}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
