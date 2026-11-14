# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVisionix 分层架构静态检查器（P0-3；仅用 ast + PyYAML，不 import torch）。
"""DVisionix 分层架构静态检查器。

分层清单 ``docs/architecture/layers.yaml`` 是 R1-R3 的单一事实来源，本工具只做
AST 静态解析，不导入 torch，也不执行被检查的模块，因此可在秒级跑完。

用法::

    python tools/check_layering.py [--root DIR] [--manifest FILE] [--json] [--no-fail]

规则：

- R1：只允许 import 同级或更底层（目标 level >= 自身 level）；目标为 level 3/4
  的共享层时始终允许。
- R2：同层、不同组件包之间禁止互相 import。
- R3：组件包（manifest 中 name 为 ``component`` 的层）内部，非 ``__init__.py``
  的模块互相 import 必须登记 exceptions。

退出码：有违规且未指定 ``--no-fail`` 时为 1，否则为 0；清单/目录错误为 2。
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fnmatch
import json
import os
import sys
from typing import Any, Iterable

import yaml

#: 规则名 -> 人类可读说明
RULE_DESCRIPTIONS = {
    "R1": "只允许 import 同级或更底层（目标 level >= 自身 level）；目标 level 3/4 始终允许",
    "R2": "同层、不同组件包之间禁止互相 import",
    "R3": "组件包内部兄弟模块互相 import 必须登记例外",
}

#: 默认仓库根目录（本脚本位于 <root>/tools/check_layering.py）
DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 被检查的包名
PACKAGE_NAME = "dvisionix"

#: 组件层的 name（R3 只作用于该层）
COMPONENT_LAYER_NAME = "component"

#: 共享层等级（目标为这些 level 时 R1 始终放行）
SHARED_LEVELS = frozenset({3, 4})


class ManifestError(ValueError):
    """分层清单缺失或格式不合法。

    这是开发者工具自身的参数/配置错误（退出码 2），不属于 DVisionix 领域异常，
    因此直接继承 ``ValueError``，避免检查器反向依赖 ``dvisionix`` 包。
    """


@dataclasses.dataclass(frozen=True)
class LayerEntry:
    """分层清单中的一条条目（一个包或一个模块）。

    Attributes:
        level: 层级编号，0 为最上层（assembly），数字越大越底层。
        name: 层的语义名（assembly / component / foundation / shared / infrastructure）。
        unit: 该条目对应的顶层模块名（如 ``dvisionix.models.heads``），用于判定 R2/R3。
        kind: ``package`` 或 ``module``；两者都按"模块本身或其子模块"匹配。
    """

    level: int
    name: str
    unit: str
    kind: str


@dataclasses.dataclass(frozen=True)
class ImportRef:
    """一条 import 语句解析出的目标模块。"""

    module: str
    line: int


@dataclasses.dataclass
class ExceptionSpec:
    """一条已登记例外；路径均为相对仓库根的 posix 路径。"""

    rule: str
    reason: str
    from_path: str | None = None
    from_glob: str | None = None
    to_path: str | None = None
    to_glob: str | None = None
    hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好的字典（含命中次数）。"""
        data: dict[str, Any] = {"rule": self.rule}
        if self.from_path is not None:
            data["from"] = self.from_path
        if self.from_glob is not None:
            data["from_glob"] = self.from_glob
        if self.to_path is not None:
            data["to"] = self.to_path
        if self.to_glob is not None:
            data["to_glob"] = self.to_glob
        data["reason"] = self.reason
        data["hits"] = self.hits
        return data


@dataclasses.dataclass
class Manifest:
    """分层清单的内存表示。"""

    path: str
    entries: list[LayerEntry]
    exceptions: list[ExceptionSpec]

    def find_layer(self, module: str) -> LayerEntry | None:
        """把模块名映射到分层条目；无法映射时返回 None。"""
        for entry in sorted(self.entries, key=lambda item: len(item.unit), reverse=True):
            if module == entry.unit or module.startswith(entry.unit + "."):
                return entry
        return None


def _as_str_list(value: Any, field: str) -> list[str]:
    """把 YAML 字段校验为字符串列表。"""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"分层清单字段 {field} 必须是字符串列表，实际为 {value!r}")
    return value


def load_manifest(path: str) -> Manifest:
    """读取并校验分层清单。

    Args:
        path: ``layers.yaml`` 路径。

    Returns:
        解析后的 :class:`Manifest`。

    Raises:
        ManifestError: 文件不存在、YAML 语法错误或字段不合法。
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ManifestError(f"分层清单不存在：{path}") from exc
    except OSError as exc:
        raise ManifestError(f"无法读取分层清单 {path}：{exc}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"分层清单 YAML 解析失败 {path}：{exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"分层清单根节点必须是映射，实际为 {type(raw).__name__}")
    if raw.get("version") != 1:
        raise ManifestError(f"分层清单 version 必须为 1，实际为 {raw.get('version')!r}")

    layers_raw = raw.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ManifestError("分层清单 layers 必须是非空列表")

    entries: list[LayerEntry] = []
    seen_units: dict[str, LayerEntry] = {}
    for item in layers_raw:
        if not isinstance(item, dict):
            raise ManifestError(f"layers 条目必须是映射，实际为 {item!r}")
        level = item.get("level")
        name = item.get("name")
        if not isinstance(level, int) or isinstance(level, bool):
            raise ManifestError(f"layer.level 必须是整数，实际为 {level!r}")
        if not isinstance(name, str) or not name:
            raise ManifestError(f"layer.name 必须是非空字符串，实际为 {name!r}")
        packages = _as_str_list(item.get("packages"), "packages")
        modules = _as_str_list(item.get("modules"), "modules")
        if not packages and not modules:
            raise ManifestError(f"layer {name!r}（level {level}）没有 packages/modules")
        layer_entries = [
            LayerEntry(level=level, name=name, unit=unit, kind="package") for unit in packages
        ] + [LayerEntry(level=level, name=name, unit=unit, kind="module") for unit in modules]
        for entry in layer_entries:
            if entry.unit in seen_units:
                raise ManifestError(f"分层清单重复登记模块：{entry.unit}")
            seen_units[entry.unit] = entry
            entries.append(entry)

    exceptions_raw = raw.get("exceptions") or []
    if not isinstance(exceptions_raw, list):
        raise ManifestError("exceptions 必须是列表")
    exceptions: list[ExceptionSpec] = []
    for item in exceptions_raw:
        if not isinstance(item, dict):
            raise ManifestError(f"exceptions 条目必须是映射，实际为 {item!r}")
        rule = item.get("rule")
        if rule not in RULE_DESCRIPTIONS:
            raise ManifestError(f"例外 rule 必须是 R1/R2/R3 之一，实际为 {rule!r}")
        from_path = item.get("from")
        from_glob = item.get("from_glob")
        to_path = item.get("to")
        to_glob = item.get("to_glob")
        if (from_path is None) == (from_glob is None):
            raise ManifestError("例外必须且只能给 from 或 from_glob 之一")
        if (to_path is None) == (to_glob is None):
            raise ManifestError("例外必须且只能给 to 或 to_glob 之一")
        if not isinstance(from_path or from_glob, str) or not isinstance(to_path or to_glob, str):
            raise ManifestError("例外的 from/from_glob/to/to_glob 必须是字符串")
        reason = item.get("reason") or ""
        if not isinstance(reason, str):
            raise ManifestError("例外 reason 必须是字符串")
        exceptions.append(
            ExceptionSpec(
                rule=rule,
                reason=reason,
                from_path=from_path,
                from_glob=from_glob,
                to_path=to_path,
                to_glob=to_glob,
            )
        )

    return Manifest(path=path, entries=entries, exceptions=exceptions)


def module_name(rel_path: str) -> str:
    """由相对路径推导模块名；``__init__.py`` 映射为包名。

    例如 ``dvisionix/models/backbones/convnext.py`` -> ``dvisionix.models.backbones.convnext``；
    ``dvisionix/models/backbones/__init__.py`` -> ``dvisionix.models.backbones``。
    """
    parts = rel_path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def source_package(rel_path: str, module: str) -> str:
    """返回源文件用于相对导入解析的包名。

    普通模块取其父包；``__init__.py`` 自身即包名。
    """
    if rel_path.endswith("/__init__.py"):
        return module
    return module.rsplit(".", 1)[0]


def resolve_relative_import(package: str, level: int, module: str | None) -> str | None:
    """把相对导入解析为绝对模块名。

    Args:
        package: 源文件所属包（如 ``dvisionix.models.backbones``）。
        level: 相对层级（``ast.ImportFrom.level``，1 表示当前包）。
        module: ``ast.ImportFrom.module``，可能为 None（``from . import X``）。

    Returns:
        绝对模块名；层级越界时返回 None。

    Examples:
        >>> resolve_relative_import("dvisionix.models.backbones", 1, "feature")
        'dvisionix.models.backbones.feature'
        >>> resolve_relative_import("dvisionix.models.backbones", 3, "registry")
        'dvisionix.registry'
    """
    if level < 1:
        return None
    parts = package.split(".")
    if level > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def iter_import_targets(
    tree: ast.AST,
    package: str,
    known_modules: set[str],
) -> Iterable[ImportRef]:
    """遍历 AST，产出所有绝对化的 import 目标模块。

    - ``ast.Import``：每个 alias 的完整模块名（绝对导入）。
    - ``ast.ImportFrom``：相对导入用 :func:`resolve_relative_import` 绝对化；
      ``from . import X`` 若 ``X`` 对应同包模块则解析为该子模块，否则退回包本身。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportRef(module=alias.name, line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    yield ImportRef(module=node.module, line=node.lineno)
                continue
            base = resolve_relative_import(package, node.level, node.module)
            if base is None:
                continue
            if node.module is not None:
                yield ImportRef(module=base, line=node.lineno)
                continue
            matched = [
                f"{base}.{alias.name}"
                for alias in node.names
                if f"{base}.{alias.name}" in known_modules
            ]
            if matched:
                for target in matched:
                    yield ImportRef(module=target, line=node.lineno)
            else:
                yield ImportRef(module=base, line=node.lineno)


def _glob_match(rel_path: str, exact: str | None, pattern: str | None) -> bool:
    """匹配例外中的路径：精确路径或 fnmatch 通配（``**`` 先替换为 ``*``）。"""
    if exact is not None:
        return rel_path == exact.replace("\\", "/")
    assert pattern is not None
    return fnmatch.fnmatchcase(rel_path, pattern.replace("**", "*"))


def find_exception(
    exceptions: list[ExceptionSpec],
    source_rel: str,
    target_rel: str,
    rule: str,
) -> ExceptionSpec | None:
    """查找能豁免某条违规的例外（按清单顺序取第一条）。"""
    for spec in exceptions:
        if spec.rule != rule:
            continue
        if not _glob_match(source_rel, spec.from_path, spec.from_glob):
            continue
        if not _glob_match(target_rel, spec.to_path, spec.to_glob):
            continue
        return spec
    return None


def _evaluate_rules(
    source: LayerEntry,
    target: LayerEntry,
    source_is_init: bool,
) -> tuple[str, str] | None:
    """判定一对（源层, 目标层）是否违规；违规时返回 (规则名, 说明)。"""
    # R1：只允许依赖同层或更底层；目标为共享层（level 3/4）时始终允许。
    # 注意：分层编号 0=最上层、4=最底层，因此"向上依赖"是 target.level < source.level。
    if target.level < source.level and target.level not in SHARED_LEVELS:
        message = (
            f"向上依赖：{source.unit}（level {source.level}）"
            f" -> {target.unit}（level {target.level}）"
        )
        return "R1", message

    if target.level != source.level:
        return None

    # R2：同层但属于不同组件包/不同顶层包。
    if source.unit != target.unit:
        message = (
            f"同层（level {source.level}）不同包互相 import：" f"{source.unit} -> {target.unit}"
        )
        return "R2", message

    # R3：组件层内部，非 __init__.py 的兄弟模块互相 import。
    if not source_is_init and source.name == COMPONENT_LAYER_NAME and source.kind == "package":
        message = f"组件包 {source.unit} 内部兄弟模块 import 未登记例外"
        return "R3", message

    return None


def check_repository(root: str, manifest_path: str | None = None) -> dict[str, Any]:
    """对仓库执行完整分层检查。

    Args:
        root: 仓库根目录绝对路径。
        manifest_path: 分层清单路径；为 None 时取
            ``<root>/docs/architecture/layers.yaml``。

    Returns:
        契约 P0-3 规定的报告字典：``violations`` / ``checked_files`` /
        ``exceptions_used``。

    Raises:
        ManifestError: 清单或包目录不存在、清单格式错误。
    """
    root = os.path.abspath(root)
    if manifest_path is None:
        manifest_path = os.path.join(root, "docs", "architecture", "layers.yaml")
    manifest = load_manifest(manifest_path)

    package_root = os.path.join(root, PACKAGE_NAME)
    if not os.path.isdir(package_root):
        raise ManifestError(f"未找到待检查的包目录：{package_root}")

    rel_files: list[str] = []
    for current_dir, _dir_names, file_names in os.walk(package_root):
        for file_name in file_names:
            if not file_name.endswith(".py"):
                continue
            abs_path = os.path.join(current_dir, file_name)
            rel_files.append(os.path.relpath(abs_path, root).replace(os.sep, "/"))
    rel_files.sort()

    module_to_rel: dict[str, str] = {}
    for rel_path in rel_files:
        module_to_rel.setdefault(module_name(rel_path), rel_path)
    known_modules = set(module_to_rel)

    violations: list[dict[str, Any]] = []
    for rel_path in rel_files:
        source_module = module_name(rel_path)
        source_layer = manifest.find_layer(source_module)
        if source_layer is None:
            # 只检查能映射到 manifest 层级的模块（如 models/__init__、data、training 跳过）。
            continue
        if rel_path.endswith("/__init__.py"):
            # R3 的例外主体是 __init__.py：聚合导出不属于兄弟模块互相依赖。
            source_is_init = True
        else:
            source_is_init = False

        abs_path = os.path.join(root, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as handle:
                source_text = handle.read()
            tree = ast.parse(source_text, filename=abs_path)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[警告] 无法读取 {rel_path}：{exc}", file=sys.stderr)
            continue
        except SyntaxError as exc:
            print(f"[警告] 无法解析 {rel_path}:{exc.lineno}：{exc.msg}", file=sys.stderr)
            continue

        package = source_package(rel_path, source_module)
        for ref in iter_import_targets(tree, package, known_modules):
            if ref.module == source_module:
                continue
            target_layer = manifest.find_layer(ref.module)
            if target_layer is None:
                continue
            evaluation = _evaluate_rules(source_layer, target_layer, source_is_init)
            if evaluation is None:
                continue
            rule, message = evaluation
            target_rel = module_to_rel.get(ref.module, ref.module.replace(".", "/") + ".py")
            exception = find_exception(manifest.exceptions, rel_path, target_rel, rule)
            if exception is not None:
                exception.hits += 1
                continue
            violations.append(
                {
                    "file": rel_path,
                    "line": ref.line,
                    "target": ref.module,
                    "target_file": target_rel,
                    "rule": rule,
                    "message": message,
                }
            )

    violations.sort(key=lambda item: (item["file"], item["line"], item["rule"], item["target"]))
    exceptions_used = [spec.to_dict() for spec in manifest.exceptions if spec.hits > 0]
    return {
        "violations": violations,
        "checked_files": len(rel_files),
        "exceptions_used": exceptions_used,
    }


def format_report(report: dict[str, Any], root: str, manifest_path: str) -> str:
    """把检查报告格式化为人类可读文本（文件:行 → 目标 → 违反规则）。"""
    lines = [
        f"分层检查：{os.path.join(root, PACKAGE_NAME)}",
        f"分层清单：{manifest_path}",
        f"扫描文件：{report['checked_files']} 个",
    ]
    used = report["exceptions_used"]
    if used:
        total_hits = sum(item.get("hits", 0) for item in used)
        lines.append(f"命中登记例外：{len(used)} 条 / 共 {total_hits} 次")

    violations = report["violations"]
    if not violations:
        lines.append("[通过] 未发现分层违规。")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{'文件:行':<60} {'目标模块':<62} 规则")
    lines.append("-" * 100)
    for item in violations:
        location = f"{item['file']}:{item['line']}"
        lines.append(f"{location:<60} {item['target']:<62} {item['rule']}")
        lines.append(f"    说明：{item['message']}")
    lines.append("")
    lines.append(f"[失败] 共 {len(violations)} 处分层违规；详见 {manifest_path}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="DVisionix 分层架构静态检查器（R1-R3，AST 解析，不 import torch）",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="仓库根目录（默认本脚本上两级）",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="分层清单路径（默认 <root>/docs/architecture/layers.yaml）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出检查结果",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="有违规时仍返回退出码 0（仅用于报告）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口；返回进程退出码。"""
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root) if args.root else DEFAULT_ROOT
    manifest_path = (
        os.path.abspath(args.manifest)
        if args.manifest
        else os.path.join(root, "docs", "architecture", "layers.yaml")
    )

    try:
        report = check_repository(root, manifest_path)
    except ManifestError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report, root, manifest_path))

    if report["violations"] and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
