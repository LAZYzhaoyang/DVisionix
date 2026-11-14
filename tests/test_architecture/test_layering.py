# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分层架构门禁（P0-4）：检查器自测 + 真实仓库零违规。
"""分层架构门禁测试。

两部分：

1. **真实仓库门禁**：对当前仓库跑 ``tools/check_layering.py``，要求零违规。
2. **检查器自测**：在 ``tmp_path`` 造迷你包与迷你 manifest，验证 R1/R2/R3
   都能被抓到、登记例外后放行、相对导入被正确绝对化、CLI 退出码符合契约。

自测用例不依赖真实仓库目录结构，也不导入 torch，保持秒级。
"""

import json

import pytest
import yaml

from tools import check_layering


def _manifest(exceptions: list[dict] | None = None) -> dict:
    """构造检查器自测用的迷你分层清单。"""
    return {
        "version": 1,
        "description": "分层检查器自测清单",
        "layers": [
            {"level": 0, "name": "assembly", "packages": ["dvisionix.assembly"]},
            {
                "level": 1,
                "name": "component",
                "packages": ["dvisionix.component", "dvisionix.component2"],
            },
            {"level": 2, "name": "foundation", "packages": ["dvisionix.foundation"]},
            {"level": 3, "name": "shared", "modules": ["dvisionix.shared"]},
            {"level": 4, "name": "infrastructure", "modules": ["dvisionix.registry"]},
        ],
        "rules": {
            "R1": "只允许 import 同级或更底层（目标 level >= 自身 level）",
            "R2": "同层、不同组件包之间禁止互相 import",
            "R3": "组件包内部，非 __init__.py 的兄弟模块互相 import 必须登记例外",
        },
        "exceptions": exceptions or [],
    }


def _write_repo(tmp_path, files: dict[str, str], exceptions: list[dict] | None = None):
    """在 tmp_path 下造一个迷你仓库与清单，返回 (root, manifest_path)。"""
    root = tmp_path / "repo"
    (root / "dvisionix").mkdir(parents=True, exist_ok=True)
    (root / "dvisionix" / "__init__.py").write_text("", encoding="utf-8")
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    manifest_path = root / "docs" / "architecture" / "layers.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(_manifest(exceptions), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return root, manifest_path


def _check(tmp_path, files: dict[str, str], exceptions: list[dict] | None = None) -> dict:
    """造迷你仓库并执行检查，返回报告字典。"""
    root, manifest_path = _write_repo(tmp_path, files, exceptions)
    return check_layering.check_repository(str(root), str(manifest_path))


@pytest.mark.unit
def test_repository_has_no_layering_violation(repo_root):
    """真实仓库必须零违规（P0-4 硬性门禁）。"""
    report = check_layering.check_repository(repo_root)

    assert report["violations"] == []
    assert report["checked_files"] > 0


@pytest.mark.unit
def test_upward_import_detected(tmp_path):
    """foundation（level 2）import component（level 1）属于向上依赖，R1 必须抓到。"""
    files = {
        "dvisionix/component/__init__.py": "",
        "dvisionix/component/ops.py": "VALUE = 1\n",
        "dvisionix/foundation/__init__.py": "",
        "dvisionix/foundation/base.py": "from ..component.ops import VALUE\n",
    }
    report = _check(tmp_path, files)

    assert [item["rule"] for item in report["violations"]] == ["R1"]
    violation = report["violations"][0]
    assert violation["file"] == "dvisionix/foundation/base.py"
    assert violation["line"] == 1
    assert violation["target"] == "dvisionix.component.ops"


@pytest.mark.unit
def test_downward_import_is_allowed(tmp_path):
    """assembly（level 0）import foundation（level 2）是正常组合，不应报违规。"""
    files = {
        "dvisionix/assembly/__init__.py": "",
        "dvisionix/assembly/model.py": "from ..foundation.ops import VALUE\n",
        "dvisionix/foundation/__init__.py": "",
        "dvisionix/foundation/ops.py": "VALUE = 1\n",
    }
    report = _check(tmp_path, files)

    assert report["violations"] == []
    assert len(report["exceptions_used"]) == 0


@pytest.mark.unit
def test_same_level_cross_package_detected(tmp_path):
    """同层不同组件包互相 import，R2 必须抓到。"""
    files = {
        "dvisionix/component/__init__.py": "",
        "dvisionix/component/a.py": "from ..component2.b import VALUE\n",
        "dvisionix/component2/__init__.py": "",
        "dvisionix/component2/b.py": "VALUE = 1\n",
    }
    report = _check(tmp_path, files)

    assert [item["rule"] for item in report["violations"]] == ["R2"]
    assert report["violations"][0]["target"] == "dvisionix.component2.b"


@pytest.mark.unit
def test_sibling_import_without_exception_detected(tmp_path):
    """组件包内兄弟模块 import 未登记例外，R3 必须抓到。"""
    files = {
        "dvisionix/component/__init__.py": "",
        "dvisionix/component/a.py": "from .b import VALUE\n",
        "dvisionix/component/b.py": "VALUE = 1\n",
    }
    report = _check(tmp_path, files)

    assert [item["rule"] for item in report["violations"]] == ["R3"]
    violation = report["violations"][0]
    assert violation["file"] == "dvisionix/component/a.py"
    assert violation["target"] == "dvisionix.component.b"


@pytest.mark.unit
@pytest.mark.parametrize(
    "exception",
    [
        {
            "from": "dvisionix/component/a.py",
            "to": "dvisionix/component/b.py",
            "rule": "R3",
            "reason": "自测：精确路径例外",
        },
        {
            "from_glob": "dvisionix/component/*.py",
            "to_glob": "dvisionix/component/b.py",
            "rule": "R3",
            "reason": "自测：通配路径例外",
        },
    ],
)
def test_registered_exception_allows_sibling_import(tmp_path, exception):
    """命中 from/to 或 from_glob/to_glob 例外时，R3 违规应被放行并计入统计。"""
    files = {
        "dvisionix/component/__init__.py": "",
        "dvisionix/component/a.py": "from .b import VALUE\n",
        "dvisionix/component/b.py": "VALUE = 1\n",
    }
    report = _check(tmp_path, files, exceptions=[exception])

    assert report["violations"] == []
    assert len(report["exceptions_used"]) == 1
    assert report["exceptions_used"][0]["rule"] == "R3"
    assert report["exceptions_used"][0]["hits"] == 1


@pytest.mark.unit
def test_shared_layer_target_always_allowed(tmp_path):
    """目标为共享层（level 3/4）时 R1 始终放行。"""
    files = {
        "dvisionix/shared.py": "VALUE = 1\n",
        "dvisionix/registry.py": "from .shared import VALUE\n",
    }
    report = _check(tmp_path, files)

    assert report["violations"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("package", "level", "module", "expected"),
    [
        (
            "dvisionix.models.backbones",
            1,
            "feature",
            "dvisionix.models.backbones.feature",
        ),
        (
            "dvisionix.models.backbones",
            2,
            "layers.window_attention",
            "dvisionix.models.layers.window_attention",
        ),
        ("dvisionix.models.backbones", 3, "registry", "dvisionix.registry"),
        ("dvisionix.models.heads.detection", 4, "registry", "dvisionix.registry"),
        ("dvisionix.models.heads.detection", 3, "layers", "dvisionix.models.layers"),
        (
            "dvisionix.models.heads.detection",
            1,
            "yolo",
            "dvisionix.models.heads.detection.yolo",
        ),
        ("dvisionix", 2, None, None),
    ],
)
def test_resolve_relative_import(package, level, module, expected):
    """相对导入必须被正确绝对化（含超出层级时返回 None）。"""
    assert check_layering.resolve_relative_import(package, level, module) == expected


@pytest.mark.unit
def test_module_name_for_file_and_package_init():
    """文件与 __init__.py 都要映射到正确的模块名。"""
    assert (
        check_layering.module_name("dvisionix/models/backbones/convnext.py")
        == "dvisionix.models.backbones.convnext"
    )
    assert (
        check_layering.module_name("dvisionix/models/backbones/__init__.py")
        == "dvisionix.models.backbones"
    )


@pytest.mark.unit
def test_cli_exit_codes_and_json(tmp_path, capsys):
    """CLI：有违规且非 --no-fail 时退出 1；--json 输出契约字段；--no-fail 退出 0。"""
    files = {
        "dvisionix/component/__init__.py": "",
        "dvisionix/component/ops.py": "VALUE = 1\n",
        "dvisionix/foundation/__init__.py": "",
        "dvisionix/foundation/base.py": "from ..component.ops import VALUE\n",
    }
    root, manifest_path = _write_repo(tmp_path, files)

    code = check_layering.main(["--root", str(root), "--manifest", str(manifest_path), "--json"])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert set(payload) == {"violations", "checked_files", "exceptions_used"}
    assert payload["violations"][0]["rule"] == "R1"

    code = check_layering.main(["--root", str(root), "--manifest", str(manifest_path), "--no-fail"])
    captured = capsys.readouterr()
    assert code == 0
    assert "处" in captured.out


@pytest.mark.unit
def test_cli_passes_on_clean_repository(tmp_path, capsys):
    """干净仓库：CLI 输出通过文案且退出码 0。"""
    files = {
        "dvisionix/assembly/__init__.py": "",
        "dvisionix/foundation/__init__.py": "",
        "dvisionix/foundation/ops.py": "VALUE = 1\n",
        "dvisionix/assembly/model.py": "from ..foundation.ops import VALUE\n",
    }
    root, manifest_path = _write_repo(tmp_path, files)

    code = check_layering.main(["--root", str(root), "--manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "未发现分层违规" in captured.out
