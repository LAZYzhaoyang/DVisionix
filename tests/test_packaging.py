# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 打包契约测试：wheel 内容与「干净安装后可用性」。
"""打包契约测试（CodePlan 7.6 步骤 4-1 / 7.1.2 D14 + D15）。

v1.0.0 的 `setup.py` 用 `find_packages()` 且没有任何 `package_data` /
`include_package_data` / `MANIFEST.in`，结果是：

- `dvisionix/config/defaults/*.yaml` **不进 wheel**，而 `Config.from_default()`
  从**安装目录**读取它们 → 非 editable 安装下必然失败；
- `tests*` 反而会被打进 wheel。

这两个问题都不会被 CI 发现，因为 CI 用的是 `pip install -e .`（editable，读源码树）。
本文件把 wheel 的内容与「装完能不能用」变成可执行断言。
"""

import os
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: wheel 里必须存在的内置默认配置（Config.from_default 依赖它们）
REQUIRED_DEFAULT_CONFIGS = (
    "dvisionix/config/defaults/base.yaml",
    "dvisionix/config/defaults/classification.yaml",
    "dvisionix/config/defaults/detection.yaml",
    "dvisionix/config/defaults/segmentation.yaml",
)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """构建一次 wheel，供本模块的用例复用。"""
    out_dir = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:  # pragma: no cover - 仅在构建环境损坏时触发
        pytest.fail(f"构建 wheel 失败（exit {proc.returncode}）：\n{proc.stderr[-2000:]}")
    wheels = sorted(out_dir.glob("*.whl"))
    assert wheels, f"pip wheel 未产出任何 wheel，输出：\n{proc.stdout[-2000:]}"
    return wheels[-1]


@pytest.mark.integration
@pytest.mark.slow
def test_wheel_ships_default_configs_and_excludes_tests(built_wheel):
    """D14：内置默认配置必须进 wheel，tests 必须不进。"""
    names = zipfile.ZipFile(built_wheel).namelist()

    missing = [name for name in REQUIRED_DEFAULT_CONFIGS if name not in names]
    assert not missing, (
        f"wheel 缺少内置默认配置 {missing}；"
        f"Config.from_default() 从安装目录读取它们，非 editable 安装会直接失败"
    )

    test_entries = [n for n in names if n.startswith("tests/")]
    assert not test_entries, f"wheel 不应包含测试代码，但发现 {len(test_entries)} 个条目"

    assert any("LICENSE" in n.upper() for n in names), "wheel 应包含 LICENSE"


@pytest.mark.integration
@pytest.mark.slow
def test_wheel_declares_runtime_dependencies(built_wheel):
    """D15：依赖必须由 pyproject 单一声明并进 METADATA（不再依赖 requirements.txt）。"""
    with zipfile.ZipFile(built_wheel) as archive:
        metadata_name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

    for dep in ("torch", "torchvision", "numpy", "opencv-python", "pyyaml"):
        assert f"Requires-Dist: {dep}" in metadata, f"METADATA 缺少运行时依赖 {dep}"
    assert "Requires-Python: >=3.10" in metadata, "Requires-Python 应为 >=3.10"


@pytest.mark.integration
@pytest.mark.slow
def test_installed_wheel_loads_default_config(built_wheel, tmp_path):
    """把 wheel 装进干净目录后，`Config.from_default()` 必须真的可用。

    用 ``--no-deps`` 安装（避免再去下载 torch），只验证 wheel 自身的载荷。
    子进程的 cwd 设在仓库之外，确保 `dvisionix` 解析到的是**安装副本**而不是源码树。
    """
    target = tmp_path / "site-packages"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--quiet",
            "--target",
            str(target),
            str(built_wheel),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert install.returncode == 0, f"安装 wheel 失败：\n{install.stderr[-2000:]}"

    probe = tmp_path / "probe_wheel.py"
    probe.write_text(
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from dvisionix.config import Config\n"
        "cfg = Config.from_default('classification')\n"
        "assert cfg.task_type == 'classification', cfg.task_type\n"
        "print('FROM_DEFAULT_OK')\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    # cwd 放到仓库之外：否则源码树里的 dvisionix 会先被解析到
    proc = subprocess.run(
        [sys.executable, str(probe), str(target)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"安装后加载默认配置失败：\n{proc.stdout}\n{proc.stderr}"
    assert "FROM_DEFAULT_OK" in proc.stdout
