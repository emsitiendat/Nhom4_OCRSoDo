from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED_PYTHON = (3, 10)

SOURCE_FILES = [
    ROOT / "src" / "tuan1_dataset_tool_auto.py",
    ROOT / "src" / "tuan1_dataset_tool_semi_auto.py",
    ROOT / "src" / "tuan2_preprocess.py",
    ROOT / "src" / "tuan3_baseline.py",
    ROOT / "src" / "tuan4_pipeline.py",
    ROOT / "src" / "tuan5_evaluate.py",
    ROOT / "src" / "tuan7_app.py",
]

REQUIRED_PATHS = [
    ROOT / "dataset",
    ROOT / "src",
    ROOT / "requirements-release.txt",
    ROOT / "setup_windows.bat",
    ROOT / "run_project.bat",
]

PROCESSED_PATHS = [
    ROOT / "dataset" / "processed_real" / "text",
    ROOT / "dataset" / "processed_real" / "diagram",
    ROOT / "dataset" / "processed_real" / "seal",
    ROOT / "dataset" / "processed_real" / "full_clahe",
]

PACKAGE_CHECKS = [
    ("numpy", "numpy"),
    ("cv2", "opencv-contrib-python"),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("streamlit", "streamlit"),
    ("paddle", "paddlepaddle"),
    ("paddleocr", "paddleocr"),
]

EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "cv2": "4.10.0",
    "sklearn": "1.7.2",
    "skimage": "0.25.2",
    "scipy": "1.15.3",
    "pandas": "2.3.3",
    "streamlit": "1.65.0",
    "paddle": "3.3.1",
    "paddleocr": "3.7.0",
}


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def get_version(module) -> str | None:
    return getattr(module, "__version__", None)


def looks_like_app_control_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "application control policy has blocked this file" in text
        or "dll load failed" in text
        or "blocked this file" in text
    )


def main() -> int:
    fatal = 0
    warnings = 0

    section("OCR SO DO / SO HONG - SYSTEM CHECK")

    print(f"Project root : {ROOT}")
    print(f"OS           : {platform.platform()}")
    print(f"Python exe   : {sys.executable}")
    print(f"Python       : {platform.python_version()}")
    print(f"64-bit       : {platform.architecture()[0]}")

    # 1. Python
    section("1. PYTHON")

    if sys.version_info[:2] == EXPECTED_PYTHON:
        ok(f"Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} detected.")
    else:
        fail(
            f"Project requires Python 3.10, but current interpreter is "
            f"{platform.python_version()}."
        )
        fatal += 1

    expected_venv = ROOT / ".venv" / "Scripts" / "python.exe"
    try:
        same_python = expected_venv.exists() and Path(sys.executable).resolve() == expected_venv.resolve()
    except OSError:
        same_python = False

    if same_python:
        ok("Running from project .venv.")
    else:
        warn("This check is not running from project .venv.")
        print("       Recommended command:")
        print(r"       .venv\Scripts\python.exe check_system.py")
        warnings += 1

    # 2. Project files
    section("2. RELEASE FILES")

    for path in REQUIRED_PATHS:
        if path.exists():
            ok(str(path.relative_to(ROOT)))
        else:
            fail(f"Missing: {path.relative_to(ROOT)}")
            fatal += 1

    for path in SOURCE_FILES:
        if path.exists():
            ok(str(path.relative_to(ROOT)))
        else:
            fail(f"Missing source: {path.relative_to(ROOT)}")
            fatal += 1

    # 3. Compile source
    section("3. SOURCE COMPILE")

    for path in SOURCE_FILES:
        if not path.exists():
            continue
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            ok(path.name)
        else:
            fail(f"{path.name} does not compile.")
            if result.stderr.strip():
                print(result.stderr.strip())
            fatal += 1

    # 4. Packages
    section("4. PYTHON PACKAGES")

    loaded = {}

    for module_name, package_name in PACKAGE_CHECKS:
        try:
            module = importlib.import_module(module_name)
            loaded[module_name] = module
            version = get_version(module)
            if version:
                expected = EXPECTED_VERSIONS.get(module_name)
                if expected and str(version) != expected:
                    warn(
                        f"{package_name}: {version} "
                        f"(release reference: {expected})"
                    )
                    warnings += 1
                else:
                    ok(f"{package_name}: {version}")
            else:
                ok(f"{package_name}: import OK")
        except Exception as exc:
            fail(f"{package_name}: import failed.")
            print(f"       {type(exc).__name__}: {exc}")
            if looks_like_app_control_error(exc):
                print()
                print("       Windows Application Control / security policy may be")
                print("       blocking a native DLL or .pyd file.")
                print("       This is an OS security-policy issue, not a Python")
                print("       syntax error in the project.")
                print("       Do NOT disable all Windows Defender protections.")
            fatal += 1

    # OpenCV capabilities
    cv2 = loaded.get("cv2")
    if cv2 is not None:
        if hasattr(cv2, "SIFT_create"):
            ok("OpenCV SIFT available.")
        else:
            fail("OpenCV SIFT is NOT available.")
            fatal += 1

        if hasattr(cv2, "ORB_create"):
            ok("OpenCV ORB available.")
        else:
            fail("OpenCV ORB is NOT available.")
            fatal += 1

    # 5. pip check
    section("5. DEPENDENCY CONSISTENCY")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )

    if pip_check.returncode == 0:
        ok("pip check: no broken requirements found.")
    else:
        fail("pip check found dependency problems.")
        print((pip_check.stdout or pip_check.stderr).strip())
        fatal += 1

    # 6. Tesseract
    section("6. TESSERACT")

    local_tesseract = ROOT / "Tesseract" / "tesseract.exe"
    local_tessdata = ROOT / "Tesseract" / "tessdata"
    local_vie = local_tessdata / "vie.traineddata"
    path_tesseract = shutil.which("tesseract")

    tesseract_cmd = None
    source = None

    if local_tesseract.exists():
        tesseract_cmd = str(local_tesseract)
        source = r"Tesseract\tesseract.exe"
    elif path_tesseract:
        tesseract_cmd = path_tesseract
        source = path_tesseract

    if tesseract_cmd:
        try:
            test = subprocess.run(
                [tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if test.returncode == 0:
                first_line = (test.stdout or test.stderr).splitlines()[0]
                ok(f"Tesseract executable OK: {source}")
                print(f"       {first_line}")
            else:
                warn(f"Tesseract exists but cannot run: {source}")
                if (test.stderr or test.stdout).strip():
                    print("       " + (test.stderr or test.stdout).strip().splitlines()[0])
                warnings += 1
        except Exception as exc:
            warn(f"Tesseract exists but execution failed: {source}")
            print(f"       {type(exc).__name__}: {exc}")
            warnings += 1
    else:
        warn("Tesseract executable not found.")
        print(r"       Expected local path: Tesseract\tesseract.exe")
        print("       Only Week 3 Tesseract baseline may be affected.")
        warnings += 1

    if local_tesseract.exists():
        if local_vie.exists():
            ok(r"Vietnamese language data: Tesseract\tessdata\vie.traineddata")
        else:
            warn(r"Missing: Tesseract\tessdata\vie.traineddata")
            print("       Week 3 OCR with lang='vie' may fail.")
            warnings += 1

    # 7. Dataset
    section("7. DATASET")

    dataset_root = ROOT / "dataset"
    if dataset_root.exists():
        ok("dataset/")
    else:
        fail("dataset/ is missing.")
        fatal += 1

    processed_ready = True
    for path in PROCESSED_PATHS:
        if path.exists():
            count = sum(1 for p in path.iterdir() if p.is_file())
            ok(f"{path.relative_to(ROOT)} ({count} files)")
        else:
            warn(f"Missing preprocess output: {path.relative_to(ROOT)}")
            processed_ready = False
            warnings += 1

    if not processed_ready:
        print()
        print("       Week 3 / 4 / 5 / 7 may require Week 2 preprocess first.")

    # 8. Paddle/PaddleOCR final status
    section("8. OCR ENGINE")

    if "paddle" in loaded:
        ok("PaddlePaddle core import passed.")
    else:
        fail("PaddlePaddle core is unavailable.")

    if "paddleocr" in loaded:
        ok("PaddleOCR import passed.")
        print("       Model initialization/download is performed when Week 4/5/7 runs.")
    else:
        fail("PaddleOCR is unavailable.")

    # Summary
    section("SUMMARY")

    if fatal == 0:
        print("SYSTEM READY")
        print(f"Fatal errors : 0")
        print(f"Warnings     : {warnings}")
        print()
        print("Next:")
        print("  run_project.bat")
        return 0

    print("SYSTEM NOT READY")
    print(f"Fatal errors : {fatal}")
    print(f"Warnings     : {warnings}")
    print()
    print("Fix the [FAIL] items before running the project.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
