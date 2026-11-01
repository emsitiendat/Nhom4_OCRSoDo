import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


# ============================================================
# UTF-8 CONSOLE
# ============================================================

def configure_console_utf8():
    """
    Bao dam heartbeat runner co the in tieng Viet tren Windows.
    Neu terminal khong ho tro reconfigure thi bo qua an toan.
    """
    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except Exception:
        pass

    try:
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except Exception:
        pass


# ============================================================
# TIME FORMAT
# ============================================================

def format_elapsed(seconds):
    seconds = max(0, int(seconds))

    hours, rem = divmod(
        seconds,
        3600
    )

    minutes, secs = divmod(
        rem,
        60
    )

    if hours > 0:
        return (
            f"{hours}h "
            f"{minutes:02d}m "
            f"{secs:02d}s"
        )

    if minutes > 0:
        return (
            f"{minutes}m "
            f"{secs:02d}s"
        )

    return f"{secs}s"


# ============================================================
# HEARTBEAT THREAD
# ============================================================

def heartbeat_worker(
    stop_event,
    label,
    heartbeat_seconds,
    start_time,
):
    """
    In thong bao dinh ky neu process con chay.

    Muc dich:
    - tranh nguoi dung nghi code bi treo;
    - khong can thay doi code thuat toan ben trong;
    - dac biet huu ich voi PaddleOCR / benchmark.
    """

    heartbeat_seconds = max(
        float(heartbeat_seconds),
        3.0
    )

    while not stop_event.wait(
        heartbeat_seconds
    ):
        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            "\n"
            "[DANG CHAY] "
            f"{label} van dang xu ly "
            f"| da chay {format_elapsed(elapsed)}",
            flush=True,
        )


# ============================================================
# RUN CHILD PROCESS
# ============================================================

def run_child(
    script,
    label,
    heartbeat_seconds,
):
    script_path = Path(script)

    if not script_path.exists():
        print(
            f"[LOI] Khong tim thay file: "
            f"{script_path}",
            flush=True,
        )
        return 2

    # --------------------------------------------------------
    # Moi truong child process
    # --------------------------------------------------------
    env = os.environ.copy()

    # Python log khong buffer
    env["PYTHONUNBUFFERED"] = "1"

    # Quan trong tren Windows:
    # tranh UnicodeEncodeError cp1252/cp1258
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Paddle Windows CPU
    env.setdefault(
        "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT",
        "False",
    )

    env.setdefault(
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
        "True",
    )

    # --------------------------------------------------------
    # Dung chinh Python hien tai dang chay heartbeat
    # Neu heartbeat duoc goi bang .venv\Scripts\python.exe,
    # child cung se dung cung .venv.
    # --------------------------------------------------------
    command = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    print(
        "=" * 62,
        flush=True,
    )

    print(
        f"BAT DAU: {label}",
        flush=True,
    )

    print(
        f"Python : {sys.executable}",
        flush=True,
    )

    print(
        f"Script : {script_path}",
        flush=True,
    )

    print(
        "=" * 62,
        flush=True,
    )

    start_time = time.perf_counter()

    stop_event = threading.Event()

    heartbeat_thread = threading.Thread(
        target=heartbeat_worker,
        args=(
            stop_event,
            label,
            heartbeat_seconds,
            start_time,
        ),
        daemon=True,
    )

    heartbeat_thread.start()

    try:
        process = subprocess.Popen(
            command,
            env=env,
        )

        return_code = process.wait()

    except KeyboardInterrupt:
        print(
            "\n[HUY] Nguoi dung da dung chuong trinh.",
            flush=True,
        )

        try:
            process.terminate()
        except Exception:
            pass

        return_code = 130

    except Exception as exc:
        print(
            "\n"
            "[LOI] Khong khoi dong duoc "
            f"{label}: {exc}",
            flush=True,
        )

        return_code = 1

    finally:
        stop_event.set()

        heartbeat_thread.join(
            timeout=2.0
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        "=" * 62,
        flush=True,
    )

    if return_code == 0:
        print(
            f"[OK] {label} hoan thanh.",
            flush=True,
        )
    else:
        print(
            f"[LOI] {label} ket thuc "
            f"voi ma loi {return_code}",
            flush=True,
        )

    print(
        "Thoi gian: "
        f"{format_elapsed(elapsed)}",
        flush=True,
    )

    print(
        "=" * 62,
        flush=True,
    )

    return return_code


# ============================================================
# MAIN
# ============================================================

def main():
    configure_console_utf8()

    parser = argparse.ArgumentParser(
        description=(
            "Chay Python script voi heartbeat "
            "de nguoi dung biet chuong trinh "
            "van dang xu ly."
        )
    )

    parser.add_argument(
        "script",
        help="Duong dan Python script can chay.",
    )

    parser.add_argument(
        "--label",
        default="Python task",
        help="Ten hien thi cua tac vu.",
    )

    parser.add_argument(
        "--heartbeat",
        type=float,
        default=12.0,
        help=(
            "So giay giua cac thong bao "
            "heartbeat. Mac dinh 12 giay."
        ),
    )

    args = parser.parse_args()

    return run_child(
        script=args.script,
        label=args.label,
        heartbeat_seconds=args.heartbeat,
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )