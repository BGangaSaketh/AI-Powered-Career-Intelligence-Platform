"""
run.py
======
Unified Launcher & Process Orchestrator for AI-Powered Career Intelligence Platform

Launches the Flask Backend API internally, verifies health, launches the
Streamlit User Interface, and presents ONE unified application URL to the user.

Usage:
    python run.py
"""

import os
import sys
import time
import signal
import socket
import urllib.request
import urllib.error
import subprocess
from typing import Optional, List

# Reconfigure stdout/stderr to UTF-8 on Windows if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def load_env_file(env_path: str = ".env"):
    """Simple parser to load .env variables if python-dotenv is not installed."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"[Launcher Warning] Could not parse .env file: {e}")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is currently open/in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def kill_process_on_port(port: int):
    """Attempt to terminate any stale process running on a specific port."""
    if not is_port_in_use(port):
        return
    print(f"[Launcher] Found existing process on port {port}. Cleaning up...")
    try:
        if sys.platform.startswith("win"):
            cmd = f'netstat -ano | findstr :{port}'
            out = subprocess.check_output(cmd, shell=True, text=True, errors="ignore")
            pids = set()
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = parts[-1]
                    if pid.isdigit() and pid != "0":
                        pids.add(pid)
            for pid in pids:
                print(f"[Launcher] Terminating stale Windows process PID {pid} on port {port}...")
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(f'fuser -k {port}/tcp', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f'lsof -ti:{port} | xargs kill -9', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
    except Exception as e:
        print(f"[Launcher Warning] Could not auto-kill process on port {port}: {e}")


def wait_for_backend_health(url: str, timeout_sec: int = 30) -> bool:
    """Poll backend health endpoint until it returns HTTP 200 OK."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CareerIntelLauncher/1.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    load_env_file()

    backend_port = int(os.getenv("PORT") or os.getenv("BACKEND_PORT") or 5000)
    streamlit_port = int(os.getenv("STREAMLIT_PORT") or 8501)
    api_base_url = os.getenv("API_BASE_URL") or f"http://127.0.0.1:{backend_port}"

    os.environ["PORT"] = str(backend_port)
    os.environ["API_BASE_URL"] = api_base_url
    os.environ["STREAMLIT_PORT"] = str(streamlit_port)

    print("=" * 65)
    print("      AI-POWERED CAREER INTELLIGENCE PLATFORM")
    print("=" * 65)
    print(f"[1/4] Checking and clearing stale processes...")
    kill_process_on_port(backend_port)
    kill_process_on_port(streamlit_port)

    print(f"[2/4] Starting Internal Backend API Server (Port {backend_port})...")
    backend_env = os.environ.copy()
    backend_proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=os.getcwd(),
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    import threading
    backend_logs = []
    def drain_backend_logs():
        try:
            for line in iter(backend_proc.stdout.readline, ''):
                if line:
                    backend_logs.append(line)
                    if len(backend_logs) > 500:
                        backend_logs.pop(0)
        except Exception:
            pass
    threading.Thread(target=drain_backend_logs, daemon=True).start()

    health_url = f"http://127.0.0.1:{backend_port}/health"
    print(f"[3/4] Verifying Backend Health ({health_url})...")
    
    if not wait_for_backend_health(health_url, timeout_sec=25):
        if backend_proc.poll() is not None:
            print("\n[ERROR] Backend Server failed to start! Log output:")
            print("".join(backend_logs[-30:]))
        else:
            print(f"\n[ERROR] Backend Server did not respond to health check at {health_url} within timeout.")

            backend_proc.terminate()
        sys.exit(1)

    print("      [OK] Backend API is healthy and operational.")

    print(f"[4/4] Launching Main Streamlit UI Application (Port {streamlit_port})...")
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        f"--server.port={streamlit_port}",
        "--server.headless=true"
    ]
    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=os.getcwd(),
        env=os.environ.copy()
    )

    print("\n" + "=" * 65)
    print("               PLATFORM READY FOR USE")
    print("=" * 65)
    print(f"  MAIN APPLICATION URL:  http://localhost:{streamlit_port}")
    print("=" * 65)
    print("  Open the URL above in your browser to access the full platform.")
    print("  Press Ctrl+C in this terminal to stop the application gracefully.\n")

    def cleanup_subprocesses(signum=None, frame=None):
        print("\n[Launcher] Shutting down services...")
        try:
            if streamlit_proc and streamlit_proc.poll() is None:
                streamlit_proc.terminate()
                streamlit_proc.wait(timeout=3)
        except Exception:
            pass

        try:
            if backend_proc and backend_proc.poll() is None:
                backend_proc.terminate()
                backend_proc.wait(timeout=3)
        except Exception:
            pass
        print("[Launcher] Platform stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_subprocesses)
    signal.signal(signal.SIGTERM, cleanup_subprocesses)

    try:
        while True:
            if backend_proc.poll() is not None:
                print(f"\n[Warning] Backend API process terminated with code {backend_proc.returncode}.")
                cleanup_subprocesses()
            if streamlit_proc.poll() is not None:
                print(f"\n[Warning] Streamlit frontend process terminated with code {streamlit_proc.returncode}.")
                cleanup_subprocesses()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup_subprocesses()


if __name__ == "__main__":
    main()
