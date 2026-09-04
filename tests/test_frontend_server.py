"""End-to-end contract for the lightweight static frontend server."""
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_static_server_serves_root_health_and_frontend():
    port = _free_port()
    server = ROOT / "frontend_server.py"
    assert server.exists(), "frontend server has not been implemented"
    process = subprocess.Popen([sys.executable, str(server), "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT)
    try:
        for _ in range(30):
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2) as response:
                    assert response.read() == b'{"status":"healthy"}'
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("static server did not start")
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            assert response.status == 200
            assert b"Card Scanner" in response.read()
        with urlopen(f"http://127.0.0.1:{port}/static/index.html") as response:
            assert response.status == 200
            assert b"<title>Card Scanner</title>" in response.read()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_static_server_serves_https_when_given_a_certificate(tmp_path):
    """The LAN frontend can provide a secure origin for browser camera access."""
    port = _free_port()
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
        "-keyout", str(key), "-out", str(cert),
    ], check=True, capture_output=True)
    process = subprocess.Popen([
        sys.executable, str(ROOT / "frontend_server.py"), "--host", "127.0.0.1", "--port", str(port),
        "--certfile", str(cert), "--keyfile", str(key),
    ], cwd=ROOT)
    try:
        import ssl
        context = ssl._create_unverified_context()
        for _ in range(30):
            try:
                with urlopen(f"https://127.0.0.1:{port}/health", context=context, timeout=0.2) as response:
                    assert response.read() == b'{"status":"healthy"}'
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("HTTPS static server did not start")
    finally:
        process.terminate()
        process.wait(timeout=5)
