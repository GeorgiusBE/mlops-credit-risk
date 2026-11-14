"""Start the local MLflow server with SQLite metadata and proxied artifacts."""

import argparse
import errno
import os
from pathlib import Path
import socket
import sys


def main():
    """Serve the tracking UI and API on loopback using persistent local storage.

    Resolve storage relative to this repository, regardless of the caller's
    working directory. The server receives artifact uploads over HTTP and
    writes them beneath --data-dir; clients do not need filesystem access.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".mlflow",
        help="Directory containing mlflow.db and artifacts (default: repository/.mlflow).",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    # Check before MLflow initializes so an existing instance produces a useful
    # message instead of a slow startup followed by an address-in-use traceback.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        other_port = 5001 if args.port == 5000 else 5000
        parser.exit(
            1,
            f"Port {args.port} is already in use; no new server was started.\n"
            f"If MLflow is already running, open http://localhost:{args.port}\n"
            "Stop the existing server before restarting it, or use another port:\n"
            f"  python scripts/start_mlflow.py --port {other_port}\n"
            f"For training on that port, set MLFLOW_TRACKING_URI=http://127.0.0.1:{other_port}\n",
        )

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = data_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    database_uri = f"sqlite:///{data_dir / 'mlflow.db'}"
    command = [
        sys.executable, "-m", "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--workers", "1",
        "--backend-store-uri", database_uri,
        "--serve-artifacts",
        "--artifacts-destination", str(artifacts_dir),
    ]
    print(f"MLflow UI: http://127.0.0.1:{args.port}", flush=True)
    print(f"Metadata: {data_dir / 'mlflow.db'}", flush=True)
    print(f"Artifacts: {artifacts_dir}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
