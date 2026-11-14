"""Check the local MLflow server and the GraphQL API used by experiment pages."""

import argparse
import json
import os
import sys
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


def check_server(tracking_uri):
    """Return the server version after checking health and the experiment UI API.

    Checking /health alone misses GraphQL dependency failures that can leave the
    tracking API working while experiment pages fail. Make no changes to runs,
    models, or server processes. Bypass proxies for local loopback connections.
    """
    base = tracking_uri.rstrip("/")
    hostname = urlparse(base).hostname
    opener = build_opener(ProxyHandler({})) if hostname in ("localhost", "127.0.0.1", "::1") else build_opener()
    with opener.open(base + "/health", timeout=3) as response:
        if response.read().decode().strip() != "OK":
            raise ValueError("The health endpoint did not return OK.")
    with opener.open(base + "/version", timeout=3) as response:
        server_version = response.read().decode().strip()
    request = Request(
        base + "/graphql",
        data=json.dumps({"query": "{ __typename }"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=3) as response:
        if response.headers.get_content_type() != "application/json":
            raise ValueError("The experiment UI API returned a non-JSON response.")
        payload = json.load(response)
    if payload.get("errors") or not (payload.get("data") or {}).get("__typename"):
        raise ValueError("The experiment UI API reported a GraphQL error.")
    return server_version


def main():
    """Print readiness or a useful error, returning a nonzero status on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    if urlparse(args.tracking_uri).scheme not in ("http", "https"):
        parser.error("--tracking-uri must be an HTTP or HTTPS server URL.")
    try:
        server_version = check_server(args.tracking_uri)
    except (OSError, ValueError) as exc:
        print(f"MLflow check failed at {args.tracking_uri}: {exc}", file=sys.stderr)
        print("For local tracking, start the server in another terminal with make mlflow.", file=sys.stderr)
        return 1
    print(f"MLflow {server_version}: server and experiment UI API are ready.")
    print(f"UI: {args.tracking_uri.rstrip('/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
