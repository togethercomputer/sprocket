import http.server
import threading


def run_health_and_metrics_server_in_background() -> None:
    "very minimal way to serve /health and /metrics from the filesystem"
    handler = lambda *a: http.server.SimpleHTTPRequestHandler(*a, directory="http")
    server = http.server.HTTPServer(("", 8000), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def set_utilization(busy: bool) -> None:
    "update prometheus metric to 1.0 or 0.0"
    open("http/metrics", "w").write(f"utilization {float(busy)}\n")
