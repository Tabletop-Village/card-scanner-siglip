#!/usr/bin/env python3
"""Dependency-free static frontend server with a liveness endpoint.

This intentionally serves the UI independently of model/vector initialization,
which lets an operator validate and deploy the frontend while CUDA indexes are
built separately.
"""
from __future__ import annotations

import argparse
import ssl
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"healthy"}'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/":
            self.path = "/static/index.html"
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--certfile", help="PEM TLS certificate for HTTPS")
    parser.add_argument("--keyfile", help="PEM TLS private key for HTTPS")
    args = parser.parse_args()
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("--certfile and --keyfile must be supplied together")
    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    scheme = "http"
    if args.certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.certfile, args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"Static frontend listening at {scheme}://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
