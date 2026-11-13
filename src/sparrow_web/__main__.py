"""Run the web app: ``python -m sparrow_web``.

For development and for a quick look. Production uses gunicorn with a single
worker and threads::

    gunicorn -k gthread -w 1 --threads 16 --timeout 0 \\
             -b 0.0.0.0:8080 "sparrow_web.app:create_app()"

One worker is not a performance choice — the scheduler, the GPU lease bookkeeping
and the SSE subscriber list are per-process singletons.
"""

from __future__ import annotations

import argparse
import os

from .app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(prog="sparrow-web", description="SPARROW Workbench")
    parser.add_argument("--host", default=os.environ.get("SPARROW_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SPARROW_PORT", "8080")))
    parser.add_argument("--no-supervisor", action="store_true",
                        help="Do not start the background scheduler and pollers.")
    args = parser.parse_args()

    app = create_app(start_supervisor=not args.no_supervisor)
    print(f"SPARROW Workbench on http://{args.host}:{args.port}", flush=True)
    # use_reloader=False always: the reloader forks, which would start a second
    # scheduler competing for the same GPU lease.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
