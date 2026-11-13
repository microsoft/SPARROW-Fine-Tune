"""``sparrow-finetune`` command line: migrate, create-user, api, worker."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from typing import Optional, Sequence

from .config import Settings


def _parse_resources(value: str) -> set[str]:
    resources = {r.strip() for r in value.split(",") if r.strip()}
    if not resources:
        raise argparse.ArgumentTypeError("at least one resource is required")
    return resources


def _builtin_registry(settings: Settings):
    """The process-wide registry with the built-in pipelines registered."""
    from . import pipelines

    pipelines.register_builtin_pipelines(pipelines.default_registry, settings)
    return pipelines.default_registry


def _create_user(settings: Settings, username: str) -> int:
    import psycopg

    from .auth import MIN_PASSWORD_LENGTH, hash_password
    from .db import connect

    username = username.strip()
    if not username:
        print("error: username must not be empty", file=sys.stderr)
        return 2
    password = os.environ.get("SPARROW_FT_PASSWORD")
    if password is None:
        password = sys.stdin.readline().rstrip("\r\n")
    if len(password) < MIN_PASSWORD_LENGTH:
        print(
            f"error: password must be at least {MIN_PASSWORD_LENGTH} characters",
            file=sys.stderr,
        )
        return 2
    try:
        with connect(settings.database_url) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, hash_password(password)),
            )
    except psycopg.errors.UniqueViolation:
        print(f"error: user {username!r} already exists", file=sys.stderr)
        return 1
    print(f"created user {username}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparrow-finetune", description="Sparrow model fine-tuning service"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="apply pending database migrations")

    create_user = sub.add_parser(
        "create-user",
        help="create a UI account; password from SPARROW_FT_PASSWORD or the first stdin line",
    )
    create_user.add_argument("username")

    api = sub.add_parser("api", help="run the HTTP API")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    worker = sub.add_parser("worker", help="run a training worker")
    worker.add_argument(
        "--resources",
        type=_parse_resources,
        default={"cpu"},
        help="comma-separated resources this worker offers, e.g. cpu,gpu",
    )
    worker.add_argument(
        "--once", action="store_true", help="run at most one job, then exit"
    )
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--lease-seconds", type=float, default=60.0)
    worker.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        if not os.environ.get("SPARROW_FT_DATABASE_URL", "").strip():
            from .db import build_database_url

            url = build_database_url()
            if url is not None:
                os.environ["SPARROW_FT_DATABASE_URL"] = url
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "migrate":
        from .db import migrate

        applied = migrate(settings.database_url)
        print("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
        return 0

    if args.command == "create-user":
        return _create_user(settings, args.username)

    if args.command == "api":
        import uvicorn

        from .api import create_app

        registry = _builtin_registry(settings)
        # proxy_headers=False: uvicorn must not rewrite the peer from X-Forwarded-For; client_key
        # trusts that header only from SPARROW_FT_TRUSTED_PROXIES.
        uvicorn.run(
            create_app(settings, registry),
            host=args.host,
            port=args.port,
            proxy_headers=False,
        )
        return 0

    from .workers import run_forever, run_once

    registry = _builtin_registry(settings)
    if args.once:
        claimed = run_once(
            settings,
            args.worker_id,
            args.resources,
            registry=registry,
            lease_seconds=args.lease_seconds,
        )
        print("ran one job" if claimed else "no job claimed")
        return 0
    try:
        run_forever(
            settings,
            args.worker_id,
            args.resources,
            registry=registry,
            poll_seconds=args.poll_seconds,
            lease_seconds=args.lease_seconds,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
