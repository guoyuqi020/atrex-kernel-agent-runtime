import sys

from .cli import check_main, client_main, server_main


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if command == "--check":
        check_main()
    elif command in {"server", "client"}:
        del sys.argv[1]
        (server_main if command == "server" else client_main)()
    else:
        raise SystemExit(
            "Usage: python -m atrex_gateway_standalone [--check | server ... | client ...]"
        )


if __name__ == "__main__":
    main()
