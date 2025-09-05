import argparse
from mm.market_maker import start_market_maker


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Polymarket Market Maker")
    parser.add_argument("--test", action="store_true", help="Dry run: no orders sent, debug logs to file")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging without enabling dry-run")
    args = parser.parse_args()
    start_market_maker(test_mode=bool(args.test), debug=bool(args.debug))

if __name__ == "__main__":
    main()

