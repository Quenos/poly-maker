import argparse
import logging
from mm.state import StateStore


logger = logging.getLogger("mm.cli")


def export_pnl(date: str) -> None:
    st = StateStore()
    realized, unrealized, fees, rebates = st.export_pnl_for_date(date)
    logger.info(
        "date=%s realized=%.2f unrealized=%.2f fees=%.2f rebates=%.2f",
        date,
        realized,
        unrealized,
        fees,
        rebates,
    )


def rebuild_positions() -> None:
    st = StateStore()
    st.rebuild_positions_from_fills()
    logger.info("Positions rebuilt from fills.")


def main() -> None:
    # Ensure basic logging for CLI runs if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="MM CLI")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("export_pnl")
    p1.add_argument("--date", required=True)

    sub.add_parser("rebuild_positions")

    args = parser.parse_args()
    if args.cmd == "export_pnl":
        export_pnl(args.date)
    elif args.cmd == "rebuild_positions":
        rebuild_positions()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
