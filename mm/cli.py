import argparse
from mm.state import StateStore


def export_pnl(date: str) -> None:
    st = StateStore()
    realized, unrealized, fees, rebates = st.export_pnl_for_date(date)
    print(f"date={date} realized={realized:.2f} unrealized={unrealized:.2f} fees={fees:.2f} rebates={rebates:.2f}")


def rebuild_positions() -> None:
    st = StateStore()
    st.rebuild_positions_from_fills()
    print("Positions rebuilt from fills.")


def main() -> None:
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
