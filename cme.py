"""
CBOE delayed options quote fetcher.

Pulls ~15-minute delayed options chain data from CBOE's public CDN endpoints.
No API key or account required. Data is the same as shown on cboe.com quote pages.

Covers index options (SPX, NDX, RUT, VIX, DJX) and major ETF options (SPY, QQQ, IWM).

Usage:
    python cme.py                      # SPX front-month chain
    python cme.py --symbol NDX         # Nasdaq-100 index options
    python cme.py --symbol SPY         # SPY ETF options
    python cme.py --symbol SPX --exp 1 # Second expiry (0-indexed)
    python cme.py --exps               # List all available expirations
    python cme.py --raw                # Dump raw JSON for inspection
    python cme.py --save               # Save chain to CSV
"""

import argparse
import json
import sys
from datetime import datetime, date

import pandas as pd
import requests

# Index symbols require a leading underscore in the CDN URL.
# ETF options use the ticker directly.
KNOWN_SYMBOLS = {
    "SPX": "_SPX",   # S&P 500 Index options
    "NDX": "_NDX",   # Nasdaq-100 Index options
    "RUT": "_RUT",   # Russell 2000 Index options
    "VIX": "_VIX",   # CBOE Volatility Index options
    "DJX": "_DJX",   # Dow Jones Index options
    "SPY": "SPY",    # SPDR S&P 500 ETF options
    "QQQ": "QQQ",    # Invesco QQQ ETF options
    "IWM": "IWM",    # iShares Russell 2000 ETF options
}

BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cboe.com/",
}


def fetch_chain(cdn_symbol: str) -> dict:
    """Fetch the full options chain JSON from CBOE CDN."""
    url = f"{BASE_URL}/{cdn_symbol}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_option_symbol(sym: str) -> tuple[str, date, str, float]:
    """
    Parse OCC option symbol into (underlying, expiry, type, strike).

    Format: {underlying}{YYMMDD}{C|P}{8-digit strike*1000}
    Example: SPX260320C00200000 -> ('SPX', date(2026,3,20), 'C', 200.0)
    """
    suffix = sym[-15:]
    underlying = sym[:-15]
    yy, mm, dd = int(suffix[0:2]), int(suffix[2:4]), int(suffix[4:6])
    opt_type = suffix[6]
    strike = int(suffix[7:]) / 1000.0
    expiry = date(2000 + yy, mm, dd)
    return underlying, expiry, opt_type, strike


def _session(underlying_prefix: str, base_symbol: str) -> str:
    """
    Return 'AM' or 'PM' based on the option's underlying prefix.

    On monthly expirations, CBOE lists both AM-settled (e.g. SPX, NDX) and
    PM-settled (e.g. SPXW, NDXP) contracts on the same date. The base symbol
    (no suffix) is AM; any variant with a trailing letter is PM.
    """
    return "AM" if underlying_prefix == base_symbol else "PM"


def parse_chain(raw: dict, base_symbol: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse raw CBOE JSON into a tidy DataFrame plus underlying info dict.

    AM/PM settlement is derived from the option prefix: the canonical symbol
    (e.g. SPX) is AM-settled; variants (e.g. SPXW) are PM-settled.
    On dates where only one settlement type exists, session is still set
    (typically PM for weekly expirations).

    Returns (df, underlying_info).
    """
    data = raw.get("data", {})
    options = data.get("options", [])

    underlying_info = {
        "symbol":        raw.get("symbol") or data.get("symbol"),
        "current_price": data.get("current_price"),
        "price_change":  data.get("price_change"),
        "price_change_pct": data.get("price_change_percent"),
        "iv30":          data.get("iv30"),
        "iv30_change":   data.get("iv30_change"),
        "timestamp":     raw.get("timestamp"),
    }

    rows = []
    for opt in options:
        sym = opt.get("option", "")
        try:
            prefix, expiry, opt_type, strike = parse_option_symbol(sym)
        except Exception:
            continue

        rows.append({
            "expiry":        expiry,
            "session":       _session(prefix, base_symbol),
            "strike":        strike,
            "type":          "CALL" if opt_type == "C" else "PUT",
            "bid":           opt.get("bid"),
            "ask":           opt.get("ask"),
            "last":          opt.get("last_trade_price"),
            "change":        opt.get("change"),
            "volume":        opt.get("volume"),
            "open_interest": opt.get("open_interest"),
            "iv":            opt.get("iv"),
            "delta":         opt.get("delta"),
            "gamma":         opt.get("gamma"),
            "theta":         opt.get("theta"),
            "vega":          opt.get("vega"),
            "rho":           opt.get("rho"),
            "theo":          opt.get("theo"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["expiry", "session", "strike", "type"]).reset_index(drop=True)
    return df, underlying_info


def expiry_label(exp: date, session: str, df: pd.DataFrame) -> str:
    """
    Return a display label for an (expiry, session) pair.

    Appends the session only when both AM and PM exist on that date,
    e.g. '2026-03-20 AM' / '2026-03-20 PM'. Otherwise just the date.
    """
    sessions_on_date = df[df["expiry"] == exp]["session"].unique()
    if len(sessions_on_date) > 1:
        return f"{exp.isoformat()} {session}"
    return exp.isoformat()


def main():
    parser = argparse.ArgumentParser(description="Fetch CBOE delayed options quotes.")
    parser.add_argument("--symbol", default="SPX",
                        help=f"Symbol (default: SPX). Available: {', '.join(KNOWN_SYMBOLS)}")
    parser.add_argument("--exp", default=0, type=int,
                        help="Expiry index, 0=front month (default: 0)")
    parser.add_argument("--exps", action="store_true",
                        help="List all available expirations and exit")
    parser.add_argument("--raw", action="store_true",
                        help="Dump raw chain JSON and exit")
    parser.add_argument("--save", action="store_true",
                        help="Save filtered chain to CSV")
    parser.add_argument("--all-exps", action="store_true",
                        help="Show/save all expirations, not just the selected one")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if symbol not in KNOWN_SYMBOLS:
        print(f"Unknown symbol '{symbol}'. Available: {', '.join(KNOWN_SYMBOLS)}")
        sys.exit(1)

    cdn_symbol = KNOWN_SYMBOLS[symbol]
    print(f"Fetching {symbol} options chain from CBOE (delayed ~15 min)...")

    try:
        raw = fetch_chain(cdn_symbol)
    except Exception as e:
        print(f"Failed to fetch chain: {e}")
        sys.exit(1)

    if args.raw:
        print(json.dumps(raw, indent=2))
        return

    df, underlying = parse_chain(raw, base_symbol=symbol)

    if df.empty:
        print("No data parsed.")
        sys.exit(1)

    # Print underlying snapshot
    print(f"\n{symbol}  price={underlying['current_price']}  "
          f"chg={underlying['price_change']} ({underlying['price_change_pct']:.2f}%)  "
          f"IV30={underlying['iv30']:.2f}  as-of={underlying['timestamp']}")

    # Build sorted list of (expiry, session) pairs — AM before PM on same date
    expirations = sorted(df.groupby(["expiry", "session"]).groups.keys())

    # List expirations mode
    if args.exps:
        print(f"\nAvailable expirations ({len(expirations)}):")
        today = date.today()
        for i, (exp, ses) in enumerate(expirations):
            dte = (exp - today).days
            mask = (df["expiry"] == exp) & (df["session"] == ses)
            n_strikes = df[mask]["strike"].nunique()
            label = expiry_label(exp, ses, df)
            marker = " <-- selected" if i == args.exp else ""
            print(f"  [{i:3d}] {label:20s}  DTE={dte:3d}  strikes={n_strikes}{marker}")
        return

    # Select expiry
    if args.all_exps:
        filtered = df
        chosen_label = "ALL"
    else:
        if args.exp >= len(expirations):
            print(f"--exp {args.exp} out of range (max {len(expirations)-1}). "
                  f"Run --exps to list available expirations.")
            sys.exit(1)
        chosen_exp, chosen_ses = expirations[args.exp]
        mask = (df["expiry"] == chosen_exp) & (df["session"] == chosen_ses)
        filtered = df[mask].copy()
        dte = (chosen_exp - date.today()).days
        chosen_label = f"{expiry_label(chosen_exp, chosen_ses, df)} (DTE={dte})"

        # Show expiry list context
        print(f"\nAvailable expirations ({len(expirations)} total):")
        for i, (exp, ses) in enumerate(expirations[:10]):
            label = expiry_label(exp, ses, df)
            marker = " <-- selected" if i == args.exp else ""
            print(f"  [{i:2d}] {label}{marker}")
        if len(expirations) > 10:
            print(f"  ... and {len(expirations) - 10} more (use --exps to list all)")

    n_strikes = filtered["strike"].nunique()
    print(f"\nOptions chain — {symbol}  expiry={chosen_label}")
    print(f"  {n_strikes} strikes, {len(filtered)} rows (calls + puts)\n")

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 140)
    print(filtered.drop(columns=["expiry", "session"]).to_string(index=False))

    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_tag = chosen_label.replace(" ", "").replace("(", "").replace(")", "").replace("=", "")
        fname = f"{symbol}_options_{exp_tag}_{ts}.csv"
        filtered.to_csv(fname, index=False)
        print(f"\nSaved to {fname}")


if __name__ == "__main__":
    main()
