#!/usr/bin/env python3
"""
Split the Olist e-commerce CSVs into chronological, referentially consistent
batches for a daily-ingestion pipeline.

Batch k can be loaded after batches 1..k-1 with foreign keys enabled and will
never cause an orphan / FK error.

Rules
-----
1. orders      -> split by order_purchase_timestamp into N chronological batches
                  of ~equal row count (or one batch per calendar day with
                  --by-date). An order is never split across batches.
2. order_items -> same batch as their order_id.
3. customers   -> same batch as the EARLIEST order that references them.
                  Customers with no orders go to batch 1.
4. products    -> EARLIEST batch whose order_items reference them (first
                  appearance). Unreferenced products go to batch 1.
5. translation -> kept as ONE file (olist_category_translation.csv), loaded
                  first. Product categories missing from it are reported,
                  never dropped.
6. Original column order / header names are kept. Everything is read as
   dtype=str (no NA parsing) so zip codes keep leading zeros and nothing is
   reformatted. Files are written UTF-8, index=False, "\\n" line endings.

Output layout
-------------
<output-dir>/olist_category_translation.csv
<output-dir>/batch_1/{olist_customers,olist_orders,olist_order_items,
                      olist_products}_dataset.csv
<output-dir>/batch_2/...
<output-dir>/batches_manifest.csv      (batch -> date range and row counts)

Usage
-----
python split_olist.py --input-dir ./data --output-dir ./output
python split_olist.py --input-dir ./data --output-dir ./output -n 5
python split_olist.py --input-dir ./data --output-dir ./output --by-date
python split_olist.py --input-dir ./data --output-dir ./output --validate-only

Exit codes: 0 = all checks passed, 1 = validation failed, 2 = bad input/usage.
Requires pandas >= 1.5.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
OUT_TRANSLATION = "olist_category_translation.csv"
BATCH_TABLES = ["customers", "products", "orders", "items"]  # load order
PK = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "items": ["order_id", "order_item_id"],
    "products": ["product_id"],
    "translation": ["product_category_name"],
}
TS = "order_purchase_timestamp"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read(path):
    # utf-8-sig: strips a UTF-8 BOM if the file has one (the Olist translation
    # file does), so the first header name is clean. keep_default_na=False
    # keeps empty cells as "" instead of NaN.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def write(df, path):
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def warn(msg):
    print(f"  WARNING: {msg}")


def batch_dir(out, k, n):
    return out / f"batch_{str(k).zfill(len(str(n))) if n > 9 else k}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# input checks
# --------------------------------------------------------------------------- #
def check_input_files(in_dir):
    print("== Input checks ==")
    missing = [n for n in FILES.values() if not (in_dir / n).exists()]
    if missing:
        print(f"ERROR: missing input file(s) in {in_dir}: {', '.join(missing)}")
        sys.exit(2)

    hashes = {}
    for name in FILES.values():
        hashes.setdefault(sha256(in_dir / name), []).append(name)
    for names in hashes.values():
        if len(names) > 1:
            warn(f"these input files have identical content: {', '.join(names)}")

    expected = set(FILES.values())
    for p in sorted(in_dir.glob("*.csv")):
        if p.name in expected:
            continue
        for name in expected:
            if p.stem.startswith(Path(name).stem):
                warn(f"'{p.name}' looks like an extra copy of '{name}' "
                     f"(only '{name}' is used)")


def check_duplicate_keys(data):
    dup_counts = {}
    for t, df in data.items():
        n_key = int(df.duplicated(PK[t], keep="first").sum())
        n_row = int(df.duplicated(keep="first").sum())
        dup_counts[t] = n_key
        if n_key:
            warn(f"{FILES[t]}: {n_key} duplicate key(s) on {PK[t]} "
                 f"({n_row} fully identical duplicate row(s))")
    if not any(dup_counts.values()):
        print("  no duplicate keys in any input table")
    return dup_counts


def check_source_orphans(data):
    """Report FK problems already present in the source (cannot be fixed by splitting)."""
    c, o, i, p = data["customers"], data["orders"], data["items"], data["products"]
    checks = [
        ("orders.customer_id not in customers", ~o["customer_id"].isin(c["customer_id"])),
        ("order_items.order_id not in orders", ~i["order_id"].isin(o["order_id"])),
        ("order_items.product_id not in products", ~i["product_id"].isin(p["product_id"])),
    ]
    for label, mask in checks:
        if mask.any():
            warn(f"SOURCE DATA: {int(mask.sum())} row(s) with {label}; "
                 f"no split can make this consistent")


def report_missing_categories(products, translation):
    known = set(translation["product_category_name"])
    cats = products["product_category_name"]
    bad = cats[(cats != "") & ~cats.isin(known)]
    print("\n== Category translation coverage ==")
    if bad.empty:
        print("  every product_category_name exists in the translation table")
    else:
        for name, cnt in bad.value_counts().items():
            print(f"  missing from translation: {name!r} ({cnt} product(s)) - kept, not dropped")
    empty = int((cats == "").sum())
    if empty:
        print(f"  {empty} product(s) have an empty product_category_name (kept as-is)")


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def assign_order_batches(orders, n, by_date):
    """Return (Series order_id -> batch number 1..N, N, per-order frame)."""
    first = orders.drop_duplicates("order_id", keep="first")[["order_id", TS]].copy()
    first["_dt"] = pd.to_datetime(first[TS], errors="coerce")
    first["_missing"] = first["_dt"].isna()
    if first["_missing"].any():
        warn(f"{int(first['_missing'].sum())} order(s) have a missing/unparseable "
             f"{TS}; they are placed in the LAST batch")
    first = first.sort_values(["_missing", "_dt", "order_id"], kind="mergesort")

    if by_date:
        day = first["_dt"].dt.strftime("%Y-%m-%d").where(~first["_missing"], "unknown-date")
        codes, uniques = pd.factorize(day)  # first-appearance order == chronological
        batches = codes + 1
        n_eff = len(uniques)
    else:
        total = len(first)
        if n > total:
            print(f"ERROR: cannot make {n} batches from {total} orders")
            sys.exit(2)
        q, r = divmod(total, n)
        sizes = [q + (1 if i < r else 0) for i in range(n)]
        batches = np.repeat(np.arange(1, n + 1), sizes)
        n_eff = n
    first["_batch"] = batches
    return pd.Series(batches, index=first["order_id"].values), n_eff, first


def split(data, n, by_date, out):
    customers, orders, items, products = (data[t] for t in ("customers", "orders", "items", "products"))

    order_batch, n, per_order = assign_order_batches(orders, n, by_date)

    ob_orders = orders["order_id"].map(order_batch).astype(int).to_numpy()

    ib = items["order_id"].map(order_batch)
    n_orphan_items = int(ib.isna().sum())
    if n_orphan_items:
        warn(f"{n_orphan_items} order_items row(s) reference an unknown order_id; "
             f"placed in batch 1 (validation will flag them)")
    ib = ib.fillna(1).astype(int).to_numpy()

    # customers: earliest batch among the orders that reference them; else 1
    cust_first = pd.Series(ob_orders, index=orders["customer_id"].values).groupby(level=0).min()
    cb = customers["customer_id"].map(cust_first).fillna(1).astype(int).to_numpy()

    # products: earliest batch among the order_items that reference them; else 1
    prod_first = pd.Series(ib, index=items["product_id"].values).groupby(level=0).min()
    pb = products["product_id"].map(prod_first).fillna(1).astype(int).to_numpy()

    assign = {"customers": cb, "orders": ob_orders, "items": ib, "products": pb}

    if out.exists():
        for old in out.glob("batch_*"):
            shutil.rmtree(old)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for k in range(1, n + 1):
        d = batch_dir(out, k, n)
        d.mkdir(parents=True, exist_ok=True)
        for t in BATCH_TABLES:
            write(data[t][assign[t] == k], d / FILES[t])
        sub = per_order[per_order["_batch"] == k]
        manifest.append({
            "batch": k,
            "folder": d.name,
            "first_purchase": sub["_dt"].min(),
            "last_purchase": sub["_dt"].max(),
            **{t: int((assign[t] == k).sum()) for t in BATCH_TABLES},
        })
    write(data["translation"], out / OUT_TRANSLATION)
    write(pd.DataFrame(manifest), out / "batches_manifest.csv")
    return n


# --------------------------------------------------------------------------- #
# validation (re-reads what was actually written to disk)
# --------------------------------------------------------------------------- #
def validate(data, out, n, source_dups):
    print("\n== Validation (re-reading written files) ==")
    errors = []
    cols = {t: list(data[t].columns) for t in data}

    # translation
    tr_path = out / OUT_TRANSLATION
    if not tr_path.exists():
        errors.append(f"{OUT_TRANSLATION} was not written")
    else:
        tr = read(tr_path)
        if len(tr) != len(data["translation"]):
            errors.append("translation row count differs from original")
        if list(tr.columns) != cols["translation"]:
            errors.append("translation columns differ from original")

    cum = {"customers": set(), "products": set(), "orders": set()}
    keyframes = {t: [] for t in BATCH_TABLES}
    counts = []

    for k in range(1, n + 1):
        d = batch_dir(out, k, n)
        b = {}
        for t in BATCH_TABLES:
            f = d / FILES[t]
            if not f.exists():
                errors.append(f"{d.name}/{FILES[t]} is missing")
                b[t] = pd.DataFrame(columns=cols[t])
                continue
            b[t] = read(f)
            if list(b[t].columns) != cols[t]:
                errors.append(f"{d.name}/{FILES[t]}: columns differ from original")
            keyframes[t].append(b[t][PK[t]])

        # simulate the load order: customers, products, orders, order_items
        cum["customers"].update(b["customers"]["customer_id"])
        cum["products"].update(b["products"]["product_id"])
        bad = ~b["orders"]["customer_id"].isin(cum["customers"])
        if bad.any():
            errors.append(f"{d.name}: {int(bad.sum())} orders.customer_id not in customers "
                          f"of batches 1..{k} (e.g. {b['orders'].loc[bad, 'customer_id'].iloc[0]})")
        cum["orders"].update(b["orders"]["order_id"])
        bad = ~b["items"]["order_id"].isin(cum["orders"])
        if bad.any():
            errors.append(f"{d.name}: {int(bad.sum())} order_items.order_id not in orders "
                          f"of batches 1..{k} (e.g. {b['items'].loc[bad, 'order_id'].iloc[0]})")
        bad = ~b["items"]["product_id"].isin(cum["products"])
        if bad.any():
            errors.append(f"{d.name}: {int(bad.sum())} order_items.product_id not in products "
                          f"of batches 1..{k} (e.g. {b['items'].loc[bad, 'product_id'].iloc[0]})")

        counts.append({"batch": k, **{t: len(b[t]) for t in BATCH_TABLES}})

    # duplicate primary keys across batches
    for t in BATCH_TABLES:
        allkeys = pd.concat(keyframes[t], ignore_index=True) if keyframes[t] else pd.DataFrame(columns=PK[t])
        dups = int(allkeys.duplicated().sum())
        if dups:
            note = f" ({source_dups[t]} already duplicated in the input)" if source_dups.get(t) else ""
            errors.append(f"{t}: {dups} duplicate primary key(s) {PK[t]} across batches{note}")

    # row counts
    cdf = pd.DataFrame(counts)
    print(f"  Rows per batch ({n} batches):")
    show = cdf if n <= 12 else pd.concat([cdf.head(3), cdf.tail(3)])
    lines = show.to_string(index=False).splitlines()
    if n > 12:
        lines.insert(4, "      ...")
    print("\n".join("    " + ln for ln in lines))
    print("\n  Sum of batches vs original:")
    for t in BATCH_TABLES:
        total, orig = int(cdf[t].sum()), len(data[t])
        ok = total == orig
        print(f"    {t:<10} {total:>9,} / {orig:>9,}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            errors.append(f"{t}: batches sum to {total}, original has {orig}")

    print()
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  - {e}")
        return False
    print("VALIDATION PASSED: cumulative FK checks OK, no duplicate keys, row counts match.")
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=".", help="folder with the 5 original CSVs (default: .)")
    ap.add_argument("--output-dir", default="output", help="where batch_k folders are written (default: ./output)")
    ap.add_argument("-n", "--batches", type=int, default=3, help="number of chronological batches (default 3)")
    ap.add_argument("--by-date", action="store_true", help="one batch per calendar day (ignores -n)")
    ap.add_argument("--validate-only", action="store_true", help="only validate an existing output dir")
    a = ap.parse_args()
    if a.batches < 1:
        print("ERROR: -n must be >= 1")
        sys.exit(2)

    in_dir, out = Path(a.input_dir), Path(a.output_dir)
    check_input_files(in_dir)
    data = {t: read(in_dir / FILES[t]) for t in FILES}
    for t, n_rows in ((t, len(df)) for t, df in data.items()):
        print(f"  loaded {FILES[t]}: {n_rows:,} rows")
    source_dups = check_duplicate_keys(data)
    check_source_orphans(data)
    report_missing_categories(data["products"], data["translation"])

    if a.validate_only:
        n = len(list(out.glob("batch_*")))
        if n == 0:
            print(f"ERROR: no batch_* folders in {out}")
            sys.exit(2)
    else:
        print("\n== Splitting ==")
        n = split(data, a.batches, a.by_date, out)
        print(f"  wrote {n} batch folder(s) to {out}")

    sys.exit(0 if validate(data, out, n, source_dups) else 1)


if __name__ == "__main__":
    main()
