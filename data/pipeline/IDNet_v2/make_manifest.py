"""Build the IDNet sampling manifest (docs/idnet-integration-plan.md stage 1).

Per country: 500 genuine + 500 fraud split as evenly as possible over the 6
fraud types (84/84/83/83/83/83; amendment 2026-09-15, EVALUATION.md §2.3),
all with disjoint document keys (a document's key is its filename stem; a
positive and its fraud derivatives share it, so key-disjoint sampling means
every sampled document contributes exactly one image — the leakage grain by
construction).

Member lists come from the remote zips' central directories via ranged reads
— nothing is downloaded here. ALL countries are sourced from the Sept-2024
Zenodo release: it is a regenerated sample (different persons than the
Jan-2024 release in our S3 mirror), so releases cannot be mixed — and only
Sept has all 6 fraud types. Deterministic for a given --seed.

Naming: positive/fraud1-4 files share the document stem; fraud5 appends
`_fake_<n>` and fraud6 `_fake_<k>_<n>` (multiple variants per document —
the lexicographically first is taken; a handful of corrupt names that match
no positive are excluded). Document key = stem with the suffix stripped.
"""
import argparse
import csv
import hashlib
import io
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import RemoteFile

# Sept-2024 Zenodo records (regenerated release, 6 fraud types per country)
ZENODO_REC = {"ALB": 13854897, "EST": 13854897, "CA": 13854897,
              "AZE": 13852263, "SRB": 13852263,
              "SVK": 13852734, "LVA": 13852734, "FIN": 13852734,
              "ESP": 13852734, "SD": 13852734,
              "AZ": 13852757, "WI": 13852757,
              "PA": 13855052, "NC": 13855175, "NV": 13855175,
              "GRC": 13854938, "RUS": 13854938, "WV": 13854938,
              "DC": 13852274, "UT": 13852274}
COUNTRIES = sorted(c.lower() for c in ZENODO_REC)

FRAUD_TYPES = ["fraud1_copy_and_move", "fraud2_face_morphing",
               "fraud3_face_replacement", "fraud4_combined",
               "fraud5_inpaint_and_rewrite", "fraud6_crop_and_replace"]
N_POS = 5979
FAKE_SUFFIX = re.compile(r"_fake_\d+(_\d+)?$")


def fraud_quotas(n_fraud_total, fraud_types):
    """Closest-to-equal integer partition, remainder to the earliest types."""
    base, rem = divmod(n_fraud_total, len(fraud_types))
    return [base + (1 if i < rem else 0) for i in range(len(fraud_types))]


def sample_country(country, n_genuine, n_fraud_total, fraud_types, seed):
    top = country.upper()
    origin = f"zenodo:{ZENODO_REC[top]}/{top}.zip"
    zf = zipfile.ZipFile(
        io.BufferedReader(RemoteFile(origin), buffer_size=256 * 1024))
    raw = {}  # folder -> [member paths]
    for n in zf.namelist():
        parts = n.split("/")
        if len(parts) == 3 and parts[2] and parts[0] == top:
            raw.setdefault(parts[1], []).append(n)

    pos_keys = {Path(n).stem for n in raw["positive"]}
    assert len(pos_keys) == N_POS, \
        f"{country}: {len(pos_keys)} positives, expected {N_POS}"
    members = {}  # folder -> {doc key: member path}
    for folder in ["positive"] + fraud_types:
        assert folder in raw, f"{country}: missing folder {folder}"
        m = {}
        for n in sorted(raw[folder]):  # sorted: variant pick is deterministic
            key = FAKE_SUFFIX.sub("", Path(n).stem)
            if key not in pos_keys:  # a few corrupt fraud6 names — exclude
                continue
            m.setdefault(key, n)
        members[folder] = m
        assert len(m) >= 2 * n_fraud_total, \
            f"{country}/{folder}: only {len(m)} usable keys"

    pool = sorted(pos_keys)
    random.Random(f"{seed}:{country}").shuffle(pool)
    quotas = [("positive", n_genuine, "0", "")] + [
        (f, q, "1", f)
        for f, q in zip(fraud_types, fraud_quotas(n_fraud_total, fraud_types))
    ]
    rows, i = [], 0
    for folder, want, label, fraud_type in quotas:
        got = 0
        while got < want:
            assert i < len(pool), f"{country}: pool exhausted filling {folder}"
            key = pool[i]
            i += 1
            if key not in members[folder]:  # tolerate e.g. fraud6's 5,978
                continue
            rows.append({
                "id": f"idnet_{country}/{folder}/{key}",
                "country": country,
                "type": f"idnet_{country}",
                "folder": folder,
                "member": members[folder][key],
                "label": label,
                "fraud_type": fraud_type,
                "origin": origin,
                "md5": "",
            })
            got += 1
    keys = [r["id"].rsplit("/", 1)[1] for r in rows]
    assert len(keys) == len(set(keys)), f"{country}: document key reused"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-genuine", type=int, default=500)
    ap.add_argument("--n-fraud-total", type=int, default=500)
    ap.add_argument("--fraud-types", nargs="+", default=FRAUD_TYPES)
    ap.add_argument("--out", default="sources/IDNet_v2/manifest.csv")
    args = ap.parse_args()

    rows = []
    for country in COUNTRIES:
        rows += sample_country(country, args.n_genuine, args.n_fraud_total,
                               args.fraud_types, args.seed)
        print(f"{country}: sampled {args.n_genuine}+{args.n_fraud_total} fraud "
              f"over {len(args.fraud_types)} types", flush=True)
    rows.sort(key=lambda r: r["id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['id']},{r['label']},{r['type']}\n".encode())
    print(f"wrote {out}  ({len(rows)} rows)")
    print(f"row-hash (id,label,type): {h.hexdigest()}")


if __name__ == "__main__":
    main()
