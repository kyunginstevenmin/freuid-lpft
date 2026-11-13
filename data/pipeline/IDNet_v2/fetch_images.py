"""Materialize the IDNet manifest (docs/idnet-integration-plan.md stage 2).

Downloads every manifest row into sources/IDNet_v2/Images/<country>/<folder>/,
then rewrites the manifest with each file's md5. Idempotent: existing files
with a matching manifest md5 are skipped; corrupt/partial files are refetched.

Every file is range-extracted from its origin zip (Sept-2024 Zenodo records
for all countries) without downloading the archive.
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import struct
import sys
import threading
import zipfile
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import RemoteFile

IMG_ROOT = Path("sources/IDNet_v2/Images")

_local = threading.local()


def dest(row):
    return IMG_ROOT / row.country / row.folder / (row.member.rsplit("/", 1)[1])


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# origin -> {member: ZipInfo}, built once under a lock, shared by all workers
_infos = {}
_infos_lock = threading.Lock()


def zip_infos(origin):
    with _infos_lock:
        if origin not in _infos:
            zf = zipfile.ZipFile(
                io.BufferedReader(RemoteFile(origin), buffer_size=256 * 1024)
            )
            _infos[origin] = {zi.filename: zi for zi in zf.infolist()}
        return _infos[origin]


def fetch_zip_member(row):
    """Read one member in a single ranged request: local header + data."""
    zi = zip_infos(row.origin)[row.member]
    if not hasattr(_local, "files"):
        _local.files = {}
    if row.origin not in _local.files:
        _local.files[row.origin] = RemoteFile(row.origin)
    rf = _local.files[row.origin]
    # local header = 30 bytes + name + extra (extra may differ from the
    # central directory's) — overshoot generously, one request total
    head = 30 + len(zi.filename) + 512
    raw = rf.pread(zi.header_offset, head + zi.compress_size)
    assert raw[:4] == b"PK\x03\x04", f"bad local header for {row.member}"
    n, m = struct.unpack("<HH", raw[26:30])
    data = raw[30 + n + m : 30 + n + m + zi.compress_size]
    if len(data) < zi.compress_size:  # overshoot was not enough
        data = rf.pread(zi.header_offset + 30 + n + m, zi.compress_size)
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    else:
        assert zi.compress_type == zipfile.ZIP_STORED
    assert len(data) == zi.file_size and zlib.crc32(data) == zi.CRC, \
        f"corrupt extract for {row.member}"
    return data


def fetch_one(row):
    d = dest(row)
    if d.exists():
        # An existing file is complete (tmp+rename), but may come from the
        # superseded Jan-2024 release under the same name — trust it only if
        # its CRC matches the origin zip's central directory.
        data = d.read_bytes()
        zi = zip_infos(row.origin)[row.member]
        if zlib.crc32(data) == zi.CRC and len(data) == zi.file_size:
            return row.id, hashlib.md5(data).hexdigest(), "skip"
    d.parent.mkdir(parents=True, exist_ok=True)
    body = fetch_zip_member(row)
    tmp = d.with_suffix(".tmp")
    tmp.write_bytes(body)
    tmp.rename(d)
    return row.id, hashlib.md5(body).hexdigest(), "fetched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="sources/IDNet_v2/manifest.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype=str).fillna("")
    if args.dry_run:
        todo = sum(1 for r in df.itertuples() if not dest(r).exists())
        print(f"{len(df)} rows: {todo} to fetch, {len(df) - todo} already on disk")
        return

    md5s, n_done = {}, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, r) for r in df.itertuples()]
        for fut in cf.as_completed(futs):
            rid, m, how = fut.result()
            md5s[rid] = m
            n_done += 1
            if n_done % 500 == 0:
                print(f"{n_done}/{len(df)}", flush=True)

    df["md5"] = df["id"].map(md5s)
    assert df["md5"].notna().all() and (df["md5"] != "").all()
    df.to_csv(args.manifest, index=False)
    print(f"done: {len(df)} files under {IMG_ROOT}; md5s written to {args.manifest}")


if __name__ == "__main__":
    main()
