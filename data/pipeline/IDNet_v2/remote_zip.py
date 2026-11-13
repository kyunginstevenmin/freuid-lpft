"""Seekable read-only file over a remote object (S3 or HTTPS with Range
support), good enough for zipfile: list a remote zip's members and extract a
few of them via ranged reads, without downloading the archive.

Origin strings:
    s3://bucket/key
    zenodo:<record_id>/<file_key>   (resolved to a zenodo.org content URL)
    https://...
"""
import io
import urllib.parse

import boto3
import requests


class RemoteFile(io.RawIOBase):
    def __init__(self, origin):
        self.origin = origin
        if origin.startswith("s3://"):
            u = urllib.parse.urlparse(origin)
            self._s3 = boto3.client("s3")
            self._bucket, self._key = u.netloc, u.path.lstrip("/")
            self._size = self._s3.head_object(Bucket=self._bucket, Key=self._key)[
                "ContentLength"
            ]
            self._url = None
        else:
            if origin.startswith("zenodo:"):
                rec, fkey = origin[len("zenodo:"):].split("/", 1)
                self._url = f"https://zenodo.org/api/records/{rec}/files/{fkey}/content"
            else:
                self._url = origin
            self._session = requests.Session()
            import time
            for attempt in range(6):
                r = self._session.head(self._url, allow_redirects=True, timeout=60)
                if r.status_code == 429:
                    time.sleep(min(2 ** attempt * 10, 300))
                    continue
                break
            r.raise_for_status()
            self._url = r.url  # pin the redirect target
            self._size = int(r.headers["Content-Length"])
        self._pos = 0

    # -- io plumbing -------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self._pos = {
            io.SEEK_SET: offset,
            io.SEEK_CUR: self._pos + offset,
            io.SEEK_END: self._size + offset,
        }[whence]
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n < 0 or self._pos + n > self._size:
            n = self._size - self._pos
        if n == 0:
            return b""
        lo, hi = self._pos, self._pos + n - 1
        if self._url is None:
            body = self._s3.get_object(
                Bucket=self._bucket, Key=self._key, Range=f"bytes={lo}-{hi}"
            )["Body"].read()
        else:
            import time
            for attempt in range(6):
                r = self._session.get(
                    self._url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120
                )
                if r.status_code in (429, 500, 502, 503):  # throttled: back off
                    time.sleep(min(2 ** attempt * 5, 120))
                    continue
                r.raise_for_status()
                break
            r.raise_for_status()
            assert r.status_code == 206, f"no Range support from {self._url}"
            body = r.content
        assert len(body) == n, f"short read {len(body)} != {n}"
        self._pos += n
        return body

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def pread(self, offset, length):
        """One ranged request at an absolute offset (ignores the cursor)."""
        pos = self._pos
        try:
            self._pos = offset
            return self.read(length)
        finally:
            self._pos = pos
