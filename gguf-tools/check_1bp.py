#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check a 1BP v5 package against the GGUF it came from: the metadata block and every
tensor (name, shape, type, bytes) must be identical, and every tensor 64-byte aligned.

    check_1bp.py model.gguf model.1bp
"""
import hashlib
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_to_1bp import LocalSource, read_gguf, ONEBP_MAGIC, ONEBP_ALIGN, ONEBP_GGML_BASE  # noqa: E402


def read_1bp(path):
    f = open(path, "rb")
    hdr = f.read(256)
    magic, version = struct.unpack_from("<II", hdr, 0)
    (count,) = struct.unpack_from("<I", hdr, 88)
    if magic != ONEBP_MAGIC or version != 5:
        raise ValueError(f"not 1BP v5 (magic {magic:#x}, version {version})")
    f.seek(256)
    buf = f.read(64 << 20)
    pos, tensors = 0, []
    for _ in range(count):
        (nl,) = struct.unpack_from("<I", buf, pos); pos += 4
        name = buf[pos:pos + nl].decode(); pos += nl + 1
        (nd,) = struct.unpack_from("<I", buf, pos); pos += 4
        dims = list(struct.unpack_from(f"<{nd}I", buf, pos)); pos += 4 * nd
        off, nbytes, quant = struct.unpack_from("<QQI", buf, pos); pos += 20
        tensors.append({"name": name, "dims": dims, "offset": off, "bytes": nbytes, "quant": quant})
    (n_kv,) = struct.unpack_from("<Q", buf, pos); pos += 8
    return f, n_kv, 256 + pos, tensors


def main():
    gguf_path, onebp_path = sys.argv[1], sys.argv[2]
    src = LocalSource(gguf_path)
    n_kv, kv, kv_start, kv_end, gt = read_gguf(src)
    f, n_kv_1bp, meta_pos, bt = read_1bp(onebp_path)
    ok = True
    if n_kv_1bp != n_kv:
        print(f"metadata count differs: {n_kv} vs {n_kv_1bp}"); ok = False
    f.seek(meta_pos)
    meta = f.read(kv_end - kv_start)
    if meta != src.read_at(kv_start, kv_end - kv_start):
        print("metadata block differs"); ok = False
    data_start = -(-(meta_pos + len(meta)) // ONEBP_ALIGN) * ONEBP_ALIGN
    if len(gt) != len(bt):
        print(f"tensor count differs: {len(gt)} vs {len(bt)}"); ok = False
    for g, b in zip(gt, bt):
        if (g["name"], list(reversed(g["dims"])), ONEBP_GGML_BASE + g["type"], g["bytes"]) != \
           (b["name"], b["dims"], b["quant"], b["bytes"]):
            print(f"entry differs: {g['name']}"); ok = False; continue
        if b["offset"] % ONEBP_ALIGN:
            print(f"{b['name']} is not 64-byte aligned"); ok = False
        h1, h2 = hashlib.sha256(), hashlib.sha256()
        pos, left = g["abs"], g["bytes"]
        f.seek(data_start + b["offset"])
        while left:
            n = min(left, 64 << 20)
            h1.update(src.read_at(pos, n)); h2.update(f.read(n))
            pos += n; left -= n
        if h1.digest() != h2.digest():
            print(f"{g['name']}: bytes differ"); ok = False
    print(f"{'OK' if ok else 'FAIL'}: {len(bt)} tensors, {n_kv} metadata keys")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
