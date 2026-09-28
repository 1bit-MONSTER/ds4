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
"""Convert a GGUF that ds4 runs into a 1BP v5 package (docs/1BP.md).

The GGUF's metadata is copied byte for byte into the package's metadata block
and every tensor is carried in its GGUF block format (1BP quant 256 + ggml type),
64-byte aligned, so ds4 runs the package exactly as it runs the GGUF.

    gguf_to_1bp.py model.gguf model.1bp
    gguf_to_1bp.py hf://antirez/deepseek-v4-gguf@f71f23d5/<file>.gguf model.1bp

An hf:// source is read with HTTP range requests, so the GGUF never has to be on disk.
Needs gguf-py (for the block sizes) and, for hf://, huggingface_hub.
"""
import argparse
import struct
import sys
import time

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

ONEBP_MAGIC = 0x00504231
ONEBP_VERSION = 5
ONEBP_ALIGN = 64
ONEBP_GGML_BASE = 256
ONEBP_DEEPSEEK_V4 = 22
GGUF_MAGIC = 0x46554747

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f",
          BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


class LocalSource:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.f.seek(0, 2)
        self.size = self.f.tell()

    def read_at(self, off, n):
        self.f.seek(off)
        return self.f.read(n)


class HfSource:
    def __init__(self, url):
        from huggingface_hub import HfFileSystem
        owner, repo, name = url[len("hf://"):].split("/", 2)  # hf://owner/repo[@rev]/file
        self.f = HfFileSystem().open(f"{owner}/{repo}/{name}", "rb", block_size=1 << 20, cache_type="none")
        self.size = self.f.size

    def read_at(self, off, n):
        self.f.seek(off)
        out = bytearray()
        while len(out) < n:
            chunk = self.f.read(n - len(out))
            if not chunk:
                raise IOError(f"short read at {off + len(out)}")
            out += chunk
        return bytes(out)


class Reader:
    """Sequential reads over a source, buffered in 8 MiB windows."""

    def __init__(self, src, pos=0):
        self.src, self.pos, self.buf, self.buf_pos = src, pos, b"", pos

    def take(self, n):
        end = self.pos + n
        if not (self.buf_pos <= self.pos and end <= self.buf_pos + len(self.buf)):
            self.buf_pos = self.pos
            self.buf = self.src.read_at(self.pos, min(max(n, 8 << 20), self.src.size - self.pos))
        out = self.buf[self.pos - self.buf_pos:end - self.buf_pos]
        self.pos = end
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t in SCALAR:
            fmt = SCALAR[t]
            return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]
        if t == STR:
            return self.string()
        if t == ARR:
            # ("array", element type, length, values): short scalar arrays decoded, others kept raw
            et, n = self.u32(), self.u64()
            if et in SCALAR:
                raw = self.take(struct.calcsize(SCALAR[et]) * n)
                return ("array", et, n, list(struct.unpack("<" + SCALAR[et][1] * n, raw)) if n <= 64 else raw)
            return ("array", et, n, [self.value(et) for _ in range(n)])
        raise ValueError(f"unknown GGUF value type {t}")


def read_gguf(src):
    r = Reader(src)
    if r.u32() != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
    if version != 3:
        raise ValueError(f"GGUF v{version}; only v3")
    kv_start = r.pos
    kv = {}
    for _ in range(n_kv):
        key, t = r.string(), r.u32()
        kv[key] = r.value(t)
    kv_end = r.pos
    tensors = []
    for _ in range(n_tensors):
        name, ndim = r.string(), r.u32()
        dims = [r.u64() for _ in range(ndim)]
        ttype, off = r.u32(), r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = kv.get("general.alignment", 32)
    data_start = -(-r.pos // align) * align
    for t in tensors:
        n = 1
        for d in t["dims"]:
            n *= d
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType(t["type"])]
        if n % block:
            raise ValueError(f"{t['name']}: {n} elements are not a whole number of {block}-element blocks")
        t["bytes"] = n // block * size
        t["abs"] = data_start + t["offset"]
        if t["abs"] + t["bytes"] > src.size:
            raise ValueError(f"{t['name']} points outside the GGUF")
    return n_kv, kv, kv_start, kv_end, tensors


def header(kv, tensors):
    arch = kv.get("general.architecture", "")
    g = lambda k, d=0: kv.get(f"{arch}.{k}", d)  # noqa: E731
    first = lambda v: (v[3][0] if isinstance(v, tuple) and isinstance(v[3], list) and v[3] else v)  # noqa: E731
    tok = kv.get("tokenizer.ggml.tokens")
    vocab = tok[2] if isinstance(tok, tuple) else 0
    by_type = {}
    for t in tensors:
        by_type[t["type"]] = by_type.get(t["type"], 0) + t["bytes"]
    dominant = max(by_type, key=by_type.get)
    theta = float(g("rope.freq_base", 0.0) or 0.0)
    swa_theta = float(g("rope.freq_base_swa", 0.0) or 0.0)
    heads = int(first(g("attention.head_count", 0)) or 0)
    fields = [
        ONEBP_MAGIC, ONEBP_VERSION, ONEBP_DEEPSEEK_V4 if arch.startswith("deepseek4") else 0,
        ONEBP_GGML_BASE + dominant, 0,
        int(g("embedding_length", 1)) or 1, int(g("block_count", 1)) or 1, heads,
        int(first(g("attention.head_count_kv", 0)) or 0),
        int(g("attention.key_length", 0) or (1 if heads else 0)),
        int(first(g("feed_forward_length", 0)) or g("expert_feed_forward_length", 1)) or 1,
        vocab or 1, int(g("context_length", 0)),
        32, 256, 32, 0, 0, 0,
        struct.unpack("<I", struct.pack("<f", theta))[0],
        int(kv.get("tokenizer.ggml.bos_token_id", 0)), int(kv.get("tokenizer.ggml.eos_token_id", 0)),
        len(tensors),
    ]
    hdr = struct.pack("<5I8i10I", *fields)
    hdr += struct.pack("<14I", int(g("expert_count", 0)), int(g("expert_used_count", 0)),
                       int(g("expert_feed_forward_length", 0)), int(g("expert_shared_feed_forward_length", 0)),
                       int(g("leading_dense_block_count", 0)), int(g("attention.sliding_window", 0)), 0, 0, 0,
                       int(round(float(g("expert_weights_scale", 0.0) or 0.0) * 1000)), 0,
                       struct.unpack("<I", struct.pack("<f", swa_theta))[0], 0, 0)
    hdr += bytes(32 + 12)  # MLA fields, reserved
    tag = str(kv.get("general.name", arch)).encode()[:63]
    hdr += tag + bytes(64 - len(tag))
    assert len(hdr) == 256
    return hdr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="a .gguf path or hf://<repo>[@rev]/<file>.gguf")
    ap.add_argument("dst", help="the .1bp to write")
    a = ap.parse_args()
    src = HfSource(a.src) if a.src.startswith("hf://") else LocalSource(a.src)
    n_kv, kv, kv_start, kv_end, tensors = read_gguf(src)

    index = bytearray()
    off = 0
    for t in tensors:
        name = t["name"].encode()
        dims = list(reversed(t["dims"]))  # 1BP lists outermost first
        if any(d >= 1 << 32 for d in dims):
            raise ValueError(f"{t['name']}: a dimension does not fit 1BP's u32")
        off = -(-off // ONEBP_ALIGN) * ONEBP_ALIGN
        t["dst_off"] = off
        index += struct.pack("<I", len(name)) + name + b"\0" + struct.pack("<I", len(dims))
        index += struct.pack(f"<{len(dims)}I", *dims) + struct.pack("<QQI", off, t["bytes"], ONEBP_GGML_BASE + t["type"])
        off += t["bytes"]
    meta = struct.pack("<Q", n_kv) + src.read_at(kv_start, kv_end - kv_start)
    head = header(kv, tensors) + bytes(index) + meta
    data_start = -(-len(head) // ONEBP_ALIGN) * ONEBP_ALIGN

    total = sum(t["bytes"] for t in tensors)
    done, t0 = 0, time.time()
    with open(a.dst, "wb") as out:
        out.write(head + bytes(data_start - len(head)))
        for t in sorted(tensors, key=lambda t: t["dst_off"]):
            pad = data_start + t["dst_off"] - out.tell()
            if pad < 0:
                raise RuntimeError("tensor layout overlaps")
            out.write(bytes(pad))
            pos, left = t["abs"], t["bytes"]
            while left:
                n = min(left, 64 << 20)
                out.write(src.read_at(pos, n))
                pos += n
                left -= n
                done += n
            if sys.stderr.isatty() or len(tensors) < 50 or t is tensors[-1] or done * 20 // total != (done - t["bytes"]) * 20 // total:
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"{done / 1e9:7.1f} / {total / 1e9:.1f} GB  {rate:6.0f} MB/s  {t['name']}", file=sys.stderr, flush=True)
    print(f"wrote {a.dst}: 1BP v5, {len(tensors)} tensors, {n_kv} metadata keys, {total / 2**30:.2f} GiB of weights")


if __name__ == "__main__":
    main()
