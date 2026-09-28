<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 1BP in DwarfStar

This fork (`1bit-MONSTER/ds4`) runs models from 1BP packages as well as GGUF. 1BP is
1bit-MONSTER's model package: one memory-mappable file with a fixed header, a tensor index,
the model's metadata and the weights.

## Version 5

| Part | Contents |
|---|---|
| header, 256 bytes | magic `1BP\0` (`0x00504231`), version 5, architecture, dominant quant, model dimensions, RoPE bases, BOS/EOS, `tensor_count` at byte 88, a 64-byte model tag |
| tensor index | for each tensor: name length (u32), name, NUL, `ndim` (u32), dimensions (u32 each, outermost first), offset (u64, from the data start), bytes (u64), quant (u32) |
| metadata | key count (u64), then the key/value entries encoded exactly as in GGUF v3 |
| data | at the next multiple of 64 bytes; every tensor at a multiple of 64 from there |

Quant values below 256 are 1BP's own tile formats (Q4NX, TQ2, TQ2NZ and the others of the 1BP
header in 1bit-MONSTER). A quant of 256 or more carries a GGUF block format unchanged: the
ggml type is `quant - 256`. An index entry with `bytes == 0` aliases an earlier tensor, whose
index is in its offset field.

Version 5 adds the metadata block and the 64-byte alignment. Versions 1-4 had neither: they
carried only 1BP's tile formats, and their models took the tokenizer from elsewhere. This fork
reads version 5 only.

## Converting

```
python gguf-tools/gguf_to_1bp.py model.gguf model.1bp
python gguf-tools/gguf_to_1bp.py hf://antirez/deepseek-v4-gguf@<rev>/<file>.gguf model.1bp
python gguf-tools/check_1bp.py model.gguf model.1bp
```

The converter copies the GGUF's metadata byte for byte and carries every tensor in its GGUF
format, so a converted package runs exactly as its GGUF. An `hf://` source is read with HTTP
range requests, and the GGUF never has to be on disk. `check_1bp.py` compares a package with
its GGUF: the metadata block, and every tensor's name, shape, type, bytes and alignment. It
needs gguf-py (`pip install gguf`).

`ds4`, `ds4-server` and the other tools take a `.1bp` wherever they take a GGUF: `model_open()`
reads the magic and hands a 1BP file to `model_open_onebp()` (`ds4_onebp.inc`), which builds
the same tensor table and metadata table the GGUF reader builds.

## Not yet

1BP's own tile formats (quant < 256) in ds4's kernels. A package with one is refused with the
tensor's name.
