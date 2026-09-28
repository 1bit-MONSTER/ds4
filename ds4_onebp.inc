/*
 * Copyright 2026 bong-water-water-bong
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 1BP v5 reader (1bit-MONSTER's model package; docs/1BP.md). Included by ds4.c
 * after parse_metadata() / parse_tensors(), whose helpers it reuses.
 *
 *   [header: 256 bytes]   magic "1BP\0", version 5, tensor_count at byte 88
 *   [tensor index]        per tensor: name_len u32, name, NUL, ndim u32,
 *                         dims u32 x ndim (outermost first), offset u64,
 *                         bytes u64, quant u32
 *   [metadata]            n_kv u64, then n_kv GGUF key/value entries, byte for
 *                         byte as in GGUF, so parse_metadata() reads them as is
 *   [data]                at the next multiple of 64; each tensor at a
 *                         multiple of 64 from there (offset is relative)
 *
 * quant >= 256 is a GGUF block format carried unchanged (quant - 256 is the
 * ggml type), which ds4's kernels already run. quant < 256 are 1BP's own tile
 * formats (Q4NX, TQ2, ...), which this reader does not run yet. An entry with
 * bytes == 0 aliases an earlier tensor: offset is that tensor's index.
 */

#define DS4_ONEBP_MAGIC        0x00504231u /* "1BP\0", little endian */
#define DS4_ONEBP_VERSION      5u
#define DS4_ONEBP_HEADER_BYTES 256u
#define DS4_ONEBP_ALIGN        64u
#define DS4_ONEBP_GGML_BASE    256u

static void model_open_onebp(ds4_model *m, ds4_cursor *c) {
    uint32_t version = 0, tensor_count = 0;
    ds4_cursor h = cursor_at(m, 4);
    if (!cursor_u32(&h, &version)) ds4_die(h.error);
    if (version != DS4_ONEBP_VERSION) {
        ds4_die("this 1BP file is not version 5; 1BP v1-v4 hold 1bit-MONSTER's own tile "
                "formats and no metadata block, which ds4 does not read");
    }
    h = cursor_at(m, 88);
    if (!cursor_u32(&h, &tensor_count)) ds4_die(h.error);

    m->version = 3;  /* ds4 prints it; the metadata below is GGUF v3's encoding */
    m->n_tensors = tensor_count;
    c->pos = DS4_ONEBP_HEADER_BYTES;
    if (m->n_tensors > c->size - c->pos) ds4_die("1BP tensor count exceeds file size");
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        uint32_t name_len = 0, ndim = 0, quant = 0;
        if (!cursor_u32(c, &name_len)) ds4_die(c->error);
        if (!cursor_has(c, (uint64_t)name_len + 1)) ds4_die(c->error);
        t->name.ptr = (const char *)(c->base + c->pos);
        t->name.len = name_len;
        c->pos += (uint64_t)name_len + 1;  /* the name, then its NUL */
        if (!cursor_u32(c, &ndim)) ds4_die(c->error);
        if (ndim == 0 || ndim > DS4_MAX_DIMS) ds4_die("1BP tensor has an unsupported number of dimensions");
        t->ndim = ndim;
        t->elements = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim = 0;
            if (!cursor_u32(c, &dim)) ds4_die(c->error);
            t->dim[ndim - 1 - d] = dim;  /* 1BP lists outermost first, GGUF innermost first */
            if (dim != 0 && t->elements > UINT64_MAX / dim) ds4_die("tensor element count overflow");
            t->elements *= dim;
        }
        uint64_t bytes = 0;
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);
        if (!cursor_u64(c, &bytes)) ds4_die(c->error);
        if (!cursor_u32(c, &quant)) ds4_die(c->error);
        if (bytes == 0) {
            /* alias: rel_offset names an earlier tensor, which shares its bytes */
            if (t->rel_offset >= i) ds4_die("1BP alias does not point to an earlier tensor");
            const ds4_tensor *src = &m->tensors[t->rel_offset];
            t->type = src->type;
            t->rel_offset = src->rel_offset;
            t->bytes = src->bytes;
            continue;
        }
        if (quant < DS4_ONEBP_GGML_BASE) {
            ds4_log(stderr, DS4_LOG_WARNING,
                    "ds4: 1BP tensor %.*s uses 1BP tile format %u, which ds4 does not run yet\n",
                    (int)t->name.len, t->name.ptr, quant);
            ds4_die("unsupported 1BP tile format");
        }
        t->type = quant - DS4_ONEBP_GGML_BASE;
        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr, DS4_LOG_WARNING, "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                    (int)t->name.len, t->name.ptr, t->type);
            t->bytes = bytes;
        } else if (t->bytes != bytes) {
            ds4_die("1BP tensor size does not match its type and shape");
        }
        if (t->rel_offset % DS4_ONEBP_ALIGN != 0) ds4_die("1BP tensor is not 64-byte aligned");
    }

    if (!cursor_u64(c, &m->n_kv)) ds4_die(c->error);
    parse_metadata(m, c);
    m->alignment = DS4_ONEBP_ALIGN;
    m->tensor_data_pos = align_up(c->pos, DS4_ONEBP_ALIGN);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) ds4_die("tensor offset overflow");
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 && (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset)) {
            ds4_die("tensor points outside 1BP file");
        }
        if (t->bytes > m->max_tensor_bytes) m->max_tensor_bytes = t->bytes;
    }
}
