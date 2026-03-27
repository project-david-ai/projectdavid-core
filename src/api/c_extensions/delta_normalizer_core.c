/*
 * delta_normalizer_core.c
 *
 * CPython extension — replaces the inner `while buffer:` state-machine loop
 * in DeltaNormalizer.async_iter_deltas with a single C function call.
 *
 * Interface (called once per streaming token after appending seg to buffer):
 *
 *   process_buffer(
 *       buf             : str,   # current accumulated buffer
 *       state           : int,   # current state (STATE_* constants below)
 *       json_depth      : int,   # brace depth for naked-json state
 *       has_emitted_text: bool,
 *       xml_buf         : str,   # accumulator for fc/tool_call_xml content
 *       run_id          : str,
 *   ) -> (
 *       events          : list[dict],   # [{type, content, run_id}, ...]
 *       new_buf         : str,
 *       new_state       : int,
 *       new_json_depth  : int,
 *       new_has_emitted_text: bool,
 *       new_xml_buf     : str,
 *   )
 *
 * All state constants are exposed as module-level ints so Python can read them.
 * Build with:  python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

/* ── Portability ──────────────────────────────────────────────────────────── */
/* memmem is a GNU extension unavailable on Windows/MSVC — provide a fallback */
#ifndef _GNU_SOURCE
static void *
portable_memmem(const void *haystack, size_t hlen,
                const void *needle,   size_t nlen)
{
    const char *h = (const char *)haystack;
    const char *n = (const char *)needle;
    if (nlen == 0) return (void *)h;
    if (hlen < nlen) return NULL;
    for (size_t i = 0; i <= hlen - nlen; i++) {
        if (memcmp(h + i, n, nlen) == 0)
            return (void *)(h + i);
    }
    return NULL;
}
#define memmem portable_memmem
#endif

/* ── State IDs ────────────────────────────────────────────────────────────── */
#define ST_CONTENT               0
#define ST_THINK                 1
#define ST_PLAN                  2
#define ST_DECISION              3
#define ST_FC                    4
#define ST_TOOL_CALL_XML         5
#define ST_TOOL_CODE_XML         6
#define ST_MD_JSON_BLOCK         7
#define ST_NAKED_JSON            8
#define ST_KIMI_ROUTER           9
#define ST_KIMI_ARGS            10
#define ST_UNICODE_TOOL_ROUTER  11
#define ST_UNICODE_TOOL_PARSING 12
#define ST_UNICODE_TOOL_ARGS    13
#define ST_CHANNEL_REASONING    14
#define ST_CHANNEL_TOOL_META    15
#define ST_CHANNEL_TOOL_PAYLOAD 16

/* ── Tags (UTF-8 literals) ────────────────────────────────────────────────── */
static const char *TAG_FC_START       = "<fc>";
static const char *TAG_FC_END         = "</fc>";
static const char *TAG_TC_START       = "<tool_call>";
static const char *TAG_TC_END         = "</tool_call>";
static const char *TAG_TCODE_START    = "<tool_code>";
static const char *TAG_TCODE_END      = "</tool_code>";
static const char *TAG_MD_JSON        = "```json";
static const char *TAG_MD_END         = "```";
static const char *TAG_NAKED_JSON     = "{";
static const char *TAG_TH_START       = "<think>";
static const char *TAG_TH_END         = "</think>";
static const char *TAG_DEC_START      = "<decision>";
static const char *TAG_DEC_END        = "</decision>";
static const char *TAG_PLAN_START     = "<plan>";
static const char *TAG_PLAN_END       = "</plan>";
static const char *TAG_CH_ANALYSIS    = "<|channel|>analysis";
static const char *TAG_CH_COMMENTARY  = "<|channel|>commentary";
static const char *TAG_CH_FINAL       = "<|channel|>final";
static const char *TAG_MSG            = "<|message|>";
static const char *TAG_CALL           = "<|call|>";
static const char *TAG_KIMI_SEC_START = "<|tool_calls_section_begin|>";
static const char *TAG_KIMI_SEC_END   = "<|tool_calls_section_end|>";
static const char *TAG_KIMI_TC_START  = "<|tool_call_begin|>";
static const char *TAG_KIMI_ARG_START = "<|tool_call_argument_begin|>";
static const char *TAG_KIMI_TC_END    = "<|tool_call_end|>";
/* Unicode tags — stored as their raw UTF-8 byte sequences */
static const char *TAG_UNICODE_TC_BEGIN   = "<\xef\xbd\x9c""tool\xe2\x96\x81""calls\xe2\x96\x81""begin\xef\xbd\x9c>";
static const char *TAG_UNICODE_TC_END     = "<\xef\xbd\x9c""tool\xe2\x96\x81""calls\xe2\x96\x81""end\xef\xbd\x9c>";
static const char *TAG_UNICODE_CALL_BEGIN = "<\xef\xbd\x9c""tool\xe2\x96\x81""call\xe2\x96\x81""begin\xef\xbd\x9c>";
static const char *TAG_UNICODE_CALL_END   = "<\xef\xbd\x9c""tool\xe2\x96\x81""call\xe2\x96\x81""end\xef\xbd\x9c>";
static const char *TAG_UNICODE_SEP        = "<\xef\xbd\x9c""tool\xe2\x96\x81""sep\xef\xbd\x9c>";

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static inline int
sw(const char *buf, Py_ssize_t buflen, const char *tag)
{
    Py_ssize_t tlen = (Py_ssize_t)strlen(tag);
    if (buflen < tlen) return 0;
    return memcmp(buf, tag, (size_t)tlen) == 0;
}

/* Returns 1 if buf is a proper prefix of any tag in the array */
static int
is_partial(const char *buf, Py_ssize_t buflen,
           const char **tags, int ntags)
{
    for (int i = 0; i < ntags; i++) {
        Py_ssize_t tlen = (Py_ssize_t)strlen(tags[i]);
        if (buflen >= tlen) continue;           /* buf longer: not a prefix  */
        if (memcmp(buf, tags[i], (size_t)buflen) == 0) return 1;
    }
    return 0;
}

/*
 * Append a {type, content, run_id} dict to the events list.
 * content is taken from [cbuf, cbuf+clen) interpreted as UTF-8.
 */
static int
emit(PyObject *events,
     const char *etype,
     const char *cbuf, Py_ssize_t clen,
     PyObject *run_id_obj)
{
    PyObject *d = PyDict_New();
    if (!d) return -1;

    PyObject *t = PyUnicode_FromString(etype);
    PyObject *c = PyUnicode_DecodeUTF8(cbuf, clen, "replace");
    if (!t || !c) { Py_XDECREF(t); Py_XDECREF(c); Py_DECREF(d); return -1; }

    PyDict_SetItemString(d, "type",   t);
    PyDict_SetItemString(d, "content", c);
    PyDict_SetItemString(d, "run_id",  run_id_obj);
    Py_DECREF(t); Py_DECREF(c);

    int r = PyList_Append(events, d);
    Py_DECREF(d);
    return r;
}

/*
 * Convenience: emit a single-char event (buf[0]) advancing buf by one byte.
 * Caller must ensure buflen > 0.
 * NOTE: advances *p and decrements *plen by 1 byte only — safe for ASCII.
 *       For multi-byte UTF-8 chars the Python fallback handles them; we only
 *       reach here on ASCII bytes in those paths.
 */
static inline int
emit_char_advance(PyObject *events, const char *etype,
                  const char **p, Py_ssize_t *plen,
                  PyObject *run_id_obj)
{
    if (emit(events, etype, *p, 1, run_id_obj) < 0) return -1;
    (*p)++;
    (*plen)--;
    return 0;
}

/* ── Dynamic string builder ───────────────────────────────────────────────── */
typedef struct {
    char   *data;
    Py_ssize_t len;
    Py_ssize_t cap;
} StrBuf;

static int sb_init(StrBuf *sb, Py_ssize_t initial) {
    sb->data = (char *)PyMem_Malloc((size_t)initial);
    if (!sb->data) { PyErr_NoMemory(); return -1; }
    sb->len = 0;
    sb->cap = initial;
    return 0;
}

static int sb_append(StrBuf *sb, const char *src, Py_ssize_t n) {
    if (sb->len + n > sb->cap) {
        Py_ssize_t newcap = (sb->cap + n) * 2;
        char *tmp = (char *)PyMem_Realloc(sb->data, (size_t)newcap);
        if (!tmp) { PyErr_NoMemory(); return -1; }
        sb->data = tmp;
        sb->cap  = newcap;
    }
    memcpy(sb->data + sb->len, src, (size_t)n);
    sb->len += n;
    return 0;
}

static void sb_free(StrBuf *sb) { PyMem_Free(sb->data); sb->data = NULL; }

/* ── Main exported function ───────────────────────────────────────────────── */

static PyObject *
py_process_buffer(PyObject *self, PyObject *args)
{
    const char *buf_in;    Py_ssize_t buf_in_len;
    int         state;
    int         json_depth;
    int         has_emitted_text_in;
    const char *xml_buf_in; Py_ssize_t xml_buf_in_len;
    PyObject   *run_id_obj;

    if (!PyArg_ParseTuple(args, "z#iiiz#U",
                          &buf_in,     &buf_in_len,
                          &state,
                          &json_depth,
                          &has_emitted_text_in,
                          &xml_buf_in, &xml_buf_in_len,
                          &run_id_obj))
        return NULL;

    /* Working copies */
    StrBuf buf;
    if (sb_init(&buf, buf_in_len > 0 ? buf_in_len : 1) < 0) return NULL;
    if (buf_in_len > 0 && sb_append(&buf, buf_in, buf_in_len) < 0) {
        sb_free(&buf); return NULL;
    }

    StrBuf xml_buf;
    if (sb_init(&xml_buf, xml_buf_in_len > 0 ? xml_buf_in_len : 1) < 0) {
        sb_free(&buf); return NULL;
    }
    if (xml_buf_in_len > 0 && sb_append(&xml_buf, xml_buf_in, xml_buf_in_len) < 0) {
        sb_free(&buf); sb_free(&xml_buf); return NULL;
    }

    int has_emitted_text = has_emitted_text_in;

    PyObject *events = PyList_New(0);
    if (!events) { sb_free(&buf); sb_free(&xml_buf); return NULL; }

    /* Pointers into buf.data that we advance as we consume */
    const char *p    = buf.data;
    Py_ssize_t  plen = buf.len;

#define FAIL  do { Py_DECREF(events); sb_free(&buf); sb_free(&xml_buf); return NULL; } while(0)
#define EMIT(etype, cbuf, clen)   if (emit(events, (etype), (cbuf), (clen), run_id_obj) < 0) FAIL
#define EMITC(etype)              if (emit_char_advance(events, (etype), &p, &plen, run_id_obj) < 0) FAIL
#define SKIP(n)                   do { p += (n); plen -= (n); } while(0)

    /* ── State machine loop ───────────────────────────────────────────────── */
    int loop_again = 1;
    while (plen > 0 && loop_again) {
        loop_again = 0;  /* set to 1 whenever we make progress */

        /* ── content ──────────────────────────────────────────────────────── */
        if (state == ST_CONTENT) {
            /* Fast path: no special chars in remainder */
            int has_lt  = (memchr(p, '<',  (size_t)plen) != NULL);
            int has_bt  = (memchr(p, '`',  (size_t)plen) != NULL);
            int has_cur = (!has_emitted_text && memchr(p, '{', (size_t)plen) != NULL);

            if (!has_lt && !has_bt && !has_cur) {
                /* Pure text — emit all at once */
                if (plen > 0) {
                    /* check non-whitespace */
                    for (Py_ssize_t i = 0; i < plen; i++) {
                        if ((unsigned char)p[i] > 32) { has_emitted_text = 1; break; }
                    }
                    EMIT("content", p, plen);
                }
                p += plen; plen = 0;
                break;
            }

            /* Find first potential special-char position */
            Py_ssize_t lt_idx  = -1, bt_idx = -1, cur_idx = -1;
            {
                const char *pp;
                pp = (const char *)memchr(p, '<', (size_t)plen);
                if (pp) lt_idx = pp - p;
                pp = (const char *)memchr(p, '`', (size_t)plen);
                if (pp) bt_idx = pp - p;
                if (!has_emitted_text) {
                    pp = (const char *)memchr(p, '{', (size_t)plen);
                    if (pp) cur_idx = pp - p;
                }
            }

            Py_ssize_t cutoff = plen;
            if (lt_idx  >= 0 && lt_idx  < cutoff) cutoff = lt_idx;
            if (bt_idx  >= 0 && bt_idx  < cutoff) cutoff = bt_idx;
            if (cur_idx >= 0 && cur_idx < cutoff) cutoff = cur_idx;

            if (cutoff > 0) {
                for (Py_ssize_t i = 0; i < cutoff; i++) {
                    if ((unsigned char)p[i] > 32) { has_emitted_text = 1; break; }
                }
                EMIT("content", p, cutoff);
                SKIP(cutoff);
                loop_again = 1;
                continue;
            }

            /* At a potential tag start — build tag table */
            typedef struct { const char *tag; int new_state; } TagEntry;
            TagEntry all_tags[] = {
                { TAG_CH_ANALYSIS,    ST_CHANNEL_REASONING },
                { TAG_CH_COMMENTARY,  ST_CHANNEL_TOOL_META },
                { TAG_CH_FINAL,       -1 },
                { TAG_MSG,            -1 },
                { TAG_FC_START,       ST_FC },
                { TAG_TC_START,       ST_TOOL_CALL_XML },
                { TAG_TCODE_START,    ST_TOOL_CODE_XML },
                { TAG_MD_JSON,        ST_MD_JSON_BLOCK },
                { TAG_TH_START,       ST_THINK },
                { TAG_DEC_START,      ST_DECISION },
                { TAG_PLAN_START,     ST_PLAN },
                { TAG_KIMI_SEC_START, ST_KIMI_ROUTER },
                { TAG_UNICODE_TC_BEGIN, ST_UNICODE_TOOL_ROUTER },
                /* naked_json only if no text emitted yet */
                { TAG_NAKED_JSON,     has_emitted_text ? -2 : ST_NAKED_JSON },
            };
            int ntags = (int)(sizeof(all_tags) / sizeof(all_tags[0]));

            int matched = 0;
            for (int i = 0; i < ntags; i++) {
                if (all_tags[i].new_state == -2) continue; /* skip naked_json */
                Py_ssize_t tlen = (Py_ssize_t)strlen(all_tags[i].tag);
                if (!sw(p, plen, all_tags[i].tag)) continue;

                /* Full match */
                SKIP(tlen);
                int ns = all_tags[i].new_state;
                if (ns >= 0) {
                    state = ns;
                    if (ns == ST_NAKED_JSON)
                        json_depth = 1;
                    if (ns == ST_FC || ns == ST_TOOL_CALL_XML ||
                        ns == ST_TOOL_CODE_XML || ns == ST_MD_JSON_BLOCK) {
                        xml_buf.len = 0; /* reset xml accumulator */
                    }
                    if (ns == ST_NAKED_JSON) {
                        /* emit the opening brace as call_arguments */
                        EMIT("call_arguments", TAG_NAKED_JSON, 1);
                    }
                }
                matched = 1;
                loop_again = 1;
                break;
            }
            if (matched) continue;

            /* Check partial match */
            const char *tag_strs[16];
            int nts = 0;
            for (int i = 0; i < ntags; i++) {
                if (all_tags[i].new_state != -2)
                    tag_strs[nts++] = all_tags[i].tag;
            }
            if (is_partial(p, plen, tag_strs, nts)) {
                break; /* wait for more data */
            }

            /* No match, no partial — emit one char */
            if ((unsigned char)p[0] > 32) has_emitted_text = 1;
            EMITC("content");
            loop_again = 1;
        }

        /* ── think / plan / decision ──────────────────────────────────────── */
        else if (state == ST_THINK || state == ST_PLAN || state == ST_DECISION) {
            const char *end_tag;
            const char *type_name;
            if (state == ST_THINK) {
                end_tag   = TAG_TH_END;
                type_name = "reasoning";
            } else if (state == ST_PLAN) {
                end_tag   = TAG_PLAN_END;
                type_name = "plan";
            } else {
                end_tag   = TAG_DEC_END;
                type_name = "decision";
            }
            Py_ssize_t etlen = (Py_ssize_t)strlen(end_tag);

            if (memchr(p, '<', (size_t)plen) == NULL) {
                EMIT(type_name, p, plen);
                p += plen; plen = 0;
                break;
            }

            const char *lt = (const char *)memchr(p, '<', (size_t)plen);
            Py_ssize_t lt_idx = lt - p;
            if (lt_idx > 0) {
                EMIT(type_name, p, lt_idx);
                SKIP(lt_idx);
                loop_again = 1;
                continue;
            }

            if (sw(p, plen, end_tag)) {
                SKIP(etlen);
                state = ST_CONTENT;
                loop_again = 1;
                continue;
            }
            if (is_partial(p, plen, &end_tag, 1)) break;

            EMITC(type_name);
            loop_again = 1;
        }

        /* ── fc / tool_call_xml / tool_code_xml / md_json_block ──────────── */
        else if (state == ST_FC || state == ST_TOOL_CALL_XML ||
                 state == ST_TOOL_CODE_XML || state == ST_MD_JSON_BLOCK) {
            const char *end_tag;
            if      (state == ST_FC)            end_tag = TAG_FC_END;
            else if (state == ST_TOOL_CALL_XML) end_tag = TAG_TC_END;
            else if (state == ST_TOOL_CODE_XML) end_tag = TAG_TCODE_END;
            else                                end_tag = TAG_MD_END;

            Py_ssize_t etlen = (Py_ssize_t)strlen(end_tag);

            if (sw(p, plen, end_tag)) {
                SKIP(etlen);
                state = ST_CONTENT;

                /* Emit accumulated xml_buf as a Python bytes object for
                 * JSON extraction — we signal this with a special event type
                 * "tool_call_raw_xml" so the Python wrapper can call
                 * _extract_json on it (that part stays in Python). */
                if (xml_buf.len > 0) {
                    EMIT("tool_call_raw_xml", xml_buf.data, xml_buf.len);
                    xml_buf.len = 0;
                }
                loop_again = 1;
                continue;
            }
            if (is_partial(p, plen, &end_tag, 1)) break;

            char first = end_tag[0];
            const char *fc = (const char *)memchr(p, first, (size_t)plen);
            Py_ssize_t idx = fc ? (fc - p) : -1;

            if (idx < 0) {
                /* No potential end-tag start: consume all */
                if (sb_append(&xml_buf, p, plen) < 0) FAIL;
                EMIT("call_arguments", p, plen);
                p += plen; plen = 0;
                break;
            } else if (idx > 0) {
                if (sb_append(&xml_buf, p, idx) < 0) FAIL;
                EMIT("call_arguments", p, idx);
                SKIP(idx);
                loop_again = 1;
                continue;
            } else {
                /* idx == 0: might be start of end_tag or just the char */
                if (sb_append(&xml_buf, p, 1) < 0) FAIL;
                EMITC("call_arguments");
                loop_again = 1;
            }
        }

        /* ── naked_json ───────────────────────────────────────────────────── */
        else if (state == ST_NAKED_JSON) {
            Py_ssize_t i;
            for (i = 0; i < plen; i++) {
                if      (p[i] == '{') json_depth++;
                else if (p[i] == '}') { json_depth--; if (json_depth == 0) { i++; break; } }
            }
            EMIT("call_arguments", p, i);
            SKIP(i);
            if (json_depth == 0) {
                state = ST_CONTENT;
            }
            loop_again = 1;
        }

        /* ── kimi_router ──────────────────────────────────────────────────── */
        else if (state == ST_KIMI_ROUTER) {
            if (sw(p, plen, TAG_KIMI_SEC_END)) {
                SKIP((Py_ssize_t)strlen(TAG_KIMI_SEC_END));
                state = ST_CONTENT; loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_KIMI_ARG_START)) {
                SKIP((Py_ssize_t)strlen(TAG_KIMI_ARG_START));
                state = ST_KIMI_ARGS; loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_KIMI_TC_START)) {
                SKIP((Py_ssize_t)strlen(TAG_KIMI_TC_START));
                loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_KIMI_TC_END)) {
                SKIP((Py_ssize_t)strlen(TAG_KIMI_TC_END));
                loop_again = 1; continue;
            }
            const char *rtags[] = {
                TAG_KIMI_SEC_END, TAG_KIMI_ARG_START,
                TAG_KIMI_TC_START, TAG_KIMI_TC_END
            };
            if (is_partial(p, plen, rtags, 4)) break;
            SKIP(1); loop_again = 1;
        }

        /* ── kimi_args ────────────────────────────────────────────────────── */
        else if (state == ST_KIMI_ARGS) {
            if (sw(p, plen, TAG_KIMI_TC_END)) {
                SKIP((Py_ssize_t)strlen(TAG_KIMI_TC_END));
                state = ST_KIMI_ROUTER; loop_again = 1; continue;
            }
            if (is_partial(p, plen, &TAG_KIMI_TC_END, 1)) break;
            EMITC("call_arguments"); loop_again = 1;
        }

        /* ── unicode_tool_router ──────────────────────────────────────────── */
        else if (state == ST_UNICODE_TOOL_ROUTER) {
            if (sw(p, plen, TAG_UNICODE_TC_END)) {
                SKIP((Py_ssize_t)strlen(TAG_UNICODE_TC_END));
                state = ST_CONTENT; loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_UNICODE_CALL_BEGIN)) {
                SKIP((Py_ssize_t)strlen(TAG_UNICODE_CALL_BEGIN));
                state = ST_UNICODE_TOOL_PARSING; loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_UNICODE_CALL_END)) {
                SKIP((Py_ssize_t)strlen(TAG_UNICODE_CALL_END));
                loop_again = 1; continue;
            }
            const char *rtags[] = {
                TAG_UNICODE_TC_END, TAG_UNICODE_CALL_BEGIN, TAG_UNICODE_CALL_END
            };
            if (is_partial(p, plen, rtags, 3)) break;
            SKIP(1); loop_again = 1;
        }

        /* ── unicode_tool_parsing ─────────────────────────────────────────── */
        else if (state == ST_UNICODE_TOOL_PARSING) {
            if (sw(p, plen, TAG_UNICODE_SEP)) {
                SKIP((Py_ssize_t)strlen(TAG_UNICODE_SEP));
                state = ST_UNICODE_TOOL_ARGS; loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_UNICODE_CALL_END)) {
                SKIP((Py_ssize_t)strlen(TAG_UNICODE_CALL_END));
                state = ST_UNICODE_TOOL_ROUTER; loop_again = 1; continue;
            }
            const char *rtags[] = { TAG_UNICODE_SEP, TAG_UNICODE_CALL_END };
            if (is_partial(p, plen, rtags, 2)) break;
            EMITC("call_arguments"); loop_again = 1;
        }

        /* ── unicode_tool_args ────────────────────────────────────────────── */
        else if (state == ST_UNICODE_TOOL_ARGS) {
            if (sw(p, plen, TAG_UNICODE_CALL_END)) {
                SKIP((Py_ssize_t)strlen(TAG_UNICODE_CALL_END));
                state = ST_UNICODE_TOOL_ROUTER; loop_again = 1; continue;
            }
            if (is_partial(p, plen, &TAG_UNICODE_CALL_END, 1)) break;
            EMITC("call_arguments"); loop_again = 1;
        }

        /* ── channel_reasoning ───────────────────────────────────────────── */
        else if (state == ST_CHANNEL_REASONING) {
            if (sw(p, plen, TAG_CH_FINAL)) {
                SKIP((Py_ssize_t)strlen(TAG_CH_FINAL));
                state = ST_CONTENT; loop_again = 1; continue;
            }
            if (sw(p, plen, TAG_CH_COMMENTARY)) {
                SKIP((Py_ssize_t)strlen(TAG_CH_COMMENTARY));
                state = ST_CHANNEL_TOOL_META; loop_again = 1; continue;
            }
            const char *rtags[] = { TAG_CH_FINAL, TAG_CH_COMMENTARY };
            if (is_partial(p, plen, rtags, 2)) break;
            EMITC("reasoning"); loop_again = 1;
        }

        /* ── channel_tool_meta ───────────────────────────────────────────── */
        else if (state == ST_CHANNEL_TOOL_META) {
            Py_ssize_t msg_len = (Py_ssize_t)strlen(TAG_MSG);
            Py_ssize_t fin_len = (Py_ssize_t)strlen(TAG_CH_FINAL);
            /* scan for TAG_MSG or TAG_CH_FINAL */
            const char *mp = (const char *)memmem(p, (size_t)plen,
                                                   TAG_MSG, (size_t)msg_len);
            const char *fp = (const char *)memmem(p, (size_t)plen,
                                                   TAG_CH_FINAL, (size_t)fin_len);
            if (mp && (!fp || mp <= fp)) {
                Py_ssize_t off = mp - p;
                SKIP(off + msg_len);
                state = ST_CHANNEL_TOOL_PAYLOAD;
                loop_again = 1; continue;
            }
            if (fp) {
                Py_ssize_t off = fp - p;
                SKIP(off + fin_len);
                state = ST_CONTENT;
                loop_again = 1; continue;
            }
            /* partial? */
            const char *rtags[] = { TAG_MSG, TAG_CH_FINAL };
            if (is_partial(p, plen, rtags, 2)) break;
            /* discard */
            p += plen; plen = 0;
        }

        /* ── channel_tool_payload ─────────────────────────────────────────── */
        else if (state == ST_CHANNEL_TOOL_PAYLOAD) {
            const char *exit_tags[] = {
                TAG_CALL, TAG_CH_FINAL, TAG_CH_ANALYSIS
            };
            int nexits = 3;
            int found_exit = -1;
            Py_ssize_t found_off = plen;

            for (int i = 0; i < nexits; i++) {
                Py_ssize_t tlen = (Py_ssize_t)strlen(exit_tags[i]);
                const char *fp2 = (const char *)memmem(p, (size_t)plen,
                                                        exit_tags[i], (size_t)tlen);
                if (fp2) {
                    Py_ssize_t off = fp2 - p;
                    if (off < found_off) { found_off = off; found_exit = i; }
                }
            }

            if (found_exit >= 0) {
                /* emit content before the tag */
                if (found_off > 0) {
                    EMIT("call_arguments", p, found_off);
                }
                Py_ssize_t tlen = (Py_ssize_t)strlen(exit_tags[found_exit]);
                SKIP(found_off + tlen);
                state = (found_exit == 2) ? ST_CHANNEL_REASONING : ST_CONTENT;
                loop_again = 1;
                continue;
            }
            /* check partial */
            if (is_partial(p, plen, exit_tags, nexits)) break;
            EMITC("call_arguments"); loop_again = 1;
        }

        else {
            /* Unknown state — should never happen; just advance */
            SKIP(1);
            loop_again = 1;
        }
    } /* while */

    /* Build remaining buffer string from [p, p+plen) */
    PyObject *new_buf    = PyUnicode_DecodeUTF8(p, plen, "replace");
    PyObject *new_xml    = PyUnicode_DecodeUTF8(xml_buf.data, xml_buf.len, "replace");
    if (!new_buf || !new_xml) {
        Py_XDECREF(new_buf); Py_XDECREF(new_xml);
        FAIL;
    }

    sb_free(&buf);
    sb_free(&xml_buf);

    /* Return: (events, new_buf, new_state, json_depth, has_emitted_text, new_xml) */
    PyObject *result = Py_BuildValue(
        "(OOiiiO)",
        events,
        new_buf,
        (int)state,
        json_depth,
        has_emitted_text,
        new_xml
    );

    /* Py_BuildValue increfs; release our refs */
    Py_DECREF(events);
    Py_DECREF(new_buf);
    Py_DECREF(new_xml);
    return result;

#undef FAIL
#undef EMIT
#undef EMITC
#undef SKIP
}

/* ── Module definition ────────────────────────────────────────────────────── */

static PyMethodDef methods[] = {
    {"process_buffer", py_process_buffer, METH_VARARGS,
     "process_buffer(buf, state, json_depth, has_emitted_text, xml_buf, run_id)"
     " -> (events, new_buf, new_state, new_json_depth, new_has_emitted_text, new_xml_buf)\n\n"
     "Single-call replacement for the inner while-buffer loop in DeltaNormalizer."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,   /* m_base    */
    "delta_normalizer_core", /* m_name    */
    NULL,                    /* m_doc     */
    -1,                      /* m_size    */
    methods,                 /* m_methods */
    NULL,                    /* m_slots   */
    NULL,                    /* m_traverse*/
    NULL,                    /* m_clear   */
    NULL,                    /* m_free    */
};

PyMODINIT_FUNC
PyInit_delta_normalizer_core(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    /* Expose state constants so Python side doesn't need to hardcode ints */
#define ADD_CONST(name, val)  PyModule_AddIntConstant(m, name, val)
    ADD_CONST("ST_CONTENT",               ST_CONTENT);
    ADD_CONST("ST_THINK",                 ST_THINK);
    ADD_CONST("ST_PLAN",                  ST_PLAN);
    ADD_CONST("ST_DECISION",              ST_DECISION);
    ADD_CONST("ST_FC",                    ST_FC);
    ADD_CONST("ST_TOOL_CALL_XML",         ST_TOOL_CALL_XML);
    ADD_CONST("ST_TOOL_CODE_XML",         ST_TOOL_CODE_XML);
    ADD_CONST("ST_MD_JSON_BLOCK",         ST_MD_JSON_BLOCK);
    ADD_CONST("ST_NAKED_JSON",            ST_NAKED_JSON);
    ADD_CONST("ST_KIMI_ROUTER",           ST_KIMI_ROUTER);
    ADD_CONST("ST_KIMI_ARGS",             ST_KIMI_ARGS);
    ADD_CONST("ST_UNICODE_TOOL_ROUTER",   ST_UNICODE_TOOL_ROUTER);
    ADD_CONST("ST_UNICODE_TOOL_PARSING",  ST_UNICODE_TOOL_PARSING);
    ADD_CONST("ST_UNICODE_TOOL_ARGS",     ST_UNICODE_TOOL_ARGS);
    ADD_CONST("ST_CHANNEL_REASONING",     ST_CHANNEL_REASONING);
    ADD_CONST("ST_CHANNEL_TOOL_META",     ST_CHANNEL_TOOL_META);
    ADD_CONST("ST_CHANNEL_TOOL_PAYLOAD",  ST_CHANNEL_TOOL_PAYLOAD);
#undef ADD_CONST

    return m;
}
