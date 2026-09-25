#!/usr/bin/env python3
"""
TEXT SPLITTER // KEV  -  Kev-0.8B cuts a document into retrieval chunks, and you watch its decision graph fire.

The sibling of SNAKE // KEV. One file, one native window (tkinter), no web server. The document is first cut
into sentences (paragraph-, heading-, list- and code-aware, all plain code). Then, one chunk at a time, the
text window ahead of the cursor is written out with numbered markers at every place the chunk may end, Kev
reads it once (prefill) and answers a small graph of questions:

    known facts ──► offers cuts (sentence ends inside the size budget) ──────────────────► [cut where] ─► answer → cut ─► CUT
                                                                                    │          (or a reflex if only one end is legal)
    window → kev prefill ─► [topic shift] ─► shift → cut rule ─────────────────────┘                     │
                                                                                          ├─► [kind] ──────────► TAG
                                                                                          └─► [cut quality] ───► CLEAN

Amber boxes are Kev questions (pointer-head choice / score), ellipses are plain code combining answers, dashed
boxes are facts the code computes. Nodes light up in the order they are actually computed; the right-hand
panel shows each question with Kev's probability for every option. Finished chunks are shaded in the text
view with their tag (narrative / dialogue / explanation / list / code / heading) and cut quality.

Install (Python 3.10+; 3.12 or 3.13 recommended):
    pip install torch "transformers>=5.17" huggingface_hub
    (NVIDIA GPU: install the CUDA build of torch from pytorch.org first; it is picked up automatically)

Run:
    python text_splitter_kev.py                       # built-in demo document
    python text_splitter_kev.py my_doc.txt            # your own text / markdown file
    python text_splitter_kev.py doc.md --target 800   # aim for ~800-char chunks (budget = 0.5x .. 1.5x target)
    python text_splitter_kev.py --mock                # preview the UI with a fake heuristic "model" (no download)
    python text_splitter_kev.py doc.txt --headless --out chunks.jsonl   # no window: split and save

Keys:  space pause/resume · n single step · r restart · + / - speed · [ / ] smaller / bigger chunks
       f fire-animation speed · s situation report (the exact text Kev reads) · o open file · v paste clipboard
       e export chunks (.jsonl / .json) · esc quit

Kev: https://github.com/jaredpalmer/kev (Apache-2.0).
"""
import argparse
import collections
import copy
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
import traceback

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

KEV_REPO = "jaredpalmer/kev-0.8b"


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Kev: Qwen3.5 backbone + merged LoRA + pointer head   (identical to snake_kev.py)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

# Kev reuses rarely-used Qwen special tokens as delimiters: <state> <q> <opt> </opt> <decide>
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class Kev:
    """Prefill the state once, then answer each question as its own row continuing the cached state
    (the "row form" kev uses for the hybrid Qwen3.5 backbone). Returns calibrated probabilities."""

    def __init__(self, repo=KEV_REPO, device=None, threads=None, log=print):
        import torch
        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

        self.torch, self.F, self.DynamicCache = torch, torch.nn.functional, DynamicCache
        torch.set_grad_enabled(False)
        if threads: torch.set_num_threads(threads)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        log(f"downloading {repo} (adapter + head) ...")
        ck = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.pt"])
        meta = torch.load(os.path.join(ck, "head.pt"), map_location="cpu")
        base, rev = meta["base"], meta.get("base_revision")
        log(f"loading tokenizer + {base} (first run downloads ~1.7 GB) ...")
        self.tok = AutoTokenizer.from_pretrained(base, revision=rev)
        lm = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.float32).model  # backbone only

        log("merging LoRA adapter ...")
        cfg = json.load(open(os.path.join(ck, "adapter_config.json"), encoding="utf-8"))
        scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
        weights = load_file(os.path.join(ck, "adapter_model.safetensors"))
        modules = dict(lm.named_modules())
        for name, a in weights.items():
            if not name.endswith(".lora_A.weight"): continue
            stem = name[: -len(".lora_A.weight")]
            mod = modules[stem.replace("base_model.model.", "", 1)]
            mod.weight += (weights[stem + ".lora_B.weight"].float() @ a.float()) * scale   # exact in fp32
        del weights

        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.lm = lm.to(device=device, dtype=self.dtype).eval()
        d, dp = lm.config.hidden_size, meta.get("head_dim", 256)
        self.hq, self.hk = nn.Linear(d, dp), nn.Linear(d, dp)
        self.hq.weight.data, self.hq.bias.data = meta["head"]["q.weight"], meta["head"]["q.bias"]
        self.hk.weight.data, self.hk.bias.data = meta["head"]["k.weight"], meta["head"]["k.bias"]
        self.hq.to(device); self.hk.to(device)
        self.scale, self.temperature = 1 / math.sqrt(dp), float(meta.get("temperature", 1.0))
        self.ids = [self.tok.convert_tokens_to_ids(t) for t in SPECIAL]
        self.name = f"kev-0.8b · {device} {str(self.dtype).removeprefix('torch.')}"
        log("warming up ...")
        self.ask(self.prefill("warm up"), "Is this a test?", ["no", "yes"])

    def _tokens(self, text):
        # caller text can never forge a delimiter token
        return self.tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids

    def prefill(self, state):
        with self.torch.no_grad():
            return self._prefill(state)

    def ask(self, prefix, instructions, options):
        with self.torch.no_grad():
            return self._ask(prefix, instructions, options)

    def _prefill(self, state):
        torch = self.torch
        S = [self.ids[0]] + self._tokens(state)
        out = self.lm(input_ids=torch.tensor([S], device=self.device),
                      position_ids=torch.arange(len(S), device=self.device)[None],
                      past_key_values=self.DynamicCache(config=self.lm.config), use_cache=True)
        return len(S), out.past_key_values

    def _ask(self, prefix, instructions, options):
        torch = self.torch
        n_state, cache = prefix
        q_id, o_id, c_id, d_id = self.ids[1:]
        row, ends = [q_id] + self._tokens(instructions), []
        for opt in options:
            row += [o_id] + self._tokens(opt) + [c_id]
            ends.append(len(row) - 1)           # each option is read at its </opt> token
        row.append(d_id)                        # ... against the <decide> token
        h = self.lm(input_ids=torch.tensor([row], device=self.device),
                    position_ids=torch.arange(n_state, n_state + len(row), device=self.device)[None],
                    past_key_values=copy.deepcopy(cache), use_cache=True).last_hidden_state[0].float()
        z = (self.hk(h[ends]) @ self.hq(h[-1])) * self.scale / self.temperature
        return self.F.softmax(z, -1).tolist()


class MockKev:
    """Stand-in for --mock: noisy probabilities nudged by the heuristic hints, with a fake latency."""
    name = "mock model (heuristic + noise)"

    def prefill(self, state):
        time.sleep(0.15); return (len(state) // 4, None)

    def ask(self, prefix, instructions, options, hint=None):
        time.sleep(0.12)
        hint = hint or [0.0] * len(options)
        z = [h * 2.6 + random.gauss(0, 0.45) for h in hint]
        m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
        return [v / s for v in e]


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Document: sentences, paragraphs, headings, word overlap  (all plain code)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

ABBR = {"mr", "mrs", "ms", "dr", "st", "sr", "jr", "prof", "vs", "etc", "e.g", "i.e", "no", "fig", "vol", "ch", "p",
        "pp", "inc", "ltd", "co", "mt", "capt", "gen", "col", "lt", "sgt", "rev", "approx", "dept", "est", "jan",
        "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "cf", "al", "ca", "op"}
SENT_END = re.compile(r"""[.!?…]+["'”’)\]]*(?=\s)""")
LIST_RE = re.compile(r"^\s*(?:[-*•+]\s|\d{1,3}[.)]\s|[a-zA-Z][.)]\s)")
HEAD_RE = re.compile(r"^\s*#{1,6}\s")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SPEECH_RE = re.compile(r"\s+[A-Z][\w.]*(?: [A-Z][\w.]*)? (?:asked|said|replied|cried|called|shouted|whispered|answered|added|muttered|agreed)\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
STOP = set("""a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves said says one two also like well much many may might must shall upon yet still even ever back
went came come go get got make made see saw know knew think thought say""".split())

MAX_CANDS = 7        # at most this many possible ends are offered per chunk
LOOKAHEAD = 260      # characters of preview shown after the last possible end


def collapse(s):
    return re.sub(r"\s+", " ", s).strip()


def _trim(text, s, e):
    while s < e and text[s].isspace(): s += 1
    while e > s and text[e - 1].isspace(): e -= 1
    return s, e


def paragraphs(text):
    """-> [(start, end, kind)] with kind in text / heading / list / code. Handles blank-line paragraphs,
    one-paragraph-per-line text, markdown headings, list items and fenced code blocks."""
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line), line)); pos += len(line) + 1
    nonblank = [l for l in lines if l[2].strip()]
    if not nonblank: return []
    blank_seps = sum(1 for a, b in zip(lines, lines[1:]) if not a[2].strip() and b[2].strip())
    lens = sorted(len(l[2]) for l in nonblank)
    per_line = blank_seps < 0.1 * len(nonblank) or lens[len(lens) // 2] > 160
    paras, cur, prev_blank, i = [], None, True, 0

    def close():
        nonlocal cur
        if cur: paras.append(tuple(cur)); cur = None

    while i < len(lines):
        s, e, line = lines[i]
        if not line.strip():
            close(); prev_blank = True; i += 1; continue
        if FENCE_RE.match(line):
            close()
            j = i + 1
            while j < len(lines) and not FENCE_RE.match(lines[j][2]): j += 1
            j = min(j, len(lines) - 1)
            paras.append((*_trim(text, s, lines[j][1]), "code"))
            i, prev_blank = j + 1, False; continue
        kind = "heading" if HEAD_RE.match(line) else "list" if LIST_RE.match(line) else "text"
        if prev_blank or per_line or kind != "text" or cur is None or cur[2] != "text":
            close(); cur = [s, e, kind]
        else:
            cur[1] = e
        prev_blank = False; i += 1
    close()
    out = []
    for s, e, kind in paras:
        s, e = _trim(text, s, e)
        body = text[s:e]
        if (kind == "text" and "\n" not in body and len(body) <= 70 and re.search(r"[A-Za-z]", body)
                and not re.search(r"""[.!?,;:"'”’)\]]$""", body)):
            kind = "heading"                                     # short standalone line without punctuation
        out.append((s, e, kind))
    return out


def segment(text):
    """-> units (sentences / headings / list items / code blocks): {start, end, para, kind}."""
    units = []
    for ps, pe, kind in paragraphs(text):
        if kind != "text":
            units.append({"start": ps, "end": pe, "para": True, "kind": kind}); continue
        body, starts, ends = text[ps:pe], [0], []
        for m in SENT_END.finditer(body):
            nxt = re.match(r"\s+(\S)", body[m.end():])
            if not nxt: continue
            ch = nxt.group(1)
            if not (ch.isupper() or ch.isdigit() or ch in "\"'“‘([¿¡"): continue
            if re.search(r"[\"'”’]$", m.group(0)) and SPEECH_RE.match(body[m.end():]): continue   # "Hi?" Tomas asked
            if m.group(0)[0] == ".":
                prev = re.search(r"(\S+)$", body[:m.start()])
                w = prev.group(1).lower().lstrip("(\"'“‘") if prev else ""
                if w in ABBR or (len(w) == 1 and w.isalpha()): continue     # Mr. / e.g. / J. R. R.
            ends.append(m.end()); starts.append(m.end() + len(nxt.group(0)) - 1)
        ends.append(len(body))
        for k, (a, b) in enumerate(zip(starts, ends)):
            s, e = _trim(text, ps + a, ps + b)
            if e > s: units.append({"start": s, "end": e, "para": k == 0, "kind": "text"})
    return units


def _bag(s):
    c = collections.Counter()
    for w in WORD_RE.findall(s):
        w = w.lower().strip("'-")
        if len(w) > 2 and w not in STOP:
            c[w[:-1] if w.endswith("s") and len(w) > 4 else w] += 1
    return c


def _cos(a, b):
    if not a or not b: return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def cohesion(text, units, w=3):
    """Word overlap between the `w` units before and after every boundary (TextTiling-style). Index j = the
    boundary right before unit j. Low values hint at a topic change."""
    bags = [_bag(text[u["start"]:u["end"]]) for u in units]
    out = [0.0] * (len(units) + 1)
    for j in range(1, len(units)):
        left, right = collections.Counter(), collections.Counter()
        for b in bags[max(0, j - w):j]: left.update(b)
        for b in bags[j:j + w]: right.update(b)
        out[j] = _cos(left, right)
    return out


class Doc:
    def __init__(self, text, name="document", target=500):
        self.text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.name = name
        self.units = segment(self.text)
        self.coh = cohesion(self.text, self.units)
        vals = sorted(self.coh[1:len(self.units)])
        self.low = vals[len(vals) // 4] if vals else 0.0          # bottom quarter = "few shared words"
        self.hi = vals[(3 * len(vals)) // 4] if vals else 1.0
        self.set_target(target)
        self.reset()

    def set_target(self, target):
        self.target = int(target)
        self.min_chars, self.max_chars = int(target * 0.5), int(target * 1.5)

    def reset(self):
        self.cursor, self.chunks = 0, []
        self.done = not self.units

    def start(self):
        return self.units[self.cursor]["start"] if not self.done else len(self.text)

    def commit(self, j, info):
        s, e = self.units[self.cursor]["start"], self.units[j - 1]["end"]
        self.chunks.append({"id": len(self.chunks), "start": s, "end": e, "chars": e - s, **info})
        self.cursor = j
        self.done = j >= len(self.units)

    def export(self, path):
        rows = [{"id": c["id"], "start": c["start"], "end": c["end"], "chars": c["chars"], "kind": c.get("kind"),
                 "cut_quality": c.get("quality"), "topic_shift": round(c.get("topic_shift", 0), 3),
                 "cut_confidence": round(c.get("cut_confidence", 0), 3), "forced": c.get("forced"),
                 "text": self.text[c["start"]:c["end"]]} for c in self.chunks]
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                json.dump({"document": self.name, "target": self.target, "chunks": rows}, f, ensure_ascii=False, indent=2)
            else:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)


def load_text(path):
    if path == "-": return sys.stdin.read()
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: continue


def prior(doc, j, size):
    """Plain-code prior for a possible end before unit j (used to pre-filter the offers and as mock hints)."""
    U, n = doc.units, len(doc.units)
    if j >= n: return 3.0
    p = 1.5 * U[j]["para"] + 2.0 * (U[j]["kind"] == "heading") - 2.5 * (U[j - 1]["kind"] == "heading")
    p += 1.2 * (1 - min(1.0, doc.coh[j] / (doc.hi or 1)))
    p += 1.0 * (1 - min(1.0, abs(size - doc.target) / doc.target))
    return p


def candidates(doc):
    """Possible chunk ends (unit indices j: cut right before unit j) + the reason when there is no real choice."""
    U, n, u0 = doc.units, len(doc.units), doc.cursor
    start = U[u0]["start"]
    size = lambda j: U[j - 1]["end"] - start
    if size(n) < doc.min_chars:
        return [n], "the rest of the document is shorter than the budget"
    inr, fits, j = [], [], u0 + 1
    while j <= n and size(j) <= doc.max_chars:
        fits.append(j)
        before_heading = j < n and U[j]["kind"] == "heading" and U[j - 1]["kind"] != "heading"
        if size(j) >= doc.min_chars or (before_heading and size(j) >= 0.4 * doc.min_chars):
            inr.append(j)                           # a heading may close a short chunk
        j += 1
    if not inr:
        if fits: return [fits[-1]], "no sentence ends inside the budget: longest fit"
        return [u0 + 1], "one sentence is longer than the whole budget"
    if len(inr) > MAX_CANDS:
        inr = sorted(sorted(inr, key=lambda j: -prior(doc, j, size(j)))[:MAX_CANDS])
    return inr, ("only one sentence end fits the budget" if len(inr) == 1 else "")


def cand_info(doc, j, i):
    U, n = doc.units, len(doc.units)
    start, end = U[doc.cursor]["start"], j >= n
    size = U[j - 1]["end"] - start
    return {"j": j, "i": i, "size": size, "end": end,
            "tail": collapse(doc.text[U[j - 1]["start"]:U[j - 1]["end"]])[-40:],
            "head": "" if end else collapse(doc.text[U[j]["start"]:U[j]["end"]])[:34],
            "para": end or U[j]["para"],
            "heading_next": not end and U[j]["kind"] == "heading",
            "ends_heading": U[j - 1]["kind"] == "heading",
            "coh": 0.0 if end else doc.coh[j],
            "low": (not end) and doc.coh[j] <= doc.low,
            "prior": prior(doc, j, size)}


def window_end(doc, cands):
    U, n = doc.units, len(doc.units)
    last = cands[-1]
    e, k = U[last - 1]["end"], last
    while k < n and k < last + 3 and U[k]["start"] < U[last - 1]["end"] + LOOKAHEAD:
        e = U[k]["end"]; k += 1
    return e


def describe(c):
    if c["end"]:
        s = f"the end of the document ({c['size']} characters)"
    else:
        s = f'after "…{c["tail"]}", before "{c["head"]}…", {c["size"]} characters'
    flags = []
    if c["para"] and not c["end"]: flags.append("ends a paragraph")
    if c["heading_next"]: flags.append("a heading comes next")
    if c["ends_heading"]: flags.append("the chunk would end on a heading")
    if c["low"]: flags.append("few shared words across the cut")
    return s + ("; " + ", ".join(flags) if flags else "")


def state_text(doc, info, win_end):
    """The exact text Kev reads: the task, where we are, and the window with numbered possible ends."""
    U, text, start = doc.units, doc.text, doc.start()
    k = len(doc.chunks)
    lines = [f"task: split a long document into chunks for a search index. A good chunk holds one complete idea, "
             f"scene or section, so it still makes sense when read on its own. Cuts go between sentences, never inside "
             f"one. Chunks should be about {doc.min_chars}-{doc.max_chars} characters long.",
             f"document: {doc.name}, {len(text)} characters. This is chunk {k + 1}; it starts at character {start} "
             f"({100 * start / max(1, len(text)):.0f}% of the way through)."]
    if doc.chunks:
        c = doc.chunks[-1]
        lines.append(f'the previous chunk ended with: "…{collapse(text[max(c["start"], c["end"] - 160):c["end"]])}"')
    else:
        lines.append("this is the start of the document.")
    lines.append("text (markers like ⟨2⟩ are the places where this chunk may end; text after the last marker is only a preview):")
    body, pos = [], start
    for c in info:
        e = U[c["j"] - 1]["end"]
        body.append(text[pos:e] + f" ⟨{c['i']}⟩")
        pos = e
    body.append(text[pos:win_end] + (" [end of document]" if info[-1]["end"] else " …"))
    lines.append("".join(body))
    facts = []
    paras = [f"⟨{c['i']}⟩" for c in info if c["para"] and not c["end"]]
    if paras: facts.append("paragraph ends at " + " ".join(paras))
    heads = [f"⟨{c['i']}⟩" for c in info if c["heading_next"]]
    if heads: facts.append("a heading follows " + " ".join(heads))
    inner = [c for c in info if not c["end"]]
    if len(inner) > 1:
        facts.append(f"fewest shared words across ⟨{min(inner, key=lambda c: c['coh'])['i']}⟩")
    if facts: lines.append("facts: " + "; ".join(facts) + ".")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  The decision graph (one "think" per chunk)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

TOPIC_LEVELS = ["no, one continuous passage", "a soft shift: a new beat, speaker or step", "a clear break: new scene, section or topic"]
QUALITY = ["clean: the next chunk starts a new idea", "acceptable: related ideas carry on", "rough: it splits one idea in two"]
CLEAN_OUT = ["clean", "ok", "rough"]
KINDS = [("narrative", "story or events told in prose"), ("dialogue", "mostly people talking"),
         ("explanation", "facts, reasoning or description"), ("list", "steps, items or a checklist"),
         ("code", "source code, data or markup"), ("heading", "titles, headings or front matter")]


def kind_hint(doc, s, e):
    t = doc.text[s:e]
    us = [u for u in doc.units if s <= u["start"] < e]
    n = max(1, len(us))
    prose = " ".join(doc.text[u["start"]:u["end"]] for u in us if u["kind"] == "text")
    h = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]
    quotes = len(re.findall(r"[\"“”]|(?:^|\s)'|'(?=\s|[,.!?])", prose))
    h[1] = min(1.4, quotes / max(1, len(t) / 45))
    h[0] += min(0.9, len(re.findall(r"\b\w+ed\b", prose)) / max(1, len(prose.split()) / 12))
    h[3] = 1.4 * sum(u["kind"] == "list" for u in us) / n
    h[4] = 1.6 * sum(u["kind"] == "code" for u in us) / n + min(0.8, len(re.findall(r"[{}();=<>]", t)) / max(1, len(t) / 25))
    h[5] = 1.5 * sum(u["kind"] == "heading" for u in us) / n
    h[2] += 0.4 * bool(re.search(r"\d", t)) + 0.3 * bool(re.search(r"\b(is|are|uses|means|because)\b", t))
    return h


def think(doc, kev, emit, mock=False):
    """Decide one chunk. Emits ("fire", node, value, sources) in the order things are computed, ("answer", ...)
    for the decisions panel, ("view", ...) / ("pick", j) for the text view, and finally ("done", j, info)."""
    t0 = time.time()
    U, text = doc.units, doc.text
    start, k = doc.start(), len(doc.chunks)

    def ask(prefix, instr, opts, hint):
        t = time.time()
        p = kev.ask(prefix, instr, opts, hint) if mock else kev.ask(prefix, instr, opts)
        return p, (time.time() - t) * 1000

    # ── plain code: possible ends + known facts
    cands, forced = candidates(doc)
    info = [cand_info(doc, j, i + 1) for i, j in enumerate(cands)]
    win_end = window_end(doc, cands)
    rest = U[-1]["end"] - start
    in_win = [u for u in U[doc.cursor:] if u["start"] < win_end]
    emit("fire", "document", f"chunk {k + 1} · {100 * start / max(1, len(text)):.0f}%", None)
    emit("fire", "budget", f"{doc.min_chars}–{doc.max_chars} · rest {rest}", None)
    emit("fire", "sentences", f"{len(in_win)} in window", None)
    paras = [f"⟨{c['i']}⟩" for c in info if c["para"] and not c["end"]]
    emit("fire", "paragraphs", " ".join(paras) if paras else "none", None)
    inner = [c for c in info if not c["end"]]
    lo = min(inner, key=lambda c: c["coh"]) if inner else None
    emit("fire", "cohesion", f"min ⟨{lo['i']}⟩ {lo['coh']:.2f}" if lo else "n/a", None)
    heads = [f"⟨{c['i']}⟩" for c in info if c["heading_next"]]
    emit("fire", "headings", ("before " + " ".join(heads)) if heads else "none", None)
    emit("fire", "offer_cuts", f"{len(cands)} possible end{'s' if len(cands) > 1 else ''}",
         ["budget", "sentences", "paragraphs", "cohesion", "headings"])
    emit("view", {"u0": doc.cursor, "cands": cands, "win_end": win_end, "pick": None})

    # ── Kev reads the window once
    state = state_text(doc, info, win_end)
    t = time.time()
    prefix = kev.prefill(state)
    emit("fire", "prefill", f"{prefix[0]} tok · {(time.time() - t) * 1000:.0f} ms", ["document"])
    emit("state", state)

    # ── topic shift (score)
    instr = ("Read the text from the start up to the last marker. Does it move on to a new topic, scene, "
             "speaker or section anywhere in there?")
    hint = [0.6, 0.8 if (paras or (lo and lo["low"])) else 0.1, 1.2 if (heads or (lo and lo["coh"] < doc.low * 0.5)) else 0.0]
    p, ms = ask(prefix, instr, TOPIC_LEVELS, hint)
    shift = sum(i * pi for i, pi in enumerate(p))
    ti = max(range(len(p)), key=p.__getitem__)
    emit("answer", "topic", {"instr": instr, "labels": ["0 continuous", "1 soft shift", "2 clear break"], "descs": TOPIC_LEVELS,
                             "probs": p, "pick": ti, "ms": ms, "score": shift})
    emit("fire", "q_topic", f"{['continuous', 'soft shift', 'clear break'][ti]} · {shift:.2f}", ["prefill"])

    # ── cut rule (code): topic answer + budget
    rule = ["end at the paragraph or sentence end closest to {t} characters",
            "end where the new beat starts, while staying near {t} characters",
            "end right before the clear break, so the new scene or section starts the next chunk"][ti].format(t=doc.target)
    emit("fire", "objective", ["near " + str(doc.target) + " ch", "at the new beat", "before the break"][ti], ["q_topic", "offer_cuts"])

    # ── cut where (pointer) or reflex
    if len(cands) == 1:
        ci, pc = 0, 1.0
        why = forced or "only one possible end"
        emit("answer", "cut", {"instr": "(skipped: reflex)", "labels": [f"⟨1⟩ {info[0]['size']}"], "descs": [why],
                               "probs": [1.0], "pick": 0, "ms": 0.0, "note": "reflex: one legal end, Kev is not asked"})
        emit("fire", "reflex", why, ["offer_cuts"])
        cut_src = ["reflex"]
    else:
        instr = f"This chunk should {rule}. At which marker should this chunk end?"
        hint = []
        for c in info:
            h = c["prior"] / 3
            if ti == 2: h += 0.8 * (c["heading_next"] or c["low"])
            if ti == 0: h += 0.6 * (1 - min(1, abs(c["size"] - doc.target) / doc.target))
            hint.append(h)
        p, ms = ask(prefix, instr, [f"⟨{c['i']}⟩: {describe(c)}" for c in info], hint)
        ci = max(range(len(p)), key=p.__getitem__)
        pc = p[ci]
        labels = [f"⟨{c['i']}⟩ {c['size']}" + (" ¶" if c["para"] and not c["end"] else "") + (" #" if c["heading_next"] else "")
                  + (" end" if c["end"] else "") for c in info]
        descs = [("·".join(f for f, on in (("low overlap", c["low"]), ("ends on heading", c["ends_heading"])) if on) + " " if (c["low"] or c["ends_heading"]) else "")
                 + (f"…{c['tail'][-20:]} ‖ {c['head'][:18]}…" if not c["end"] else "end of document") for c in info]
        emit("answer", "cut", {"instr": instr, "labels": labels, "descs": descs, "probs": p, "pick": ci, "ms": ms})
        emit("fire", "q_cut", f"⟨{info[ci]['i']}⟩ {p[ci] * 100:.0f}%", ["prefill", "objective", "offer_cuts"])
        cut_src = ["q_cut"]
    c = info[ci]
    emit("fire", "cut", f"⟨{c['i']}⟩ · {c['size']} ch", cut_src)
    emit("pick", c["j"])
    emit("fire", "CUT", f"✂ {c['size']} CH", ["cut"])

    # ── kind (choice) → TAG
    instr = f"The chunk is the text from the start up to marker ⟨{c['i']}⟩. What kind of text is it mostly?"
    p, ms = ask(prefix, instr, [f"{kk}: {d}" for kk, d in KINDS], kind_hint(doc, start, U[c["j"] - 1]["end"]))
    ki = max(range(len(p)), key=p.__getitem__)
    kind = KINDS[ki][0]
    emit("answer", "kind", {"instr": instr, "labels": [kk for kk, _ in KINDS], "descs": [d for _, d in KINDS], "probs": p, "pick": ki, "ms": ms})
    emit("fire", "q_kind", f"{kind} {p[ki] * 100:.0f}%", ["prefill", "cut"])
    emit("fire", "TAG", kind.upper(), ["q_kind"])

    # ── cut quality (score) → CLEAN
    if c["end"]:
        qi, qscore = 0, 0.0
        emit("answer", "quality", {"instr": "(skipped: end of document)", "labels": ["end"], "descs": ["nothing after this cut to judge"],
                                   "probs": [1.0], "pick": 0, "ms": 0.0, "note": "end of document: nothing to judge"})
        emit("fire", "q_clean", "end of document", ["cut"])
        emit("fire", "CLEAN", "END", ["q_clean"], 0)
    else:
        instr = f"The chunk ends at marker ⟨{c['i']}⟩ and the next chunk starts right after it. How clean is that cut?"
        good = c["para"] or c["heading_next"] or c["low"]
        hint = [1.0 + 0.5 * c["heading_next"], 0.4, -0.5] if good else [0.0, 0.6, 0.5 + 0.8 * c["ends_heading"]]
        p, ms = ask(prefix, instr, QUALITY, hint)
        qscore = sum(i * pi for i, pi in enumerate(p))
        qi = max(range(len(p)), key=p.__getitem__)
        emit("answer", "quality", {"instr": instr, "labels": ["0 clean", "1 acceptable", "2 rough"], "descs": QUALITY,
                                   "probs": p, "pick": qi, "ms": ms, "score": qscore})
        emit("fire", "q_clean", f"{CLEAN_OUT[qi]} · score {qscore:.2f}", ["prefill", "cut"])
        emit("fire", "CLEAN", CLEAN_OUT[qi].upper(), ["q_clean"], qi)

    emit("done", c["j"], {"kind": kind, "quality": "end" if c["end"] else CLEAN_OUT[qi], "quality_idx": qi,
                          "quality_score": qscore, "topic_shift": shift, "cut_confidence": pc, "forced": forced or None,
                          "marker": c["i"], "options": len(cands), "ms": (time.time() - t0) * 1000})


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Headless mode
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

def run_headless(kev, doc, mock, out):
    print(f"{doc.name}: {len(doc.text)} chars, {len(doc.units)} sentences/blocks, budget {doc.min_chars}-{doc.max_chars}\n")
    while not doc.done:
        answers, result = {}, {}

        def emit(kind, *a):
            if kind == "answer": answers[a[0]] = a[1]
            if kind == "done": result["j"], result["info"] = a[0], a[1]
        think(doc, kev, emit, mock)
        doc.commit(result["j"], result["info"])
        ch, parts = doc.chunks[-1], []
        for q in ("topic", "cut", "kind", "quality"):
            r = answers[q]
            parts.append(f"{q}={r['labels'][r['pick']].split(' ')[-1] if q != 'cut' else r['labels'][r['pick']].split(' ')[0]}({r['probs'][r['pick']]:.2f})")
        print(f"chunk {ch['id'] + 1:3d} [{ch['start']:6d}:{ch['end']:6d}] {ch['chars']:5d} ch | " + "  ".join(parts)
              + f"  [{result['info']['ms']:.0f} ms]")
        print(f"      “{collapse(doc.text[ch['start']:ch['end']])[:96]}…”")
    sizes = [c["chars"] for c in doc.chunks]
    print(f"\n{len(sizes)} chunks · avg {sum(sizes) / max(1, len(sizes)):.0f} · min {min(sizes, default=0)} · max {max(sizes, default=0)} chars")
    if out:
        doc.export(out); print(f"saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Window (tkinter)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

BG, PANEL, GRIDC = "#040705", "#070c08", "#0e1a11"
GREEN, GREEN_DIM, GREEN_FAINT, EDGE_DIM = "#3dff72", "#1d4a28", "#16301c", "#1f4629"
TXT, TXT_DIM = "#9dffb8", "#2f6b3d"
AMBER, AMBER_FILL, AMBER_DIM, AMBER_FILL_DIM, AMBER_TXT = "#ffb13b", "#3a2607", "#4d3610", "#110b02", "#ffd488"
RED, RED_DIM = "#ff4a5e", "#4a1a20"
ALERT_COL = [GREEN, AMBER, RED]

W, H = 1200, 856                        # base layout (scaled by --zoom / DPI)
TX1, TY1, TX2, TY2 = 12, 38, 560, 494   # text panel
DX1 = 570                               # decisions panel (to W - 12, 38 .. 520)
GX, GY, GW, GH = 12, 528, 1176, 320     # graph panel

# id: (kind, x, y, width, label)  (x, y = centre, relative to the graph panel)
NODES = {
    "document":   ("input",   76,  42, 128, "document"),
    "budget":     ("input",   76,  92, 128, "size budget"),
    "sentences":  ("input",   76, 142, 128, "sentences"),
    "paragraphs": ("input",   76, 192, 128, "paragraph ends"),
    "cohesion":   ("input",   76, 242, 128, "word overlap"),
    "headings":   ("input",   76, 292, 128, "headings"),
    "prefill":    ("comb",   240,  42, 140, "window → kev prefill"),
    "offer_cuts": ("offer",  240, 217, 116, "offers cuts"),
    "q_topic":    ("q",      392, 112, 128, "topic shift"),
    "objective":  ("comb",   540, 142, 140, "shift → cut rule"),
    "q_cut":      ("q",      690, 175, 124, "cut where"),
    "reflex":     ("reflex", 690, 272, 124, "reflex"),
    "cut":        ("comb",   840, 175, 128, "answer → cut"),
    "q_kind":     ("q",      986,  97, 120, "kind"),
    "q_clean":    ("q",      986, 252, 120, "cut quality"),
    "TAG":        ("out",   1112,  97, 104, "TAG"),
    "CUT":        ("out",   1112, 175, 104, "CUT"),
    "CLEAN":      ("out",   1112, 252, 104, "CLEAN"),
}
NODE_H = 34
EDGES = [
    ("document", "prefill", ""),
    ("prefill", "q_topic", "thin"), ("prefill", "q_cut", "thin"), ("prefill", "q_kind", "thin"), ("prefill", "q_clean", "thin"),
    ("budget", "offer_cuts", ""), ("sentences", "offer_cuts", ""), ("paragraphs", "offer_cuts", ""),
    ("cohesion", "offer_cuts", ""), ("headings", "offer_cuts", ""),
    ("q_topic", "objective", ""), ("offer_cuts", "objective", ""),
    ("objective", "q_cut", ""), ("offer_cuts", "q_cut", ""),
    ("offer_cuts", "reflex", "red"),
    ("q_cut", "cut", ""), ("reflex", "cut", "red"),
    ("cut", "CUT", ""), ("cut", "q_kind", ""), ("cut", "q_clean", ""),
    ("q_kind", "TAG", ""), ("q_clean", "CLEAN", ""),
]
PANEL_ORDER = [("topic", "TOPIC"), ("cut", "CUT"), ("kind", "KIND"), ("quality", "CLEAN")]

DEMO_NAME = "demo: The Harbor Light at Carrow Point"
DEMO = """# The Harbor Light at Carrow Point

Carrow Point has kept a light since 1791. The first keeper burned whale oil in a copper pan and climbed the tower eleven times a night to trim the wicks. Fishing boats from three villages learned to steer by it, and for most of two centuries nobody in the harbor could remember a winter without it.

Ines Varga took the post in the autumn of 1968, the year the old keeper's cottage lost its roof. She moved into the tower room with a camp bed, a radio and a crate of paperbacks. The harbor master gave her the logbook, a ring of brass keys and a warning that the stairs were steeper than they looked.

"Did you trim the wick?" Tomas asked from the doorway, shaking rain off his coat.

"Twice," said Ines. "The wind keeps eating it."

"Then we burn the spare tonight. The Merrow is still out past the reef."

"We burn the spare tonight," she agreed, and went to fetch the ladder.

## How the lamp works

The modern lamp is a 1000-watt metal-halide bulb set inside a second-order Fresnel lens. The lens is made of concentric glass prisms that bend stray light into a single horizontal beam, which is why a small bulb can be seen more than twenty nautical miles away. The whole assembly turns on a bath of mercury, so a push from one finger is enough to rotate two tonnes of glass. A clockwork motor, wound every four hours, keeps the rotation at exactly one turn every thirty seconds, giving Carrow Point its signature: two white flashes, then darkness.

## Nightly checklist

1. Check the fuel level in the backup generator.
2. Clean the lens panels with a dry cloth, never with solvent.
3. Wind the rotation clock at 20:00, 00:00 and 04:00.
4. Log wind speed and visibility at 20:00 and again at 02:00.
5. Test the fog signal for ten seconds if visibility drops below two miles.

## Logging readings

Since 2011 the readings go straight into a small script instead of the paper log:

```python
import time

def log_reading(path, wind_knots, visibility_nm):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.0f},{wind_knots},{visibility_nm}\\n")
```

The paper logbook still sits on the desk, though. Ines never trusted anything that could not survive a spilled cup of tea. On the last page of the 1968 volume she wrote a single line in pencil: the light was lit every night this year. Every keeper after her has added the same sentence at the end of each December, and so far no one has had to leave it out.
"""


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def rrect(x1, y1, x2, y2, r):
    return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]


class App:
    def __init__(self, args, doc):
        import tkinter as tk
        import tkinter.font as tkfont
        self.tk, self.args, self.doc = tk, args, doc
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on high-DPI screens
            except Exception:
                pass
        self.root = root = tk.Tk()
        root.title("TEXT SPLITTER // KEV")
        root.configure(bg=BG)
        s = args.zoom or root.winfo_fpixels("1i") / 96.0
        s = min(s, root.winfo_screenwidth() * 0.96 / W, root.winfo_screenheight() * 0.9 / H)
        self.S = max(0.5, s)
        fams = set(tkfont.families())
        self.family = next((f for f in ("Consolas", "Cascadia Mono", "JetBrains Mono", "DejaVu Sans Mono", "Menlo", "Courier New") if f in fams), "Courier")
        self.cv = tk.Canvas(root, width=W * self.S, height=H * self.S, bg=BG, highlightthickness=0)
        self.cv.pack()
        root.resizable(False, False)

        self.kev, self.events = None, queue.Queue()
        self.paused, self.busy, self.step_once = False, False, False
        self.move_delay, self.fire_gap = args.move_ms / 1000, args.fire_ms / 1000
        self.next_event, self.next_think = 0.0, 0.0
        self.answers, self.state, self.show_state = {}, "", False
        self.view, self.last_ms, self.status = None, None, "loading"
        self.anims, self.error, self.pending = [], None, None
        self.msg, self.msg_until = "", 0.0
        self.nodes, self.edges = {}, []

        self.draw_static()
        self.make_text()
        self.draw_graph()
        self.render_text()
        self.draw_status()
        self.draw_panel()
        for key, fn in {"<space>": self.toggle_pause, "n": self.step, "N": self.step, "r": self.restart, "R": self.restart,
                        "plus": self.faster, "equal": self.faster, "KP_Add": self.faster, "minus": self.slower, "KP_Subtract": self.slower,
                        "bracketleft": self.smaller, "bracketright": self.bigger,
                        "f": self.toggle_fire, "F": self.toggle_fire, "s": self.toggle_state, "S": self.toggle_state,
                        "o": self.open_file, "O": self.open_file, "v": self.paste, "V": self.paste, "e": self.export, "E": self.export,
                        "<Escape>": lambda e=None: root.destroy()}.items():
            root.bind(key if key.startswith("<") else f"<{key}>" if len(key) > 1 else key, fn)
        threading.Thread(target=self.load, daemon=True).start()
        self.root.after(16, self.pump)

    # ── helpers for scaled drawing
    def P(self, *v):
        return [x * self.S for x in v]

    def font(self, size, bold=False, italic=False):
        return (self.family, -max(6, round(size * self.S)), ("bold " if bold else "") + ("italic" if italic else "") or "normal")

    def text(self, x, y, s, size=11, color=TXT, anchor="nw", bold=False, tags=(), width=None):
        kw = {"width": width * self.S} if width else {}
        return self.cv.create_text(*self.P(x, y), text=s, fill=color, font=self.font(size, bold), anchor=anchor, tags=tags, **kw)

    def flash(self, msg, secs=5):
        self.msg, self.msg_until = msg, time.time() + secs
        self.draw_status()
        self.root.after(int(secs * 1000) + 50, self.draw_status)

    # ── model loading
    def load(self):
        try:
            if self.args.mock:
                self.kev = MockKev()
            else:
                self.kev = Kev(device=self.args.device, threads=self.args.threads,
                               log=lambda m: self.events.put(("status", m)))
            self.events.put(("loaded",))
        except Exception as e:
            traceback.print_exc()
            self.events.put(("error", f"could not load Kev: {e}"))

    # ── static chrome
    def draw_static(self):
        c, S = self.cv, self.S
        self.text(14, 9, "TEXT SPLITTER // KEV", 13, GREEN, bold=True)
        self.text(W - 14, 10, "[space] pause  [n] step  [r] restart  [+/-] speed  [ [ ] ] size  [f] fire  [s] situation  "
                              "[o] open  [v] paste  [e] export  [esc] quit", 9.5, TXT_DIM, anchor="ne")
        c.create_line(*self.P(12, 30, W - 12, 30), fill=GREEN_FAINT)
        c.create_rectangle(*self.P(TX1, TY1, TX2, TY2), outline=GREEN_DIM, width=S, fill=PANEL)
        c.create_rectangle(*self.P(DX1, 38, W - 12, 520), outline=GREEN_DIM, width=S, fill=PANEL)
        self.text(DX1 + 10, 43, "DECISIONS", 9, TXT_DIM)
        c.create_rectangle(*self.P(GX, GY, GX + GW, GY + GH), outline=GREEN_DIM, width=S, fill=PANEL)
        self.text(GX + 8, GY + 4, "GRAPH", 9, TXT_DIM)

    # ── text view
    def make_text(self):
        tk, S = self.tk, self.S
        t = self.txt = tk.Text(self.root, bg=PANEL, fg=TXT_DIM, font=self.font(10.5), wrap="word", bd=0,
                               highlightthickness=0, insertwidth=0, padx=round(8 * S), pady=round(6 * S),
                               spacing1=round(1 * S), spacing3=round(1 * S), cursor="arrow", takefocus=0,
                               selectbackground=GREEN_DIM, selectforeground="#ffffff")
        self.cv.create_window(*self.P(TX1 + 1, TY1 + 1), window=t, anchor="nw",
                              width=(TX2 - TX1 - 2) * S, height=(TY2 - TY1 - 2) * S)
        t.tag_configure("chA", foreground="#8cefaa", background="#0c1f12")
        t.tag_configure("chB", foreground="#67c585", background="#060f08")
        for i, col in enumerate(ALERT_COL):
            t.tag_configure(f"hdr{i}", foreground=BG, background=col, font=self.font(8.5, bold=True))
        t.tag_configure("window", foreground="#e6ffec", background="#10271a")
        t.tag_configure("rest", foreground=TXT_DIM)
        t.tag_configure("note", foreground=TXT_DIM, font=self.font(9.5, italic=True))
        t.tag_configure("marker", foreground=AMBER_TXT, background=AMBER_FILL, font=self.font(9.5, bold=True))
        t.tag_configure("pick", foreground=BG, background=AMBER, font=self.font(9.5, bold=True))

    def render_text(self):
        t, d = self.txt, self.doc
        text, chunks = d.text, d.chunks
        t.configure(state="normal")
        t.delete("1.0", "end")
        if not d.units:
            t.insert("end", "(empty document: press [o] to open a file or [v] to paste text)", "note")
            t.configure(state="disabled"); return
        first = 0                                   # very long documents: only render the neighbourhood
        if len(text) > 200_000:
            lim = d.start() - 20_000
            while first < len(chunks) and chunks[first]["end"] < lim: first += 1
            if first: t.insert("end", f"… {chunks[first]['start']} earlier characters, {first} chunks …\n\n", "note")
        focus = pick_idx = None
        pos = chunks[first]["start"] if first < len(chunks) else d.start()
        for ch in chunks[first:]:
            t.insert("end", text[pos:ch["start"]])
            q = ch.get("quality_idx", 0)
            t.insert("end", f" {ch['id'] + 1} · {ch.get('kind', '')} · {ch.get('quality', '')} ", f"hdr{q}")
            t.insert("end", " ")
            t.insert("end", text[ch["start"]:ch["end"]], "chA" if ch["id"] % 2 == 0 else "chB")
            pos = ch["end"]
        if not d.done:
            s = d.start()
            t.insert("end", text[pos:s])
            focus = t.index("end-1c")
            v = self.view
            if v and v["u0"] == d.cursor:
                p = s
                for i, j in enumerate(v["cands"], 1):
                    e = d.units[j - 1]["end"]
                    t.insert("end", text[p:e], "window")
                    if v.get("pick") == j:
                        pick_idx = t.index("end-1c")
                        t.insert("end", f" ✂ {i} ", "pick")
                    else:
                        t.insert("end", f" ⟨{i}⟩ ", "marker")
                    p = e
                t.insert("end", text[p:v["win_end"]], "window")
                t.insert("end", text[v["win_end"]:], "rest")
            else:
                t.insert("end", text[s:], "rest")
        else:
            t.insert("end", text[pos:])
        t.configure(state="disabled")
        if focus:
            t.yview(f"{focus} - 1 lines linestart")
            if pick_idx: t.see(pick_idx)
            elif self.view: t.see(t.index(f"{focus} + {self.view['win_end'] - d.start()} chars"))
            t.see(focus)

    # ── graph
    def node_box(self, nid):
        kind, x, y, w, _ = NODES[nid]
        h = 40 if kind == "comb" else NODE_H
        return GX + x - w / 2, GY + y - h / 2, GX + x + w / 2, GY + y + h / 2

    def draw_graph(self):
        c, S = self.cv, self.S
        incoming = collections.defaultdict(list)
        for src, dst, kind in EDGES: incoming[dst].append(src)
        for dst in incoming: incoming[dst].sort(key=lambda s: NODES[s][2])
        for src, dst, kind in EDGES:
            x1, y1a, x1b, y1b = self.node_box(src)
            x2a, y2a, x2b, y2b = self.node_box(dst)
            sx, sy = x1b, (y1a + y1b) / 2
            dx, dy = x2a, (y2a + y2b) / 2
            k = incoming[dst].index(src)
            xm = dx - 9 - 6 * k
            pts = [sx, sy, dx, dy] if abs(sy - dy) < 1 else [sx, sy, xm, sy, xm, dy, dx, dy]
            halo = c.create_line(*self.P(*pts), fill=BG, width=6 * S, joinstyle="round", capstyle="round")
            core = c.create_line(*self.P(*pts), fill=GREEN_FAINT, width=S, joinstyle="round",
                                 dash=(4, 3) if kind == "red" else ())
            self.edges.append({"src": src, "dst": dst, "kind": kind, "pts": pts, "halo": halo, "core": core, "lit": False})
        for nid, (kind, x, y, w, label) in NODES.items():
            x1, y1, x2, y2 = self.node_box(nid)
            if kind == "comb":
                halo = c.create_oval(*self.P(x1 - 3, y1 - 3, x2 + 3, y2 + 3), outline=BG, width=4 * S)
                body = c.create_oval(*self.P(x1, y1, x2, y2), outline=GREEN_DIM, fill=PANEL, width=S)
            else:
                r = NODE_H / 2 if kind == "offer" else 5
                halo = c.create_polygon(*self.P(*rrect(x1 - 3, y1 - 3, x2 + 3, y2 + 3, r + 2)), smooth=True, outline=BG, fill="", width=4 * S)
                body = c.create_polygon(*self.P(*rrect(x1, y1, x2, y2, r)), smooth=True, outline=GREEN_DIM, fill=PANEL, width=S,
                                        dash=(3, 3) if kind in ("input", "offer", "reflex") else ())
            t1 = self.text(GX + x, GY + y - 7, label, 9.5, TXT_DIM, anchor="center", bold=kind == "out")
            t2 = self.text(GX + x, GY + y + 7, "", 9.5, TXT_DIM, anchor="center", bold=kind == "out")
            self.nodes[nid] = {"kind": kind, "halo": halo, "body": body, "t1": t1, "t2": t2, "lit": False, "flash": 0.0,
                               "label": label, "w": w, "color": None}
        self.paint_all()

    def node_colors(self, n, now):
        kind, lit = n["kind"], n["lit"]
        f = max(0.0, 1 - (now - n["flash"]) / 0.55) if lit else 0.0
        if kind == "q":
            base = (AMBER, AMBER_FILL, AMBER_TXT) if lit else (AMBER_DIM, AMBER_FILL_DIM, AMBER_DIM)
        elif kind == "reflex":
            base = (RED, RED_DIM, "#ffb0b8") if lit else (RED_DIM, PANEL, RED_DIM)
        else:
            col = n.get("color") or GREEN
            base = (col, mix(PANEL, col, 0.12 if kind != "out" else 0.18), mix(col, "#ffffff", 0.45)) if lit else (GREEN_DIM, PANEL, TXT_DIM)
        outline, fill, txt = base
        if f:
            outline, fill, txt = mix(outline, "#ffffff", 0.7 * f), mix(fill, outline, 0.5 * f), mix(txt, "#ffffff", 0.8 * f)
        halo = mix(BG, outline, (0.28 + 0.5 * f)) if lit else BG
        return outline, fill, txt, halo, f

    def paint_node(self, nid, now=None):
        n, c = self.nodes[nid], self.cv
        outline, fill, txt, halo, f = self.node_colors(n, now or time.time())
        c.itemconfigure(n["body"], outline=outline, fill=fill, width=(2.2 if n["lit"] else 1) * self.S)
        c.itemconfigure(n["halo"], outline=halo)
        c.itemconfigure(n["t1"], fill=txt)
        c.itemconfigure(n["t2"], fill=txt)
        return f > 0

    def paint_edge(self, e):
        red = e["kind"] == "red"
        if e["lit"]:
            col = RED if red else GREEN
            self.cv.itemconfigure(e["core"], fill=col, width=(1.6 if e["kind"] == "thin" else 2.4) * self.S)
            self.cv.itemconfigure(e["halo"], fill=mix(BG, col, 0.22))
        else:
            self.cv.itemconfigure(e["core"], fill=RED_DIM if red else EDGE_DIM, width=self.S)
            self.cv.itemconfigure(e["halo"], fill=BG)

    def paint_all(self):
        for nid in self.nodes: self.paint_node(nid)
        for e in self.edges: self.paint_edge(e)

    def reset_graph(self):
        for n in self.nodes.values():
            n["lit"], n["color"] = False, None
            self.cv.itemconfigure(n["t2"], text="")
        for e in self.edges: e["lit"] = False
        self.paint_all()

    def fire(self, nid, value, sources, color_idx=None):
        n, now = self.nodes[nid], time.time()
        n["lit"], n["flash"] = True, now
        if color_idx is not None: n["color"] = ALERT_COL[color_idx]
        width = int(n["w"] / (6.2 if n["kind"] != "out" else 6.8))
        v = str(value)
        v = v if len(v) <= width - 2 else v[: width - 3] + "…"
        self.cv.itemconfigure(n["t2"], text=f"▸ {v}")
        self.paint_node(nid, now)
        for e in self.edges:
            if e["dst"] != nid: continue
            if (sources is None and self.nodes[e["src"]]["lit"]) or (sources and e["src"] in sources):
                e["lit"] = True
                self.paint_edge(e)
                self.cv.tag_raise(e["halo"]); self.cv.tag_raise(e["core"])
                self.add_pulse(e)
        for k in ("halo", "body", "t1", "t2"):
            self.cv.tag_raise(n[k])

    def add_pulse(self, e):
        col = RED if e["kind"] == "red" else "#d9ffe4"
        dot = self.cv.create_oval(0, 0, 0, 0, fill=col, outline="")
        glow = self.cv.create_oval(0, 0, 0, 0, fill=mix(BG, GREEN if e["kind"] != "red" else RED, 0.45), outline="")
        pts = list(zip(e["pts"][::2], e["pts"][1::2]))
        segs = [(a, b, math.dist(a, b)) for a, b in zip(pts, pts[1:])]
        dur = max(0.18, min(0.45, sum(s[2] for s in segs) / 900)) * (self.fire_gap / 0.11 if self.fire_gap < 0.11 else 1)
        self.anims.append({"dot": dot, "glow": glow, "segs": segs, "t0": time.time(), "dur": dur})

    def animate(self):
        now, S, keep = time.time(), self.S, []
        for a in self.anims:
            u = (now - a["t0"]) / a["dur"]
            if u >= 1:
                self.cv.delete(a["dot"]); self.cv.delete(a["glow"]); continue
            total = sum(s[2] for s in a["segs"]) or 1
            d = u * total
            x, y = a["segs"][-1][1]
            for (x1, y1), (x2, y2), L in a["segs"]:
                if d <= L or L == 0:
                    t = d / L if L else 1
                    x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                    break
                d -= L
            self.cv.coords(a["glow"], (x - 6) * S, (y - 6) * S, (x + 6) * S, (y + 6) * S)
            self.cv.coords(a["dot"], (x - 3) * S, (y - 3) * S, (x + 3) * S, (y + 3) * S)
            self.cv.tag_raise(a["glow"]); self.cv.tag_raise(a["dot"])
            keep.append(a)
        self.anims = keep
        for nid, n in self.nodes.items():
            if n["lit"] and now - n["flash"] < 0.6: self.paint_node(nid, now)

    # ── status under the text
    def draw_status(self):
        self.cv.delete("status")
        d = self.doc
        sizes = [c["chars"] for c in d.chunks]
        pct = 100 * (d.start() if d.units else 0) / max(1, len(d.text)) if not d.done else 100
        avg = f"{sum(sizes) / len(sizes):.0f}" if sizes else "-"
        rng = f"{min(sizes)}-{max(sizes)}" if sizes else "-"
        self.text(14, 499, f"CHUNKS {len(sizes):<3} DONE {pct:3.0f}%  AVG {avg:<5} RANGE {rng:<9} BUDGET {d.min_chars}-{d.max_chars}",
                  10.5, GREEN, tags="status")
        if time.time() < self.msg_until:
            self.text(14, 512, self.msg, 9.5, AMBER, tags="status"); return
        speed = f"{self.last_ms / 1000:.2f} s/chunk" if self.last_ms else "…"
        mode = "DONE · [e] export  [r] again" if d.done else "PAUSED" if self.paused else "AUTO"
        name = self.kev.name if self.kev else "kev-0.8b"
        self.text(14, 512, f"{mode} · {name} · {speed} · {d.name[:48]}", 9.5, AMBER if (self.paused or d.done) else TXT_DIM, tags="status")

    # ── decisions panel
    def draw_panel(self):
        c = self.cv
        c.delete("panel")
        x0, y, x1 = DX1 + 10, 60, W - 22
        if self.error:
            self.text(x0, y, self.error, 11, RED, tags="panel", width=x1 - x0)
            return
        if self.kev is None:
            self.text(x0, y, "LOADING KEV-0.8B", 14, GREEN, bold=True, tags="panel")
            self.text(x0, y + 26, self.status, 11, TXT, tags="panel", width=x1 - x0)
            self.text(x0, y + 60, "The first run downloads the adapter and the Qwen3.5-0.8B base (~1.7 GB).\n"
                                  "Progress is printed in the terminal.", 10, TXT_DIM, tags="panel", width=x1 - x0)
            return
        if self.show_state:
            self.text(x0, y, "SITUATION REPORT  (the exact state text Kev reads)", 9.5, AMBER, tags="panel")
            st = self.state or "…"
            if len(st) > 2600: st = st[:2600] + " …"
            self.text(x0, y + 16, st, 8.5, TXT, tags="panel", width=x1 - x0)
            return
        bar_x, bar_w = x0 + 118, 140
        for qid, tag in PANEL_ORDER:
            a = self.answers.get(qid)
            c.create_rectangle(*self.P(x0, y + 1, x0 + 50, y + 14), outline=AMBER if a else AMBER_DIM, tags="panel")
            self.text(x0 + 25, y + 7.5, tag, 8.5, AMBER if a else AMBER_DIM, anchor="center", bold=True, tags="panel")
            if not a:
                self.text(x0 + 58, y + 1, "thinking…" if self.busy else "", 9.5, TXT_DIM, tags="panel")
                y += 34
                continue
            meta = f"{a['ms']:.0f} ms" if a["ms"] else a.get("note", "")
            if "score" in a: meta = f"score {a['score']:.2f} · " + meta
            it = self.text(x0 + 58, y + 1, a["instr"], 9, TXT, tags="panel", width=x1 - x0 - 58 - (110 if a["ms"] else 0))
            if a["ms"]:
                self.text(x1, y + 1, meta, 8.5, TXT_DIM, anchor="ne", tags="panel")
            elif a.get("note"):
                bb = c.bbox(it); y = bb[3] / self.S + 1
                it = self.text(x0 + 58, y, a["note"], 8.5, TXT_DIM, tags="panel", width=x1 - x0 - 58)
            bb = c.bbox(it)
            y = max(y + 17, bb[3] / self.S + 4)
            for i, (lab, desc, p) in enumerate(zip(a["labels"], a["descs"], a["probs"])):
                pick = i == a["pick"]
                col = GREEN if pick else TXT_DIM
                self.text(x0 + 6, y, ("▸ " if pick else "  ") + lab[:15], 9, col, bold=pick, tags="panel")
                c.create_rectangle(*self.P(bar_x, y + 3, bar_x + bar_w, y + 10), fill=GREEN_FAINT, outline="", tags="panel")
                c.create_rectangle(*self.P(bar_x, y + 3, bar_x + max(1, bar_w * p), y + 10), fill=GREEN if pick else "#2c7a44", outline="", tags="panel")
                self.text(bar_x + bar_w + 6, y, f"{p * 100:5.1f}%", 9, col, tags="panel")
                room = int((x1 - (bar_x + bar_w + 56)) / (8.5 * 0.6))
                d = desc if len(desc) <= room else desc[:room - 1] + "…"
                self.text(bar_x + bar_w + 56, y + 0.5, d, 8.5, TXT_DIM if not pick else TXT, tags="panel")
                y += 14
            y += 8

    # ── controls
    def toggle_pause(self, _=None):
        self.paused = not self.paused; self.draw_status()

    def step(self, _=None):
        self.paused = True; self.step_once = True; self.draw_status()

    def later(self, fn):
        """Run fn now, or once the current chunk finishes (the worker thread reads the document)."""
        if self.busy: self.pending = fn
        else: fn()

    def restart(self, _=None):
        def go():
            self.doc.reset(); self.view = None; self.answers = {}
            self.reset_graph(); self.render_text(); self.draw_status(); self.draw_panel()
            self.next_think = time.time() + 0.3
        self.later(go)

    def set_doc(self, doc):
        def go():
            self.doc = doc; self.paused = False
            self.restart()
            self.flash(f"loaded {doc.name}: {len(doc.text)} chars, {len(doc.units)} sentences/blocks")
        self.later(go)

    def resize(self, delta):
        def go():
            t = max(150, min(6000, self.doc.target + delta))
            self.doc.set_target(t); self.restart()
            self.flash(f"target {t} chars → budget {self.doc.min_chars}-{self.doc.max_chars} (restarted)")
        self.later(go)

    def smaller(self, _=None): self.resize(-100)
    def bigger(self, _=None): self.resize(+100)

    def faster(self, _=None):
        self.move_delay = max(0.0, self.move_delay - 0.1)

    def slower(self, _=None):
        self.move_delay = min(3.0, self.move_delay + 0.1)

    def toggle_fire(self, _=None):
        self.fire_gap = {0.11: 0.35, 0.35: 0.02}.get(round(self.fire_gap, 2), 0.11)

    def toggle_state(self, _=None):
        self.show_state = not self.show_state; self.draw_panel()

    def open_file(self, _=None):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Open a text document",
                                          filetypes=[("Text", "*.txt *.md *.markdown *.rst *.html *.csv *.py"), ("All files", "*.*")])
        if not path: return
        try:
            self.set_doc(Doc(load_text(path), os.path.basename(path), self.doc.target))
        except Exception as e:
            self.flash(f"could not open: {e}")

    def paste(self, _=None):
        try:
            txt = self.root.clipboard_get()
        except Exception:
            self.flash("the clipboard has no text"); return
        if not txt.strip():
            self.flash("the clipboard is empty"); return
        self.set_doc(Doc(txt, "clipboard", self.doc.target))

    def export(self, _=None):
        if not self.doc.chunks:
            self.flash("nothing to export yet"); return
        from tkinter import filedialog
        stem = re.sub(r"[^\w-]+", "_", os.path.splitext(self.doc.name)[0])[:40].strip("_") or "document"
        path = filedialog.asksaveasfilename(title="Export chunks", initialfile=f"{stem}_chunks.jsonl", defaultextension=".jsonl",
                                            filetypes=[("JSON lines", "*.jsonl"), ("JSON", "*.json")])
        if not path: return
        try:
            k = self.doc.export(path)
            self.flash(f"saved {k} chunks{'' if self.doc.done else ' (so far)'} → {path}")
        except Exception as e:
            self.flash(f"could not save: {e}")

    # ── main loop
    def start_think(self):
        self.busy, self.answers, self.view = True, {}, None
        self.reset_graph(); self.draw_panel()
        snapshot = copy.deepcopy(self.doc)
        emit = lambda *e: self.events.put(e)

        def work():
            try:
                think(snapshot, self.kev, emit, self.args.mock)
            except Exception as e:
                traceback.print_exc()
                emit("error", f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def handle(self, ev):
        """Apply one worker event. Returns the pause before the next event (so fast GPUs still show the firing)."""
        kind = ev[0]
        if kind == "status":
            self.status = ev[1]; self.draw_panel(); return 0
        if kind == "loaded":
            self.draw_panel(); self.draw_status(); return 0
        if kind == "error":
            self.error, self.busy = ev[1], False; self.draw_panel(); return 0
        if kind == "state":
            self.state = ev[1]
            if self.show_state: self.draw_panel()
            return 0
        if kind == "view":
            self.view = ev[1]; self.render_text(); return self.fire_gap
        if kind == "pick":
            if self.view: self.view["pick"] = ev[1]; self.render_text()
            return 0
        if kind == "answer":
            self.answers[ev[1]] = ev[2]; self.draw_panel(); return 0
        if kind == "fire":
            nid, value, sources = ev[1], ev[2], ev[3]
            self.fire(nid, value, sources, ev[4] if len(ev) > 4 else None)
            return self.fire_gap * (0.3 if NODES[nid][0] == "input" else 1)
        if kind == "done":
            self.last_ms = ev[2]["ms"]
            self.doc.commit(ev[1], ev[2])
            self.view, self.busy = None, False
            if self.pending:
                fn, self.pending = self.pending, None; fn()
            self.render_text(); self.draw_status()
            self.next_think = time.time() + self.move_delay
            return 0
        return 0

    def pump(self):
        now = time.time()
        while now >= self.next_event:
            try: ev = self.events.get_nowait()
            except queue.Empty: break
            gap = self.handle(ev)
            if gap: self.next_event = now + gap; break
        if (self.kev and not self.busy and not self.error and not self.doc.done and now >= self.next_think
                and self.events.empty() and (not self.paused or self.step_once)):
            self.step_once = False
            self.start_think()
        self.animate()
        self.root.after(16, self.pump)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Kev-0.8B splits a document into retrieval chunks, with its decision graph firing live.")
    ap.add_argument("file", nargs="?", help="text / markdown file to split ('-' = stdin; default: built-in demo)")
    ap.add_argument("--target", type=int, default=500, help="target chunk size in characters (budget 0.5x-1.5x, default 500)")
    ap.add_argument("--mock", action="store_true", help="fake heuristic model (preview the UI without downloading Kev)")
    ap.add_argument("--headless", action="store_true", help="no window: split the whole document and print the decisions")
    ap.add_argument("--out", help="with --headless: save chunks to .jsonl / .json")
    ap.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--seed", type=int, help="random seed (mock noise)")
    ap.add_argument("--move-ms", type=int, default=400, help="pause after each chunk (default 400)")
    ap.add_argument("--fire-ms", type=int, default=110, help="minimum time between graph nodes firing (default 110)")
    ap.add_argument("--zoom", type=float, help="UI scale (default: follow screen DPI)")
    args = ap.parse_args()
    if args.seed is not None: random.seed(args.seed)
    if args.file:
        doc = Doc(load_text(args.file), "stdin" if args.file == "-" else os.path.basename(args.file), args.target)
    else:
        doc = Doc(DEMO, DEMO_NAME, args.target)
    if args.headless:
        kev = MockKev() if args.mock else Kev(device=args.device, threads=args.threads)
        run_headless(kev, doc, args.mock, args.out)
        return
    App(args, doc).run()


if __name__ == "__main__":
    main()
