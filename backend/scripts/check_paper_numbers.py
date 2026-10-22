#!/usr/bin/env python3
"""Number checker for INDICON 850 (plan E7, gate G1).

Maps every headline number in the revised paper to a released evidence file
(`INDICON2026_submit/paper_claims.yaml`) and fails on mismatch or on a
deployment number outside the deployment study table.

Usage (repository root):
    python backend/scripts/check_paper_numbers.py
    python backend/scripts/check_paper_numbers.py --tex path/to.tex --claims path/to.yaml

Exit status: 0 when every check passes, 1 otherwise.
Stdlib only (no PyYAML): the claims file uses a minimal YAML subset
(maps, lists, inline lists, quoted scalars, `#` comments) parsed below.
"""

import json
import os
import re
import sys


# --------------------------------------------------------------------------
# Minimal YAML-subset parser (only what paper_claims.yaml uses)
# --------------------------------------------------------------------------

def _strip_comment(line):
    out = []
    in_s = False
    in_d = False
    for ch in line:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _split_inline(text):
    """Split comma-separated inline content, respecting double quotes."""
    parts, cur = [], []
    in_d = False
    for ch in text:
        if ch == '"':
            in_d = not in_d
            cur.append(ch)
        elif ch == "," and not in_d:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p != ""]


def _scalar(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1]
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?", text):
        return float(text)
    return text


def _parse_inline_list(text):
    return [_scalar(p) for p in _split_inline(text[1:-1])]


def _parse_value(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return _parse_inline_list(text)
    return _scalar(text)


def parse_minimal_yaml(path):
    """Parse the restricted YAML subset used by paper_claims.yaml.

    Supports: `key: value` maps, `- ` list items, nested blocks by
    indentation, inline lists `[a, b, "c d"]`, quoted scalars and `#`
    comments. Anything else raises ValueError.
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().splitlines()
    return _rebuild(raw, path)


def _rebuild(raw, path):
    """Indentation-driven rebuild supporting pending (list-or-map) blocks."""
    root = {}
    stack = [(-1, root)]  # (indent, container)
    pending = []  # (key_indent, parent_dict, key)

    for lineno, raw_line in enumerate(raw, 1):
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError("bad indent at %s:%d" % (path, lineno))
        parent = stack[-1][1]
        if content.startswith("- "):
            if isinstance(parent, dict):
                # attach a new list to the pending key of this parent
                for i in range(len(pending) - 1, -1, -1):
                    if pending[i][1] is parent and pending[i][0] < indent:
                        key_indent, pmap, pkey = pending.pop(i)
                        pmap[pkey] = []
                        stack.append((key_indent, pmap[pkey]))
                        parent = pmap[pkey]
                        break
                else:
                    raise ValueError("list item without parent at %s:%d"
                                     % (path, lineno))
            item = content[2:].strip()
            if item == "":
                parent.append({})
                stack.append((indent, parent[-1]))
            elif ": " in item and not item.startswith("["):
                key, _, val = item.partition(":")
                entry = {}
                parent.append(entry)
                stack.append((indent, entry))
                if val.strip() != "":
                    entry[key.strip()] = _parse_value(val)
            else:
                parent.append(_parse_value(item))
        elif ":" in content:
            key, _, val = content.partition(":")
            key, val = key.strip(), val.strip()
            if not isinstance(parent, dict):
                raise ValueError("map entry in list at %s:%d" % (path, lineno))
            # attach to the innermost pending block of this parent, if any
            for i in range(len(pending) - 1, -1, -1):
                if pending[i][1] is parent and pending[i][0] < indent:
                    key_indent, pmap, pkey = pending.pop(i)
                    pmap[pkey] = {}
                    stack.append((key_indent, pmap[pkey]))
                    parent = pmap[pkey]
                    break
            if val == "":
                pending.append((indent, parent, key))
                parent[key] = {}  # placeholder; replaced if a list follows
            else:
                parent[key] = _parse_value(val)
        else:
            raise ValueError("unparsable line at %s:%d" % (path, lineno))
    # any leftover pending keys stay as {} (empty maps)
    return root


# --------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------

def load_json_pointer(doc, pointer):
    """Dotted path with integer indices; or a list of keys.

    Dots inside key names (e.g. `fixed_0.7`) are handled by longest-match:
    at each level the longest leading segment that exists is consumed.
    """
    if isinstance(pointer, list):
        node = doc
        for key in pointer:
            if isinstance(node, list):
                node = node[int(key)]
            elif isinstance(node, dict) and key in node:
                node = node[key]
            else:
                raise KeyError("path %r missing key %r" % (pointer, key))
        return node
    parts = str(pointer).split(".")
    node, i = doc, 0
    while i < len(parts):
        if isinstance(node, list):
            node = node[int(parts[i])]
            i += 1
            continue
        if not isinstance(node, dict):
            raise KeyError("path %r dead-ends at %r" % (pointer, parts[i]))
        for j in range(len(parts), i, -1):
            candidate = ".".join(parts[i:j])
            if candidate in node:
                node = node[candidate]
                i = j
                break
        else:
            raise KeyError("path %r missing key %r" % (pointer, parts[i]))
    return node


def strip_tex_comments(text):
    out_lines = []
    for line in text.splitlines():
        buf, i = [], 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(line[i])
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def normalize_tex(text):
    text = text.replace("\\%", "PERCENT")
    text = text.replace("{,}", ",")
    text = text.replace("$", "")
    text = text.replace("\\,", "")
    text = text.replace("\\allowbreak", "")
    text = text.replace("~", " ")
    text = text.replace("PERCENT", "%")
    return text


def find_spans(lines, start_re, end_re):
    spans = []
    start = None
    for i, line in enumerate(lines):
        if start is None and re.search(start_re, line):
            start = i
        elif start is not None and re.search(end_re, line):
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(lines) - 1))
    return spans


def table_spans(lines):
    spans = []
    start = None
    for i, line in enumerate(lines):
        if r"\begin{table" in line:
            start = i
        elif r"\end{table" in line and start is not None:
            spans.append((start, i))
            start = None
    return spans


def in_spans(idx, spans):
    return any(s <= idx <= e for s, e in spans)


def lit_regex(literal):
    # strict token boundaries so "109" does not match "1090"; a trailing
    # sentence period is allowed (only a trailing *digit* is rejected).
    return re.compile(r"(?<![\d.,])" + re.escape(literal) + r"(?!\d)")


def main(argv):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(script_dir))
    tex_path = os.path.join(repo, "INDICON2026_submit", "INDICON2026Conference.tex")
    claims_path = os.path.join(repo, "INDICON2026_submit", "paper_claims.yaml")
    for i, arg in enumerate(argv):
        if arg == "--tex" and i + 1 < len(argv):
            tex_path = argv[i + 1]
        if arg == "--claims" and i + 1 < len(argv):
            claims_path = argv[i + 1]

    failures = []
    passes = []

    try:
        claims_doc = parse_minimal_yaml(claims_path)
    except Exception as exc:  # noqa: BLE001
        print("FAIL: cannot parse claims file: %s" % exc)
        return 1

    try:
        with open(tex_path, encoding="utf-8") as fh:
            tex_raw = fh.read()
    except OSError as exc:
        print("FAIL: cannot read tex: %s" % exc)
        return 1

    tex_nocomment = strip_tex_comments(tex_raw)
    lines = tex_nocomment.splitlines()
    search_text = normalize_tex(tex_nocomment)
    search_lines = search_text.splitlines()

    # Regions -------------------------------------------------------------
    abstract = find_spans(lines, r"\\begin\{abstract\}", r"\\end\{abstract\}")
    contrib = find_spans(lines, r"\\subsection\*\{Contributions\}", r"\\section\{")
    concl = find_spans(lines, r"\\section\{Discussion and Conclusion\}",
                        r"\\bibliographystyle|\\end\{document\}")
    tables = table_spans(lines)
    deploy_tables = [sp for sp in tables
                     if "deployment" in normalize_tex(
                         "\n".join(lines[sp[0]:sp[1] + 1])).lower()]

    def region_of(idx):
        tags = []
        if in_spans(idx, abstract):
            tags.append("abstract")
        if in_spans(idx, contrib):
            tags.append("contributions")
        if in_spans(idx, concl):
            tags.append("conclusion")
        if in_spans(idx, deploy_tables):
            tags.append("deployment-table")
        return tags

    banned_spans = ("abstract", "contributions", "conclusion")
    whitelist = claims_doc.get("whitelist", [])

    # Released claims ------------------------------------------------------
    json_cache = {}

    def get_json(rel):
        if rel not in json_cache:
            with open(os.path.join(repo, rel), encoding="utf-8") as fh:
                json_cache[rel] = json.load(fh)
        return json_cache[rel]

    for claim in claims_doc.get("claims", []):
        cid = claim.get("id", "?")
        kind = claim.get("kind", "value")
        src = claim.get("source", {})
        tex_lits = claim.get("tex", [])
        if not isinstance(tex_lits, list):
            tex_lits = [tex_lits]
        tex_lits = [str(t) for t in tex_lits]
        try:
            if kind == "fraction":
                num = int(load_json_pointer(get_json(src["file"]), src["num"]))
                den = int(load_json_pointer(get_json(src["file"]), src["den"]))
                expected = "%d/%d" % (num, den)
                found = any(lit == expected for lit in tex_lits
                            if lit_regex(lit).search(search_text))
                if found:
                    passes.append("%s: tex %s matches %s %s/%s" %
                                  (cid, expected, src["file"], num, den))
                else:
                    failures.append(
                        "%s: MISMATCH/MISSING tex %s not found; evidence is "
                        "%s num=%s den=%s" % (cid, tex_lits, src["file"], num, den))
            elif kind == "ci":
                rnd = int(src.get("round", 3))
                lo = float(load_json_pointer(get_json(src["file"]), src["lo"]))
                hi = float(load_json_pointer(get_json(src["file"]), src["hi"]))
                lo_s = ("%.*f" % (rnd, lo))
                hi_s = ("%.*f" % (rnd, hi))
                ok = any(lo_s in sl and hi_s in sl for sl in search_lines)
                if ok:
                    passes.append("%s: CI [%s, %s] on one line; matches %s" %
                                  (cid, lo_s, hi_s, src["file"]))
                else:
                    failures.append(
                        "%s: MISMATCH/MISSING CI endpoints %s/%s not on one "
                        "tex line; evidence %s lo=%s hi=%s"
                        % (cid, lo_s, hi_s, src["file"], lo, hi))
            else:  # kind == value: numeric-token scan (trailing-zero tolerant)
                rnd = int(src.get("round", 3))
                tol = 0.5 * 10 ** (-rnd) + 1e-9
                actual = float(load_json_pointer(get_json(src["file"]),
                                                 src["path"]))
                hit_lines = []
                seen = set()
                for m in re.finditer(r"(?<![\d.])([+-]?\d+(?:\.\d+)?)(?!\d)",
                                     search_text):
                    try:
                        if abs(float(m.group(1)) - actual) <= tol:
                            ln = search_text.count("\n", 0, m.start()) + 1
                            if ln not in seen:
                                seen.add(ln)
                                hit_lines.append(ln)
                    except ValueError:
                        pass
                if hit_lines:
                    passes.append(
                        "%s: tex %s matches %s %s=%s (lines %s)" %
                        (cid, tex_lits, src["file"], src["path"], actual,
                         sorted(set(hit_lines))[:5]))
                else:
                    failures.append(
                        "%s: MISMATCH/MISSING tex %s absent or != evidence "
                        "%s %s=%s (tol %s)"
                        % (cid, tex_lits, src["file"], src.get("path"), actual,
                           tol))
        except (KeyError, OSError, ValueError, TypeError) as exc:
            failures.append("%s: EVIDENCE ERROR %s" % (cid, exc))

    # Deployment numbers: only inside the deployment study table ------------
    seen_dep = set()
    for dep in claims_doc.get("deployment", []):
        did = dep.get("id", "?")
        lits = dep.get("tex", [])
        if not isinstance(lits, list):
            lits = [lits]
        for lit in lits:
            lit = str(lit)
            for m in lit_regex(lit).finditer(search_text):
                ln = search_text.count("\n", 0, m.start()) + 1
                if (did, lit, ln) in seen_dep:
                    continue
                seen_dep.add((did, lit, ln))
                ctx = search_text[max(0, m.start() - 8):m.end() + 8]
                if any(re.search(pat, ctx) for pat in whitelist):
                    continue
                tags = region_of(ln - 1)
                if "deployment-table" in tags:
                    passes.append("%s: %s on line %d inside deployment table"
                                  % (did, lit, ln))
                else:
                    where = "+".join(tags) if tags else "body-text"
                    failures.append(
                        "%s: DEPLOYMENT NUMBER %s on line %d outside the "
                        "deployment table (%s); merge must move it "
                        "into the deployment study table%s"
                        % (did, lit, ln, where,
                           "; also banned in abstract/contributions/conclusion"
                           if any(t in banned_spans for t in tags) else ""))

    # G0: banned tokens ------------------------------------------------------
    banned_tokens = (["verified", "Person Re-ID", "Multimodal Person Re-ID",
                      "10.7", "2.2", "0.822", "0.67", "0.83"]
                     + [str(t) for dep in claims_doc.get("deployment", [])
                        for t in (dep.get("tex", [])
                                  if isinstance(dep.get("tex", []), list)
                                  else [dep.get("tex", [])])])
    for span_name, spans in (("abstract", abstract),
                             ("contributions", contrib),
                             ("conclusion", concl)):
        body = normalize_tex("\n".join("\n".join(lines[s:e + 1])
                                       for s, e in spans))
        for tok in banned_tokens:
            if tok in ("6.0", "12.0", "109", "312,544"):
                continue  # latency/scale context, checked by placement rule
            if lit_regex(tok).search(body):
                failures.append(
                    "G0: banned token %r inside %s; merge must remove it" %
                    (tok, span_name))
    # keywords line: no Person Re-ID
    for i, line in enumerate(lines, 1):
        if "IEEEkeywords" in line:
            for j in range(i - 1, min(len(lines), i + 2)):
                if re.search(r"Person Re-?ID", lines[j]):
                    failures.append(
                        "G0: 'Person Re-ID' in keywords (line %d); must become "
                        "'Person Descriptor Retrieval'" % (j + 1))

    # Parser-gap rule: at most two gap sentences ------------------------------
    gap_hits = []
    for m in re.finditer(r"[^.]*pars\w*[^.]{0,120}gap[^.]*\.[^.]*|"
                         r"[^.]*gap[^.]{0,120}pars\w*[^.]*\.[^.]*|"
                         r"oracle-filter[^.]*\.[^.]*|"
                         r"parsed windows differ[^.]*\.[^.]*",
                         tex_nocomment, flags=re.IGNORECASE):
        sent = " ".join(m.group(0).split())
        if "NL parser" in sent and "gap" not in sent.lower():
            continue
        gap_hits.append(sent[:160])
    if len(gap_hits) > 2:
        failures.append(
            "PARSER-GAP: %d gap sentences found (max 2 allowed: one in "
            "Evaluation, one in future scope): %s"
            % (len(gap_hits), " || ".join(gap_hits[:5])))

    print("check_paper_numbers: %d passed, %d failed" % (len(passes), len(failures)))
    for line in passes:
        print("  PASS " + line)
    for line in failures:
        print("  FAIL " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))