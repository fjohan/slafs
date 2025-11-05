#!/usr/bin/env python3
import argparse
import io
import sys
import zipfile
import gzip
import unicodedata as ud
from collections import defaultdict, deque
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set

# ---------- small utils ----------

def normalize_key(s: str) -> str:
    """Normalize a lemgram or token key for consistent matching."""
    if not s:
        return ""
    s = s.strip()
    s = ud.normalize("NFC", s)

    # unify dash variants to '-'
    s = (s.replace("\u2010", "-")
           .replace("\u2011", "-")
           .replace("\u2012", "-")
           .replace("\u2013", "-")
           .replace("\u2014", "-"))

    # ---- handle pipe-wrapped compound lemgrams ----
    if "|" in s:
        # split on pipes and keep last non-empty piece
        parts = [p.strip() for p in s.split("|") if p.strip()]
        if parts:
            s = parts[-1]  # usually "förslag..nn.1"
    # ----------------------------------------------

    return s

def open_maybe_compressed(path: str):
    """Open text file possibly compressed as .gz or a .zip containing a single file."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        # pick the first non-directory file
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError("ZIP has no files.")
        return io.TextIOWrapper(zf.open(names[0], "r"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def get_feat(fr: ET.Element, att: str) -> Optional[str]:
    for f in fr.findall("./feat"):
        if f.get("att") == att:
            return f.get("val")
    return None

# ---------- SALDO parsing & animacy ----------

ANIMATE_ROOT_SENSES = {
    "människa..1",
    "person..1",
    "djur..1",
    #"varelse..1",
}

def parse_saldo_lmf(saldo_xml: str):
    """
    Return:
      sense2info: sense_id -> (writtenForm, pos, lemgram)
      primary_parents: sense_id -> [parent_sense_id, ...] via label="primary"
      lemgram2senses: lemgram -> [sense_id, ...]
      lemgram2form: lemgram -> a representative writtenForm
    """
    sense2info: Dict[str, Tuple[str, str, str]] = {}
    primary_parents: Dict[str, List[str]] = defaultdict(list)
    lemgram2senses: Dict[str, List[str]] = defaultdict(list)
    lemgram2form: Dict[str, str] = {}

    tree = ET.parse(saldo_xml)
    root = tree.getroot()

    for le in root.findall(".//LexicalEntry"):
        fr = le.find("./Lemma/FormRepresentation")
        if fr is None:
            continue
        written = normalize_key(get_feat(fr, "writtenForm") or "")
        pos     = get_feat(fr, "partOfSpeech") or ""
        lemgram = normalize_key(get_feat(fr, "lemgram") or "")

        # Representative lemma form per lemgram (first wins; SALDO is consistent)
        if lemgram and lemgram not in lemgram2form:
            lemgram2form[lemgram] = written

        for s in le.findall("./Sense"):
            sid = s.get("id")
            if not sid:
                continue
            sid = normalize_key(sid)  # e.g. "djur..1"
            sense2info[sid] = (written, pos, lemgram)
            if lemgram:
                lemgram2senses[lemgram].append(sid)

            for rel in s.findall("./SenseRelation"):
                label = None
                for feat in rel.findall("./feat"):
                    if feat.get("att") == "label":
                        label = feat.get("val")
                        break
                if label == "primary":
                    targets = (rel.get("targets") or "").split()
                    for t in targets:
                        t = normalize_key(t)
                        if t:
                            primary_parents[sid].append(t)

    return sense2info, primary_parents, lemgram2senses, lemgram2form

def compute_animacy_and_paths(
    sense2info: Dict[str, Tuple[str, str, str]],
    primary_parents: Dict[str, List[str]]
):
    """
    Returns two callables:
      is_animate(sense_id) -> bool
      paths_to_roots(sense_id) -> List[List[str]]   (each path is sense_id list from node to root)
    """
    memo_anim: Dict[str, bool] = {}

    def is_animate(sid: str, seen: Optional[Set[str]]=None) -> bool:
        if sid in memo_anim:
            return memo_anim[sid]
        if seen is None: seen = set()
        if sid in seen:
            memo_anim[sid] = False
            return False
        seen.add(sid)

        if sid in ANIMATE_ROOT_SENSES:
            memo_anim[sid] = True
            return True
        for p in primary_parents.get(sid, []):
            if is_animate(p, seen):
                memo_anim[sid] = True
                return True
        memo_anim[sid] = False
        return False

    def paths_to_roots(sid: str) -> List[List[str]]:
        """All primary-parent paths until a node with no parent, or an animate root."""
        paths: List[List[str]] = []

        def dfs(cur: str, seen: Set[str], acc: List[str]):
            if cur in seen:
                paths.append(acc + [cur])
                return
            seen = set(seen)
            seen.add(cur)
            parents = primary_parents.get(cur, [])
            if not parents:
                paths.append(acc + [cur])
                return
            for p in parents:
                dfs(p, seen, acc + [cur])
        dfs(sid, set(), [])
        return paths

    return is_animate, paths_to_roots

def pick_best_path(paths: List[List[str]], target_set: Set[str]) -> Optional[List[str]]:
    """Choose a path that reaches a target (shortest first). If none reach, return the shortest overall."""
    if not paths:
        return None
    reaching = [p for p in paths if p and p[-1] in target_set]
    if reaching:
        return min(reaching, key=len)
    return min(paths, key=len)

# ---------- frequency parsing ----------

def build_lemgram_frequencies(stats_path: str) -> Dict[str, int]:
    """
    Read the big stats file (plain/gz/zip), keep NN rows,
    extract lemgram from field 3, frequency from field 5, sum by lemgram.
    """
    sums: Dict[str, int] = defaultdict(int)
    with open_maybe_compressed(stats_path) as fh:
        for line in fh:
            # tab-separated: w, POS, |lemgram|, -, freq, per_million
            if "\t" not in line:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            pos = parts[1]
            if not pos.startswith("NN"):
                continue
            lemgram_raw = parts[2]
            lemgram = normalize_key(lemgram_raw)
            if len(lemgram) <= 1:
                continue
            try:
                freq = int(parts[4])
            except ValueError:
                # header or malformed row
                continue
            sums[lemgram] += freq
    return sums

# ---------- join & write ----------

def senses_to_path_string(path_sense_ids: List[str],
                          sense2info: Dict[str, Tuple[str, str, str]]) -> str:
    """Render a path as 'form → form → ...'."""
    forms = []
    for sid in path_sense_ids:
        w, _, _ = sense2info.get(sid, (sid, "", ""))
        forms.append(w)
    return " \u2192 ".join(forms)  # arrow

def aggregate_lemgram_animacy(
    lemgram2senses: Dict[str, List[str]],
    sense2info: Dict[str, Tuple[str, str, str]],
    is_animate,
    paths_to_roots,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    For each lemgram, decide a single animacy label and one representative path (as readable string).
    Policy:
      - animate if ANY sense is animate (clean, recall-oriented)
      - path = shortest path of any animate sense to an animate root
              (fallback: shortest path of any sense)
    Returns:
      lemgram2animacy: lemgram -> 'animate' | 'inanimate' | 'unknown'
      lemgram2path:    lemgram -> 'form → ...' ('' if unknown)
    """
    lemgram2animacy: Dict[str, str] = {}
    lemgram2path: Dict[str, str] = {}
    for lg, senses in lemgram2senses.items():
        nn_senses = [s for s in senses if sense2info.get(s, ("","", ""))[1] == "nn"]
        if not nn_senses:
            lemgram2animacy[lg] = "unknown"
            lemgram2path[lg] = ""
            continue

        animate_senses = [s for s in nn_senses if is_animate(s)]
        if animate_senses:
            # choose best animate path
            cand_paths = [pick_best_path(paths_to_roots(s), ANIMATE_ROOT_SENSES) for s in animate_senses]
            cand_paths = [p for p in cand_paths if p]
            if cand_paths:
                best = min(cand_paths, key=len)
                lemgram2animacy[lg] = "animate"
                lemgram2path[lg] = senses_to_path_string(best, sense2info)
                continue

        # else: no animate sense
        all_paths = [pick_best_path(paths_to_roots(s), ANIMATE_ROOT_SENSES) for s in nn_senses]
        all_paths = [p for p in all_paths if p]
        if all_paths:
            best = min(all_paths, key=len)
            lemgram2animacy[lg] = "inanimate"
            lemgram2path[lg] = senses_to_path_string(best, sense2info)
        else:
            lemgram2animacy[lg] = "unknown"
            lemgram2path[lg] = ""

    return lemgram2animacy, lemgram2path

def main():
    ap = argparse.ArgumentParser(description="Join SALDO animacy+paths with lemma frequencies.")
    ap.add_argument("--saldo-xml", required=True, help="Path to saldo.xml (LMF format).")
    ap.add_argument("--stats", required=True,
                    help="Path to stats_all.txt (plain), .gz or single-file .zip.")
    ap.add_argument("--out", default="lemma_freq_animacy_paths.tsv",
                    help="Output TSV (writtenForm, lemgram, frequency, animacy, path).")
    ap.add_argument("--unmatched", default="unmatched_lemgrams.txt",
                    help="Write lemgrams seen in frequency but not in SALDO here.")
    args = ap.parse_args()

    print("[1/4] Parsing SALDO…", file=sys.stderr)
    sense2info, primary_parents, lemgram2senses, lemgram2form = parse_saldo_lmf(args.saldo_xml)
    is_animate, paths_to_roots = compute_animacy_and_paths(sense2info, primary_parents)

    print("[2/4] Building lemma frequencies (NN only)…", file=sys.stderr)
    freq_by_lemgram = build_lemgram_frequencies(args.stats)

    print("[3/4] Aggregating animacy & paths per lemgram…", file=sys.stderr)
    lg2anim, lg2path = aggregate_lemgram_animacy(
        lemgram2senses, sense2info, is_animate, paths_to_roots
    )

    print("[4/4] Joining & writing…", file=sys.stderr)
    # Header
    out = io.open(args.out, "w", encoding="utf-8", newline="")
    out.write("writtenForm\tlemgram\tfrequency\tanimacy\tpath\n")

    # Join on lemgram, prefer SALDO writtenForm; fallback: derive from lemgram token
    unmatched = []
    for lg, freq in sorted(freq_by_lemgram.items(), key=lambda kv: (-kv[1], kv[0])):
        form = lemgram2form.get(lg) or lg.split("..", 1)[0]
        anim = lg2anim.get(lg, "unknown")
        path = lg2path.get(lg, "")
        if lg not in lg2anim:
            unmatched.append(lg)
        out.write(f"{form}\t{lg}\t{freq}\t{anim}\t{path}\n")
    out.close()

    with io.open(args.unmatched, "w", encoding="utf-8") as uh:
        uh.write("\n".join(sorted(set(unmatched))))

    print(f"Done. Wrote: {args.out}", file=sys.stderr)
    print(f"Unmatched lemgrams: {len(set(unmatched))} (see {args.unmatched})", file=sys.stderr)

if __name__ == "__main__":
    main()


