"""
Heuristic NOVA classifier for tokenized recipe TSVs

This script reads `ar_gk_test.tsv` in the same directory, groups tokens into
recipes by blank lines, imports NOVA keyword dictionaries from
`nova_keywords.py` (robust to common accidental formatting), applies a
keyword+regex-based scoring heuristic, assigns a NOVA label (1..4) per
recipe, and writes `classified_nova_recipes.csv` with `recipe_text` and
`nova_label`.

Run: python classify_nova.py

Requirements: Python 3.6+. Uses pandas if available; falls back to csv
parsing if pandas is not installed.
"""

import os
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "ar_gk_test.tsv")
OUT_CSV = os.path.join(BASE_DIR, "classified_nova_recipes.csv")
NOVA_MODULE_PATH = os.path.join(BASE_DIR, "nova_keywords.py")


def load_nova_module() -> Tuple[Dict[int, List[str]], Dict[int, float], Dict[int, List[str]]]:
    """Attempt to import NOVA dictionaries, with a fallback that reads and
    execs the file after stripping Markdown code fences if necessary.

    Returns (NOVA_KEYWORDS, NOVA_WEIGHTS, NOVA_REGEX_PATTERNS)
    """
    try:
        # Preferred: normal import
        import importlib.util

        spec = importlib.util.spec_from_file_location("nova_keywords", NOVA_MODULE_PATH)
        nova = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nova)  # type: ignore

        NOVA_KEYWORDS = getattr(nova, "NOVA_KEYWORDS")
        NOVA_WEIGHTS = getattr(nova, "NOVA_WEIGHTS")
        NOVA_REGEX_PATTERNS = getattr(nova, "NOVA_REGEX_PATTERNS", {})
        return NOVA_KEYWORDS, NOVA_WEIGHTS, NOVA_REGEX_PATTERNS
    except Exception as e:
        # Fallback: read the file and try to exec after removing code fences
        print(f"Warning: direct import of nova_keywords.py failed: {e}")
        print("Attempting to parse nova_keywords.py as text and exec it (stripping fences)...")
        txt = open(NOVA_MODULE_PATH, "r", encoding="utf-8").read()
        # strip common markdown code fences ``` or ```python
        txt_stripped = re.sub(r"^\s*```(?:python)?\s*", "", txt, flags=re.IGNORECASE)
        txt_stripped = re.sub(r"\s*```\s*$", "", txt_stripped, flags=re.IGNORECASE)

        # Execute in a controlled namespace
        env = {}
        try:
            exec(txt_stripped, env)
        except Exception as e2:
            print("Failed to exec nova_keywords.py content:", e2)
            raise

        NOVA_KEYWORDS = env.get("NOVA_KEYWORDS")
        NOVA_WEIGHTS = env.get("NOVA_WEIGHTS")
        NOVA_REGEX_PATTERNS = env.get("NOVA_REGEX_PATTERNS", {})

        if NOVA_KEYWORDS is None or NOVA_WEIGHTS is None:
            raise RuntimeError("nova_keywords.py did not define NOVA_KEYWORDS or NOVA_WEIGHTS")

        return NOVA_KEYWORDS, NOVA_WEIGHTS, NOVA_REGEX_PATTERNS


def read_tsv_grouped(tsv_path: str) -> List[str]:
    """Read the TSV and group tokens into recipes separated by blank lines.

    Returns a list of recipe texts (strings).
    """
    # Try to use pandas for convenience
    try:
        import pandas as pd

        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["token", "tag"],
                         dtype=str, keep_default_na=False, na_values=[""], skip_blank_lines=False)
        tokens = df["token"].tolist()
    except Exception:
        # Fallback: simple parser
        tokens = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    tokens.append("")
                else:
                    # split first tab
                    parts = line.split("\t")
                    tokens.append(parts[0])

    recipes: List[str] = []
    current_tokens: List[str] = []

    for t in tokens:
        # treat empty string or strings of whitespace as a separator
        if t is None:
            # treat None as separator
            if current_tokens:
                recipes.append(" ".join(current_tokens))
                current_tokens = []
            continue

        if isinstance(t, float):
            # NaN from pandas; treat as separator
            if current_tokens:
                recipes.append(" ".join(current_tokens))
                current_tokens = []
            continue

        token = str(t).strip()
        if token == "":
            if current_tokens:
                recipes.append(" ".join(current_tokens))
                current_tokens = []
            else:
                # consecutive separators -> ignore
                continue
        else:
            current_tokens.append(token)

    # catch any trailing recipe
    if current_tokens:
        recipes.append(" ".join(current_tokens))

    return recipes


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation (preserve alphanumerics and spaces), and
    collapse whitespace.
    """
    if text is None:
        return ""
    t = text.lower()
    # Replace slashes and hyphens with spaces so phrases like 'instant-mix' match 'instant mix'
    t = t.replace("/", " ").replace("-", " ")
    # Remove punctuation except alphanumerics and whitespace
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_keyword_patterns(nova_keywords: Dict[int, List[str]]) -> Dict[int, List[re.Pattern]]:
    patterns: Dict[int, List[re.Pattern]] = {}
    for lvl, words in nova_keywords.items():
        pats = []
        for w in words:
            if not w:
                continue
            # Treat entries as literal phrases; escape to avoid accidental regex
            esc = re.escape(w)
            # Word boundary around phrase
            pat = re.compile(r"\b" + esc + r"\b", flags=re.IGNORECASE)
            pats.append(pat)
        patterns[lvl] = pats
    return patterns


def classify_recipes(recipes: List[str], nova_keywords, nova_weights, nova_regex_patterns) -> List[Tuple[str, int, Dict[int, float]]]:
    results = []

    # Precompile keyword patterns
    kw_patterns = build_keyword_patterns(nova_keywords)

    # Precompile any regex patterns provided (NOVA_REGEX_PATTERNS)
    regex_patterns_compiled: Dict[int, List[re.Pattern]] = {}
    for lvl, patterns in (nova_regex_patterns or {}).items():
        regex_patterns_compiled[lvl] = [re.compile(p, flags=re.IGNORECASE) for p in patterns]

    for recipe in recipes:
        norm = normalize_text(recipe)
        scores: Dict[int, float] = defaultdict(float)
        counts: Dict[int, int] = defaultdict(int)

        for lvl in sorted(nova_keywords.keys()):
            # keyword matches
            pats = kw_patterns.get(lvl, [])
            for pat in pats:
                matches = pat.findall(norm)
                if matches:
                    counts[lvl] += len(matches)

            # regex patterns (explicit) for this level
            for pat in regex_patterns_compiled.get(lvl, []):
                m = pat.findall(norm)
                if m:
                    counts[lvl] += len(m)

            # compute weighted score
            weight = float(nova_weights.get(lvl, 1.0))
            scores[lvl] = counts[lvl] * weight

        # Decide final label: highest score, tie -> higher level, no matches -> 1
        if not any(v > 0 for v in scores.values()):
            final = 1
        else:
            # max by (score, level) chooses higher level on tie
            best_lvl = max(((s, lvl) for lvl, s in scores.items()), key=lambda x: (x[0], x[1]))[1]
            final = int(best_lvl)

        results.append((recipe, final, dict(scores)))

    return results


def save_results(results: List[Tuple[str, int, Dict[int, float]]], out_csv: str):
    # Try to use pandas for nice CSV writing, else fallback
    rows = [(r[0], r[1]) for r in results]
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=["recipe_text", "nova_label"])
        df.to_csv(out_csv, index=False)
    except Exception:
        # simple CSV writer
        import csv

        with open(out_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["recipe_text", "nova_label"])
            for row in rows:
                writer.writerow(row)


def print_summary(results: List[Tuple[str, int, Dict[int, float]]], head: int = 10):
    print(f"Total recipes classified: {len(results)}")
    counts = Counter(r[1] for r in results)
    for lvl in sorted(counts.keys()):
        print(f"  NOVA {lvl}: {counts[lvl]}")

    print("\nFirst labeled recipes:")
    for recipe, label, scores in results[:head]:
        print("-" * 60)
        print(f"NOVA {label} | scores: {scores}")
        print(recipe)


def main():
    if not os.path.exists(TSV_PATH):
        print(f"TSV file not found: {TSV_PATH}")
        sys.exit(1)

    nova_keywords, nova_weights, nova_regex_patterns = load_nova_module()

    recipes = read_tsv_grouped(TSV_PATH)
    print(f"Loaded {len(recipes)} recipes from TSV (grouped by blank lines)")

    results = classify_recipes(recipes, nova_keywords, nova_weights, nova_regex_patterns)

    save_results(results, OUT_CSV)
    print(f"Saved classified recipes to: {OUT_CSV}")

    print_summary(results, head=10)


if __name__ == "__main__":
    main()
