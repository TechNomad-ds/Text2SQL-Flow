import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import ijson
from tqdm import tqdm

# -----------------------------
# Streaming JSON array I/O
# -----------------------------
def iter_json_array(path: str) -> Iterable[dict]:
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item

def write_json_array(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for rec in records:
            if not first:
                f.write(",\n")
            json.dump(rec, f, ensure_ascii=False)
            first = False
        f.write("\n]\n")

# -----------------------------
# SQL normalization
# -----------------------------
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"--[^\n\r]*")
WS_RE = re.compile(r"\s+")

def normalize_sql_exact(s: str) -> str:
    """Conservative normalization for exact/near-duplicate detection."""
    if not s:
        return ""
    s = str(s)
    s = BLOCK_COMMENT_RE.sub(" ", s)
    s = LINE_COMMENT_RE.sub(" ", s)
    s = s.strip()
    s = WS_RE.sub(" ", s)
    s = s.lower()

    s = s.rstrip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s

# -----------------------------
# SQL tokenizer and ngrams
# -----------------------------
TOKEN_RE = re.compile(
    r"""
    '(?:[^'\\]|\\.)*'          |  # single-quoted strings
    "(?:[^"\\]|\\.)*"          |  # double-quoted strings
    \b\d+\.\d+\b               |  # floats
    \b\d+\b                    |  # ints
    <=|>=|<>|!=                |  # multi-char operators
    [=<>*/(),.;]               |  # single-char operators/punct
    [A-Za-z_][A-Za-z0-9_]*        # identifiers/keywords
    """,
    re.VERBOSE,
)

def tokenize_sql(s: str) -> List[str]:
    if not s:
        return []
    return [m.group(0).lower() for m in TOKEN_RE.finditer(s)]

def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# -----------------------------
# Spec parsing
# -----------------------------
@dataclass
class TestSpec:
    name: str
    path: str
    skey: str

def parse_tests(test_specs: List[str]) -> List[TestSpec]:
    out: List[TestSpec] = []
    for s in test_specs:
        parts = s.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid test spec: {s}")
        out.append(TestSpec(parts[0], parts[1], parts[2]))
    return out

# -----------------------------
# Build exact-match set
# -----------------------------
def build_test_sql_set(tests: List[TestSpec]) -> Tuple[set, Dict[str, int]]:
    global_set = set()
    per_test_counts: Dict[str, int] = {t.name: 0 for t in tests}

    for t in tests:
        seen = 0
        for rec in tqdm(iter_json_array(t.path), desc=f"Indexing exact SQL [{t.name}]", unit="sql"):
            norm = normalize_sql_exact(rec.get(t.skey, ""))
            if not norm:
                continue
            seen += 1
            global_set.add(norm)
        per_test_counts[t.name] = seen

    return global_set, per_test_counts

# -----------------------------
# Build n-gram inverted index over test SQLs
# -----------------------------
def build_test_ngram_index(
    tests: List[TestSpec],
    n: int,
) -> dict:
    """
    Build a global index across all test specs:
      - gram_sets: list[set(ngram)]
      - norm_sql_list: list[str]
      - test_name_list: list[str]
      - inv: dict[ngram] -> set(record_id)
    """
    inv: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
    gram_sets: List[Set[Tuple[str, ...]]] = []
    norm_sql_list: List[str] = []
    test_name_list: List[str] = []

    rid = 0
    for t in tests:
        for rec in tqdm(iter_json_array(t.path), desc=f"Building {n}-gram index [{t.name}]", unit="sql"):
            norm = normalize_sql_exact(rec.get(t.skey, ""))
            if not norm:
                continue
            toks = tokenize_sql(norm)
            gs = ngrams(toks, n)

            gram_sets.append(gs)
            norm_sql_list.append(norm)
            test_name_list.append(t.name)

            for g in gs:
                inv[g].add(rid)

            rid += 1

    return {
        "n": n,
        "inv": inv,
        "gram_sets": gram_sets,
        "norm_sql_list": norm_sql_list,
        "test_name_list": test_name_list,
        "num_records": rid,
    }

def best_jaccard_against_tests(
    input_grams: Set[Tuple[str, ...]],
    test_index: dict,
    threshold: float,
    max_candidates: int = 2000,
) -> Tuple[float, Optional[int], int]:
    """
    Return:
      (best_jaccard, best_rid, num_candidates_considered)
    """
    inv = test_index["inv"]
    gram_sets = test_index["gram_sets"]

    if not input_grams:
        return 0.0, None, 0

    counter = Counter()
    for g in input_grams:
        for rid in inv.get(g, ()):
            counter[rid] += 1

    if not counter:
        return 0.0, None, 0

    candidate_ids = [rid for rid, _ in counter.most_common(max_candidates)]

    best_j = 0.0
    best_rid = None

    for rid in candidate_ids:
        tj = jaccard(input_grams, gram_sets[rid])
        if tj > best_j:
            best_j = tj
            best_rid = rid
            # early exit if already exceeds threshold strongly
            if best_j >= threshold:
                break

    return best_j, best_rid, len(candidate_ids)

# -----------------------------
# Main
# -----------------------------
def main():
    input_path = ""
    report_path = ""

    input_sql_key = "SQL"
    input_question_key = "question"
    data_source_key = "data_source"

    test_specs = [
        "dev_spider:data/spider/dev.json:query",
        "test_spider:data/spider/test.json:query",
        "dev_bird:data/bird/dev_20240627/dev.json:SQL",
        "dev_ehrsql:data/EHRSQL/dev.json:query",
    ]

    # Analysis config: try multiple n-gram sizes
    ngram_n_values = [8, 10, 12]
    jaccard_threshold = 0.8  
    max_candidates = 2000

    tests = parse_tests(test_specs)

    # 1) exact set 
    test_sql_set, per_test_nonempty = build_test_sql_set(tests)

    # Map exact norm SQL to test name 
    norm_to_testname: Dict[str, str] = {}
    for t in tests:
        for rec in tqdm(iter_json_array(t.path), desc=f"Mapping exact SQL → test [{t.name}]", unit="sql"):
            norm = normalize_sql_exact(rec.get(t.skey, ""))
            if norm and norm not in norm_to_testname:
                norm_to_testname[norm] = t.name

    # -----------------------------
    # Analysis: try multiple n-gram sizes
    # -----------------------------
    results_by_n: Dict[int, Dict[str, any]] = {}
    for ngram_n in ngram_n_values:
        print(f"\n[Analyzing with n={ngram_n}...]")
        test_ng = build_test_ngram_index(tests, n=ngram_n)

        # Stats by data_source
        stats_by_source: Dict[str, dict] = defaultdict(lambda: {
            "total": 0,
            "filtered_exact": 0,
            "filtered_jaccard": 0,
            "filtered_total": 0,
            "filtered_by_test": defaultdict(int),
        })

        total_in = 0
        total_filtered = 0
        filtered_exact = 0
        filtered_jaccard = 0

        for idx, rec in enumerate(
            tqdm(iter_json_array(input_path), desc=f"Analyzing n={ngram_n}", unit="sql")
        ):
            total_in += 1
            data_source = (rec.get(data_source_key) or "unknown").lower()
            raw_sql = rec.get(input_sql_key, "")
            norm = normalize_sql_exact(raw_sql)

            stats = stats_by_source[data_source]
            stats["total"] += 1

            # Rule A: exact match
            if norm and norm in test_sql_set:
                total_filtered += 1
                filtered_exact += 1
                stats["filtered_exact"] += 1
                stats["filtered_total"] += 1

                tn = norm_to_testname.get(norm, "unknown")
                stats["filtered_by_test"][tn] += 1
                continue

            # Rule B: n-gram Jaccard >= threshold
            toks = tokenize_sql(norm)
            grams = ngrams(toks, ngram_n)
            best_j, best_rid, cand = best_jaccard_against_tests(
                grams, test_ng, threshold=jaccard_threshold, max_candidates=max_candidates
            )

            if best_j >= jaccard_threshold and best_rid is not None:
                total_filtered += 1
                filtered_jaccard += 1
                stats["filtered_jaccard"] += 1
                stats["filtered_total"] += 1

                tn = test_ng["test_name_list"][best_rid]
                stats["filtered_by_test"][tn] += 1

        # Convert defaultdict to regular dict for serialization
        for source in stats_by_source:
            stats_by_source[source]["filtered_by_test"] = dict(stats_by_source[source]["filtered_by_test"])

        results_by_n[ngram_n] = {
            "total_input": total_in,
            "total_filtered": total_filtered,
            "filtered_exact": filtered_exact,
            "filtered_jaccard": filtered_jaccard,
            "by_data_source": dict(stats_by_source),
        }

    # -----------------------------
    # Generate TXT report
    # -----------------------------
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("Containment Analysis Report")
    lines.append("=" * 80)
    lines.append(f"Input file: {input_path}")
    lines.append(f"Jaccard threshold: {jaccard_threshold} (60%)")
    lines.append(f"Test sets: {', '.join(t.name for t in tests)}")
    lines.append(f"N-gram sizes tested: {ngram_n_values}")
    lines.append("")

    # Summary across all n values
    lines.append("Summary Across All N-gram Sizes")
    lines.append("-" * 80)
    for ngram_n in sorted(ngram_n_values):
        r = results_by_n[ngram_n]
        pct = (r['total_filtered'] / max(1, r['total_input'])) * 100.0
        lines.append(f"\nN-gram size: {ngram_n}")
        lines.append(f"  Total input: {r['total_input']:,}")
        lines.append(f"  Total filtered: {r['total_filtered']:,} ({pct:.2f}%)")
        lines.append(f"    - Exact match: {r['filtered_exact']:,}")
        lines.append(f"    - Jaccard >= {jaccard_threshold}: {r['filtered_jaccard']:,}")
        lines.append("")

    # Detailed breakdown by data_source for each n
    lines.append("\n" + "=" * 80)
    lines.append("Detailed Breakdown by Data Source")
    lines.append("=" * 80)
    for ngram_n in sorted(ngram_n_values):
        r = results_by_n[ngram_n]
        lines.append(f"\nN-gram size: {ngram_n}")
        lines.append("-" * 80)
        for source in sorted(r["by_data_source"].keys()):
            stats = r["by_data_source"][source]
            total = stats["total"]
            filtered = stats["filtered_total"]
            pct = (filtered / max(1, total)) * 100.0
            lines.append(f"\n  [{source.upper()}]")
            lines.append(f"    Total: {total:,}")
            lines.append(f"    Filtered: {filtered:,} ({pct:.2f}%)")
            lines.append(f"      - Exact match: {stats['filtered_exact']:,}")
            lines.append(f"      - Jaccard >= {jaccard_threshold}: {stats['filtered_jaccard']:,}")
            if stats["filtered_by_test"]:
                lines.append(f"    Filtered by test set:")
                for test_name, count in sorted(stats["filtered_by_test"].items(), key=lambda x: -x[1]):
                    lines.append(f"      - {test_name}: {count:,}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Analysis complete. Report written to: {report_path}")
    print(f"     Tested n-gram sizes: {ngram_n_values}")
    print(f"     Threshold: {jaccard_threshold} (60%)")

if __name__ == "__main__":
    main()
