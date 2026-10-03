"""Build the frozen common measurement bank.

The bank is a shared prompt/token pool every compatible instrument pipeline can
measure (FP16 signal stats, compression perturbation, SPR quantities, shared
layer behavior). It is NOT training data for every instrument — its purpose is
measurement. Each record carries ID, source, text, token count, category, and
license/source metadata; a `.meta.json` sidecar records the exact provenance
(dataset, revision, tokenizer, seed, command, git commit) so the bank is
reproducible.

Leakage rule: probe training examples must not be reused in
the probe evaluation partition. This script records `category` per record so
probe-domain items stay separable from supervised splits; it never mixes splits
silently (source split is recorded verbatim per record).

Bank-source decision: the default source is WikiText-2-raw train (the pinned
reproduction corpus) with a small default size. BOTH are explicit CLI defaults
for smoke use only — the LEAD must choose the production source/size and
re-run; the sidecar records whatever was actually used.

Usage (from the repo root):
    python scripts/build_measurement_bank.py --help
    # smoke/offline (no network): build from a local text file
    python scripts/build_measurement_bank.py --input-text-file /tmp/texts.txt \\
        --tokenizer whitespace --license "UNVERIFIED - confirm before publication" \\
        --output data/manifests/r1_common_measurement_bank.jsonl
    # production (needs network + HF access):
    python scripts/build_measurement_bank.py --dataset wikitext \\
        --dataset-config wikitext-2-raw-v1 --split train \\
        --revision b08601e04326c79dfdd32d625aee71d232d685c3 \\
        --tokenizer google/gemma-2-2b --max-examples 10000 \\
        --license "CC-BY-SA-4.0 (WikiText-2) - confirm before publication"
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUT = "data/manifests/r1_common_measurement_bank.jsonl"
DEFAULT_MAX_EXAMPLES = 1000  # smoke default only; LEAD chooses production size


def make_record_id(source: str, index: int) -> str:
    """Stable, deterministic record ID (fail loudly on bad inputs)."""
    if not source or not source.strip():
        raise ValueError("record source must be a non-empty string")
    if index < 0:
        raise ValueError(f"record index must be non-negative, got {index}")
    return f"{source.strip()}:{index:06d}"


def count_tokens_whitespace(text: str) -> int:
    """Offline token counter (whitespace split), labeled as such in the record."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return len(text.split())


def build_records(
    texts: list[str],
    *,
    source: str,
    tokenizer_name: str,
    count_fn,
    category: str,
    license: str,
    dataset_id: str,
    dataset_config: str | None,
    dataset_split: str,
    dataset_revision: str | None,
) -> tuple[list[dict], int]:
    """Build manifest records, skipping empty/whitespace-only texts.

    Returns (records, n_skipped_empty). Raises if nothing survives filtering —
    an empty bank is never written silently.
    """
    if not texts:
        raise ValueError("no input texts supplied; refusing to write an empty bank")
    records: list[dict] = []
    skipped = 0
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        index = len(records)
        records.append(
            {
                "id": make_record_id(source, index),
                "source": source,
                "text": text,
                "token_count": int(count_fn(text)),
                "tokenizer": tokenizer_name,
                "category": category,
                "license": license,
                "dataset_id": dataset_id,
                "dataset_config": dataset_config,
                "dataset_split": dataset_split,
                "dataset_revision": dataset_revision,
            }
        )
    if not records:
        raise ValueError(
            f"all {len(texts)} input texts were empty/whitespace-only; "
            "refusing to write an empty bank"
        )
    return records, skipped


def write_manifest(records: list[dict], path: Path) -> None:
    """Write records as JSONL (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip() if out.returncode == 0 else ""
    return commit or None


def write_sidecar(
    *,
    path: Path,
    records: list[dict],
    n_skipped_empty: int,
    args: argparse.Namespace,
) -> None:
    """Write the provenance sidecar next to the manifest."""
    total_tokens = sum(r["token_count"] for r in records)
    sidecar = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "dataset_id": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.split,
        "dataset_revision": args.revision,
        "tokenizer": args.tokenizer,
        "category": args.category,
        "license": args.license,
        "text_field": args.text_field,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "n_records": len(records),
        "n_skipped_empty": n_skipped_empty,
        "total_tokens": total_tokens,
        "sha256_manifest": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    sidecar_path = path.with_suffix(path.suffix + ".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"[BANK] sidecar -> {sidecar_path}")


def load_text_file(path: Path, max_examples: int) -> list[str]:
    """Load one-text-per-line from a local file (offline path)."""
    if not path.exists():
        raise FileNotFoundError(
            f"input text file not found: {path}. "
            "Provide --input-text-file pointing at a real file."
        )
    texts = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return texts[:max_examples]


def load_hf_texts(
    dataset: str,
    config: str | None,
    split: str,
    revision: str | None,
    text_field: str,
    max_examples: int,
    seed: int,
) -> list[str]:
    """Load texts from HuggingFace (network path). Fail loudly with remediation."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "the 'datasets' package is required for HF loading "
            "(pip install -r requirements.txt)."
        ) from exc
    if not dataset or not dataset.strip():
        raise ValueError("--dataset must be a non-empty HF dataset id")
    load_kwargs: dict = {"split": split, "streaming": True}
    if config:
        load_kwargs["name"] = config
    if revision:
        load_kwargs["revision"] = revision
    try:
        ds = load_dataset(dataset, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load dataset {dataset!r} (config={config!r}, split={split!r}, "
            f"revision={revision!r}): {exc}. Check the id/revision, network access, "
            "and (for gated datasets) `huggingface-cli login`."
        ) from exc
    texts: list[str] = []
    for row in ds:
        if len(texts) >= max_examples:
            break
        if text_field not in row:
            raise ValueError(
                f"text field {text_field!r} missing from dataset rows "
                f"(available: {sorted(row.keys())}); pass --text-field explicitly."
            )
        texts.append(row[text_field])
    _ = seed  # recorded in sidecar; selection is deterministic prefix order
    return texts


def make_tokenizer_counter(tokenizer: str):
    """Return (tokenizer_name, count_fn) for an HF id or the 'whitespace' fallback."""
    if tokenizer == "whitespace":
        return "whitespace", count_tokens_whitespace
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "the 'transformers' package is required for HF tokenizers "
            "(pip install -r requirements.txt), or use --tokenizer whitespace "
            "for offline smoke use."
        ) from exc
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load tokenizer {tokenizer!r}: {exc}. Check network access "
            "and (for gated tokenizers) `huggingface-cli login`."
        ) from exc

    def _count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return tokenizer, _count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen common measurement bank."
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="HF dataset id (default: wikitext, the pinned repro corpus family)",
    )
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--revision",
        default=None,
        help="pinned dataset revision; recorded verbatim (null if omitted, with warning)",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--input-text-file",
        type=Path,
        default=None,
        help="offline path: one text per line; skips HF loading entirely",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/gemma-2-2b",
        help="HF tokenizer id, or 'whitespace' for offline smoke use (labeled as such)",
    )
    parser.add_argument("--category", default="general-web")
    parser.add_argument(
        "--license",
        required=True,
        help="REQUIRED: license/source statement recorded verbatim per record",
    )
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_examples <= 0:
        raise SystemExit("--max-examples must be positive")
    if args.revision is None:
        print("[BANK] WARNING: no --revision pinned; recording dataset_revision=null. "
              "Pin a revision for production banks.")

    if args.input_text_file is not None:
        texts = load_text_file(args.input_text_file, args.max_examples)
        source = f"local:{args.input_text_file.name}"
        dataset_id, dataset_config, dataset_split = source, None, "local"
    else:
        texts = load_hf_texts(
            args.dataset, args.dataset_config, args.split, args.revision,
            args.text_field, args.max_examples, args.seed,
        )
        source = args.dataset
        dataset_id, dataset_config, dataset_split = args.dataset, args.dataset_config, args.split

    tokenizer_name, count_fn = make_tokenizer_counter(args.tokenizer)
    records, skipped = build_records(
        texts,
        source=source,
        tokenizer_name=tokenizer_name,
        count_fn=count_fn,
        category=args.category,
        license=args.license,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        dataset_revision=args.revision,
    )
    write_manifest(records, args.output)
    print(f"[BANK] wrote {len(records)} records ({skipped} empty skipped) -> {args.output}")
    write_sidecar(path=args.output, records=records, n_skipped_empty=skipped, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
