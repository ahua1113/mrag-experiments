#!/usr/bin/env python3
"""Run DSE or ColPali on the same ViDoRe QA-format retrieval benchmark.

The script intentionally runs one model at a time so that small GPUs do not
need to keep both models resident. Embeddings are cached in deterministic
shards and can be reused after an interrupted run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image


DEFAULT_DATASETS = (
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
)


@dataclass
class BenchmarkData:
    dataset_name: str
    query_ids: list[str]
    queries: list[str]
    corpus_ids: list[str]
    corpus_images: list[Image.Image]
    qrels: dict[str, dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DSE or ColPali on identical ViDoRe corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=("dse", "colpali"), required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all queries")
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=0,
        help="0 means the full corpus; sampled corpora always retain every positive page",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--top-k-output", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-recompute", action="store_true")

    parser.add_argument("--colpali-model", default="vidore/colpali-v1.3")

    parser.add_argument("--dse-model", default="MrLight/dse-qwen2-2b-mrl-v1")
    parser.add_argument("--dse-dimension", type=int, default=512)
    parser.add_argument("--dse-max-image-patches", type=int, default=144)
    parser.add_argument("--dse-gpu-memory", default="3.2GiB")
    parser.add_argument("--dse-cpu-memory", default="12GiB")
    parser.add_argument(
        "--dse-no-4bit",
        action="store_true",
        help="Disable NF4 quantization (normally unsuitable for a 4 GB GPU)",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def stable_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def unique_id(raw: Any, prefix: str, seen: set[str]) -> str:
    candidate = str(raw) if raw not in (None, "") else prefix
    base = candidate
    suffix = 1
    while candidate in seen:
        candidate = f"{base}__{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def image_key(row: dict[str, Any], row_index: int) -> str:
    for field in ("image_filename", "page", "docId", "document_id"):
        value = row.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"
    return f"row:{row_index}"


def load_benchmark_data(
    dataset_name: str,
    split: str,
    max_queries: int,
    corpus_size: int,
    seed: int,
) -> BenchmarkData:
    dataset = load_dataset(dataset_name, split=split)

    corpus_ids: list[str] = []
    corpus_images: list[Image.Image] = []
    doc_key_to_id: dict[str, str] = {}
    seen_doc_ids: set[str] = set()
    rows: list[tuple[str, str, str]] = []
    seen_query_ids: set[str] = set()

    for row_index, row in enumerate(dataset):
        if "query" not in row or "image" not in row:
            raise ValueError(
                f"{dataset_name} is not a ViDoRe QA-format dataset: "
                "expected 'query' and 'image' columns."
            )

        key = image_key(row, row_index)
        if key not in doc_key_to_id:
            raw_doc_id = row.get("image_filename") or row.get("page") or row.get("docId")
            doc_id = unique_id(raw_doc_id, f"doc_{len(corpus_ids)}", seen_doc_ids)
            image = row["image"]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL.Image for {doc_id}, got {type(image)!r}")
            doc_key_to_id[key] = doc_id
            corpus_ids.append(doc_id)
            corpus_images.append(image.convert("RGB"))

        raw_query_id = row.get("questionId") or row.get("query_id") or row_index
        query_id = unique_id(raw_query_id, f"query_{row_index}", seen_query_ids)
        rows.append((query_id, str(row["query"]), doc_key_to_id[key]))

    rng = random.Random(seed)
    if max_queries > 0 and max_queries < len(rows):
        selected = sorted(rng.sample(range(len(rows)), max_queries))
        rows = [rows[index] for index in selected]

    if corpus_size > 0 and corpus_size < len(corpus_ids):
        positive_ids = {positive_id for _, _, positive_id in rows}
        if corpus_size < len(positive_ids):
            raise ValueError(
                f"--corpus-size={corpus_size} is smaller than the "
                f"{len(positive_ids)} distinct positive pages."
            )
        negative_ids = [doc_id for doc_id in corpus_ids if doc_id not in positive_ids]
        keep_ids = set(positive_ids)
        keep_ids.update(rng.sample(negative_ids, corpus_size - len(positive_ids)))
        kept = [i for i, doc_id in enumerate(corpus_ids) if doc_id in keep_ids]
        corpus_ids = [corpus_ids[i] for i in kept]
        corpus_images = [corpus_images[i] for i in kept]

    query_ids = [query_id for query_id, _, _ in rows]
    queries = [query for _, query, _ in rows]
    qrels = {query_id: {positive_id: 1} for query_id, _, positive_id in rows}
    return BenchmarkData(dataset_name, query_ids, queries, corpus_ids, corpus_images, qrels)


class DSEBackend:
    def __init__(self, args: argparse.Namespace):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        self.name = "dse"
        self.model_name = args.dse_model
        self.dimension = args.dse_dimension
        self.device = args.device
        self.process_vision_info = process_vision_info
        max_pixels = args.dse_max_image_patches * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=28 * 28,
            max_pixels=max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        if not args.dse_no_4bit:
            model_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="auto",
                max_memory={0: args.dse_gpu_memory, "cpu": args.dse_cpu_memory},
            )
        else:
            model_kwargs["device_map"] = args.device

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **model_kwargs
        ).eval()
        self.model.padding_side = "left"

    def _encode(self, items: Sequence[Any], is_query: bool) -> list[torch.Tensor]:
        messages = []
        for item in items:
            if is_query:
                image = Image.new("RGB", (28, 28))
                content = [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 1,
                        "resized_width": 1,
                    },
                    {"type": "text", "text": f"Query: {item}"},
                ]
            else:
                content = [
                    {"type": "image", "image": item},
                    {"type": "text", "text": "What is shown in this image?"},
                ]
            messages.append([{"role": "user", "content": content}])

        texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            + "<|endoftext|>"
            for message in messages
        ]
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        cache_position = torch.arange(
            0, len(texts), device=inputs["input_ids"].device
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False
        )
        with torch.inference_mode():
            output = self.model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
            )
        embeddings = output.hidden_states[-1][:, -1, : self.dimension]
        embeddings = F.normalize(embeddings, p=2, dim=-1).float().cpu()
        result = list(torch.unbind(embeddings, dim=0))
        del output, model_inputs, inputs, embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def encode_documents(self, images: Sequence[Image.Image]) -> list[torch.Tensor]:
        return self._encode(images, is_query=False)

    def encode_queries(self, queries: Sequence[str]) -> list[torch.Tensor]:
        return self._encode(queries, is_query=True)

    def score(
        self,
        query_embeddings: Sequence[torch.Tensor],
        doc_embeddings: Sequence[torch.Tensor],
        score_batch_size: int,
    ) -> torch.Tensor:
        queries = torch.stack(list(query_embeddings))
        docs = torch.stack(list(doc_embeddings))
        return queries @ docs.T

    def cache_config(self, args: argparse.Namespace) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "max_image_patches": args.dse_max_image_patches,
            "four_bit": not args.dse_no_4bit,
        }


class ColPaliBackend:
    def __init__(self, args: argparse.Namespace):
        from sentence_transformers import MultiVectorEncoder

        self.name = "colpali"
        self.model_name = args.colpali_model
        try:
            self.model = MultiVectorEncoder(self.model_name, device=args.device)
        except TypeError:
            self.model = MultiVectorEncoder(self.model_name)
            if hasattr(self.model, "to"):
                self.model.to(args.device)

    @staticmethod
    def _as_cpu_tensor_list(values: Any) -> list[torch.Tensor]:
        if isinstance(values, torch.Tensor):
            return [value.detach().cpu() for value in torch.unbind(values, dim=0)]
        return [torch.as_tensor(value).detach().cpu() for value in values]

    def encode_documents(self, images: Sequence[Image.Image]) -> list[torch.Tensor]:
        values = self.model.encode_document(list(images), batch_size=len(images))
        return self._as_cpu_tensor_list(values)

    def encode_queries(self, queries: Sequence[str]) -> list[torch.Tensor]:
        values = self.model.encode_query(list(queries), batch_size=len(queries))
        return self._as_cpu_tensor_list(values)

    def score(
        self,
        query_embeddings: Sequence[torch.Tensor],
        doc_embeddings: Sequence[torch.Tensor],
        score_batch_size: int,
    ) -> torch.Tensor:
        chunks = []
        for start in range(0, len(query_embeddings), score_batch_size):
            batch = query_embeddings[start : start + score_batch_size]
            scores = self.model.similarity(batch, doc_embeddings)
            chunks.append(torch.as_tensor(scores).float().cpu())
        return torch.cat(chunks, dim=0)

    def cache_config(self, args: argparse.Namespace) -> dict[str, Any]:
        return {"model": self.model_name}


def shard_encode(
    items: Sequence[Any],
    item_ids: Sequence[str],
    encode_fn: Any,
    cache_dir: Path,
    batch_size: int,
    force_recompute: bool,
) -> tuple[list[torch.Tensor], float, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[torch.Tensor] = []
    elapsed_total = 0.0
    encoded_count = 0

    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        ids = list(item_ids[start:end])
        shard = cache_dir / f"{start:06d}_{end:06d}.pt"
        payload = None
        cache_hit = False
        if shard.exists() and not force_recompute:
            try:
                candidate = torch.load(shard, map_location="cpu", weights_only=False)
                if candidate.get("ids") == ids:
                    payload = candidate
                    cache_hit = True
            except (OSError, RuntimeError, ValueError, EOFError):
                payload = None

        if payload is None:
            started = time.perf_counter()
            batch_embeddings = encode_fn(items[start:end])
            elapsed = time.perf_counter() - started
            payload = {
                "ids": ids,
                "embeddings": [value.detach().cpu() for value in batch_embeddings],
                "elapsed_seconds": elapsed,
            }
            temporary = shard.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, shard)
            encoded_count += len(ids)

        embeddings.extend(payload["embeddings"])
        elapsed_total += float(payload.get("elapsed_seconds", 0.0))
        print(f"  [{end:>4}/{len(items)}] {'cached' if cache_hit else 'encoded'}")

    return embeddings, elapsed_total, encoded_count


def compute_metrics(
    scores: torch.Tensor,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray]:
    ranking = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
    doc_to_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
    per_query: list[dict[str, Any]] = []

    for query_index, query_id in enumerate(query_ids):
        relevant = {
            doc_to_index[doc_id]
            for doc_id, relevance in qrels[query_id].items()
            if relevance > 0 and doc_id in doc_to_index
        }
        ordered = ranking[query_index]
        relevant_ranks = [
            rank for rank, doc_index in enumerate(ordered, start=1) if doc_index in relevant
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None

        row: dict[str, Any] = {"query_id": query_id, "first_relevant_rank": first_rank}
        for k in (1, 5, 10):
            hits = sum(rank <= k for rank in relevant_ranks)
            row[f"recall_at_{k}"] = hits / max(1, len(relevant))
        row["mrr_at_10"] = 1.0 / first_rank if first_rank and first_rank <= 10 else 0.0

        for k in (5, 10):
            dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks if rank <= k)
            ideal_count = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            row[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
        per_query.append(row)

    metric_names = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_5", "ndcg_at_10")
    metrics = {
        name: float(np.mean([row[name] for row in per_query])) for name in metric_names
    }
    return metrics, per_query, ranking


def write_outputs(
    args: argparse.Namespace,
    backend: Any,
    data: BenchmarkData,
    metrics: dict[str, float],
    per_query: list[dict[str, Any]],
    ranking: np.ndarray,
    scores: torch.Tensor,
    timings: dict[str, float],
) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_name(data.dataset_name)}__{backend.name}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data.dataset_name,
        "split": args.split,
        "model_type": backend.name,
        "model_name": backend.model_name,
        "num_queries": len(data.query_ids),
        "num_documents": len(data.corpus_ids),
        "metrics": metrics,
        "timings_seconds": timings,
        "config": {
            "seed": args.seed,
            "max_queries": args.max_queries,
            "corpus_size": args.corpus_size,
            "batch_size": args.batch_size,
            "score_batch_size": args.score_batch_size,
            **backend.cache_config(args),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "query_id", "query", "first_relevant_rank", "recall_at_1", "recall_at_5",
            "recall_at_10", "mrr_at_10", "ndcg_at_5", "ndcg_at_10",
            "rank", "doc_id", "score", "is_relevant",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query_index, query_id in enumerate(data.query_ids):
            summary = per_query[query_index]
            relevant = data.qrels[query_id]
            for rank, doc_index in enumerate(
                ranking[query_index, : args.top_k_output], start=1
            ):
                doc_id = data.corpus_ids[int(doc_index)]
                writer.writerow(
                    {
                        **summary,
                        "query": data.queries[query_index],
                        "rank": rank,
                        "doc_id": doc_id,
                        "score": float(scores[query_index, int(doc_index)]),
                        "is_relevant": int(relevant.get(doc_id, 0) > 0),
                    }
                )
    return json_path, csv_path


def build_backend(args: argparse.Namespace) -> Any:
    if args.model == "dse":
        return DSEBackend(args)
    return ColPaliBackend(args)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.score_batch_size < 1:
        raise ValueError("Batch sizes must be positive integers.")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    print(f"Loading {args.model} backend...")
    backend_started = time.perf_counter()
    backend = build_backend(args)
    model_load_seconds = time.perf_counter() - backend_started

    for dataset_name in args.datasets:
        print(f"\nLoading benchmark: {dataset_name}")
        data = load_benchmark_data(
            dataset_name, args.split, args.max_queries, args.corpus_size, args.seed
        )
        print(f"Queries: {len(data.query_ids)} | Documents: {len(data.corpus_ids)}")

        cache_identity = {
            "dataset": dataset_name,
            "split": args.split,
            "queries": stable_digest(data.query_ids),
            "documents": stable_digest(data.corpus_ids),
            "backend": backend.cache_config(args),
        }
        cache_key = stable_digest([json.dumps(cache_identity, sort_keys=True)])
        cache_root = args.cache_dir / safe_name(dataset_name) / backend.name / cache_key

        print("Encoding documents (cached shards are reused)...")
        doc_embeddings, doc_seconds, _ = shard_encode(
            data.corpus_images,
            data.corpus_ids,
            backend.encode_documents,
            cache_root / "documents",
            args.batch_size,
            args.force_recompute,
        )
        print("Encoding queries (cached shards are reused)...")
        query_embeddings, query_seconds, _ = shard_encode(
            data.queries,
            data.query_ids,
            backend.encode_queries,
            cache_root / "queries",
            args.batch_size,
            args.force_recompute,
        )

        score_started = time.perf_counter()
        scores = backend.score(query_embeddings, doc_embeddings, args.score_batch_size)
        score_seconds = time.perf_counter() - score_started
        if tuple(scores.shape) != (len(data.query_ids), len(data.corpus_ids)):
            raise RuntimeError(
                f"Unexpected score matrix shape {tuple(scores.shape)}; expected "
                f"({len(data.query_ids)}, {len(data.corpus_ids)})."
            )

        metrics, per_query, ranking = compute_metrics(
            scores, data.query_ids, data.corpus_ids, data.qrels
        )
        timings = {
            "model_load": model_load_seconds,
            "document_encoding": doc_seconds,
            "query_encoding": query_seconds,
            "scoring": score_seconds,
            "document_seconds_per_item": doc_seconds / max(1, len(data.corpus_ids)),
            "query_seconds_per_item": query_seconds / max(1, len(data.query_ids)),
        }
        json_path, csv_path = write_outputs(
            args, backend, data, metrics, per_query, ranking, scores, timings
        )
        print(json.dumps(metrics, indent=2))
        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
