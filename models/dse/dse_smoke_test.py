#!/usr/bin/env python3
"""Minimal, self-contained smoke test for MrLight DSE-Qwen2-2B-MRL.

The script creates two synthetic document screenshots, embeds them and two
queries, then computes the 2x2 cosine-similarity matrix.  A successful run
proves that the model, processor, Qwen-VL image path and CUDA inference path
work together; semantic top-1 accuracy is reported separately.
"""

from __future__ import annotations
import argparse
import gc
import json
import sys
import textwrap
import time
from pathlib import Path
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-page DSE retrieval smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="MrLight/dse-qwen2-2b-mrl-v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--max-image-patches", type=int, default=144)
    parser.add_argument("--gpu-memory", default="3.2GiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument("--work-dir", type=Path, default=Path("dse_smoke_output"))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never access Hugging Face; useful when --model is a local directory",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable NF4 loading; normally unsuitable for a 4 GB GPU",
    )
    return parser.parse_args()


from typing import Any


def import_dependencies() -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        import transformers
        from PIL import Image, ImageDraw, ImageFont
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2VLForConditionalGeneration,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install the DSE runtime first:\n"
            "  pip install -U torch torchvision transformers accelerate "
            "bitsandbytes qwen-vl-utils pillow"
        ) from exc

    return {
        "torch": torch,
        "F": functional,
        "transformers": transformers,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Qwen2VLForConditionalGeneration": Qwen2VLForConditionalGeneration,
    }


def load_font(image_font: Any, size: int, bold: bool = False) -> Any:
    candidates = (
        ("arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("arial.ttf", "DejaVuSans.ttf")
    )
    for name in candidates:
        try:
            return image_font.truetype(name, size)
        except OSError:
            pass
    return image_font.load_default()


def create_demo_pages(deps: dict[str, Any], output_dir: Path) -> list[Any]:
    Image = deps["Image"]
    ImageDraw = deps["ImageDraw"]
    ImageFont = deps["ImageFont"]
    output_dir.mkdir(parents=True, exist_ok=True)

    title_font = load_font(ImageFont, 38, bold=True)
    body_font = load_font(ImageFont, 29)
    small_font = load_font(ImageFont, 25)

    finance = Image.new("RGB", (672, 504), "white")
    draw = ImageDraw.Draw(finance)
    draw.text((35, 25), "ACME FINANCIAL RESULTS", fill="#102a43", font=title_font)
    draw.text((35, 85), "Annual revenue and operating profit", fill="#334e68", font=body_font)
    columns = [35, 225, 420, 635]
    rows = [145, 205, 265, 325]
    for x in columns:
        draw.line((x, rows[0], x, rows[-1]), fill="#486581", width=3)
    for y in rows:
        draw.line((columns[0], y, columns[-1], y), fill="#486581", width=3)
    table = (
        ("Year", "Revenue", "Operating profit"),
        ("2024", "$18.4M", "$3.1M"),
        ("2025", "$24.8M", "$5.6M"),
    )
    for row_index, row in enumerate(table):
        y = 160 + row_index * 60
        draw.text((50, y), row[0], fill="black", font=small_font)
        draw.text((245, y), row[1], fill="black", font=small_font)
        draw.text((440, y), row[2], fill="black", font=small_font)
    draw.text((35, 390), "Revenue increased by 34.8% in 2025.", fill="#1f7a4d", font=body_font)

    ai_page = Image.new("RGB", (672, 504), "white")
    draw = ImageDraw.Draw(ai_page)
    draw.text((35, 25), "TRAINING AN AI MODEL", fill="#3c096c", font=title_font)
    body = (
        "The model is trained in three stages: data preparation, supervised "
        "fine-tuning, and evaluation. Training uses image-text pairs and a "
        "contrastive objective. The optimizer updates the neural network "
        "weights over multiple epochs."
    )
    y = 100
    for line in textwrap.wrap(body, width=43):
        draw.text((40, y), line, fill="#222222", font=body_font)
        y += 48
    draw.rounded_rectangle((40, 365, 630, 455), radius=18, fill="#ede7f6")
    draw.text((65, 392), "DATA  ->  TRAIN  ->  EVALUATE", fill="#5a189a", font=body_font)

    pages = [finance, ai_page]
    names = ["finance_page.png", "ai_training_page.png"]
    for image, name in zip(pages, names):
        image.save(output_dir / name)
    return pages


def clear_memory(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    torch = deps["torch"]
    F = deps["F"]

    if args.embedding_dim <= 0 or args.embedding_dim > 1536:
        raise SystemExit("--embedding-dim must be between 1 and 1536.")
    if args.max_image_patches <= 0:
        raise SystemExit("--max-image-patches must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Check the NVIDIA driver and CUDA-enabled "
            "PyTorch installation."
        )
    cuda_index = 0
    if args.device.startswith("cuda:"):
        cuda_index = int(args.device.split(":", maxsplit=1)[1])

    args.work_dir.mkdir(parents=True, exist_ok=True)
    pages = create_demo_pages(deps, args.work_dir)
    page_names = ["finance_page.png", "ai_training_page.png"]
    queries = [
        "Which page contains annual revenue and operating profit for 2025?",
        "Which page explains how an artificial intelligence model is trained?",
    ]

    print("=" * 72)
    print("DSE SMOKE TEST")
    print("=" * 72)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {deps['transformers'].__version__}")
    print(f"CUDA:         {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(cuda_index)}")
        total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3
        print(f"VRAM:         {total_gib:.2f} GiB")
    print(f"Model:        {args.model}")
    print(f"4-bit NF4:    {not args.no_4bit}")
    print(f"Dimension:    {args.embedding_dim}")
    print(f"Max patches:  {args.max_image_patches}")

    max_pixels = args.max_image_patches * 28 * 28
    processor = deps["AutoProcessor"].from_pretrained(
        args.model,
        min_pixels=28 * 28,
        max_pixels=max_pixels,
        local_files_only=args.local_files_only,
    )
    processor.tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if not args.no_4bit:
        model_kwargs.update(
            quantization_config=deps["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
            max_memory={cuda_index: args.gpu_memory, "cpu": args.cpu_memory},
            offload_folder=str(args.work_dir / "offload"),
        )
    else:
        model_kwargs["device_map"] = {"": args.device}

    print("\nLoading model...")
    load_started = time.perf_counter()
    model = deps["Qwen2VLForConditionalGeneration"].from_pretrained(
        args.model, **model_kwargs
    ).eval()
    model.padding_side = "left"
    load_seconds = time.perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.2f} seconds.")
    if hasattr(model, "hf_device_map"):
        print("Device map:")
        for module_name, device in model.hf_device_map.items():
            print(f"  {module_name or '<root>'}: {device}")

    def encode(item: Any, is_query: bool) -> Any:
        if is_query:
            content = [
                {
                    "type": "image",
                    "image": deps["Image"].new("RGB", (28, 28)),
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
        message = [{"role": "user", "content": content}]
        prompt = (
            processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            + "<|endoftext|>"
        )
        image_inputs, video_inputs = deps["process_vision_info"](message)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
            return_tensors="pt",
        )
        inputs = {
            key: value.to(args.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        cache_position = torch.arange(1, device=inputs["input_ids"].device)
        model_inputs = model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False
        )
        with torch.inference_mode():
            output = model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
            )
        embedding = output.hidden_states[-1][:, -1, : args.embedding_dim]
        embedding = F.normalize(embedding, p=2, dim=-1)[0].float().cpu()
        del output, model_inputs, inputs
        clear_memory(torch)
        return embedding

    document_embeddings = []
    document_times = []
    print("\nEncoding two document pages...")
    for name, page in zip(page_names, pages):
        started = time.perf_counter()
        embedding = encode(page, is_query=False)
        elapsed = time.perf_counter() - started
        document_embeddings.append(embedding)
        document_times.append(elapsed)
        print(f"  {name:<24} shape={tuple(embedding.shape)} time={elapsed:.2f}s")

    query_embeddings = []
    query_times = []
    print("\nEncoding two queries...")
    for query in queries:
        started = time.perf_counter()
        embedding = encode(query, is_query=True)
        elapsed = time.perf_counter() - started
        query_embeddings.append(embedding)
        query_times.append(elapsed)
        print(f"  shape={tuple(embedding.shape)} time={elapsed:.2f}s | {query}")

    documents = torch.stack(document_embeddings)
    query_matrix = torch.stack(query_embeddings)
    scores = query_matrix @ documents.T

    finite = bool(torch.isfinite(documents).all() and torch.isfinite(query_matrix).all())
    correct_shape = tuple(scores.shape) == (2, 2)
    doc_norms = torch.linalg.vector_norm(documents, dim=1)
    query_norms = torch.linalg.vector_norm(query_matrix, dim=1)
    normalized = bool(
        torch.allclose(doc_norms, torch.ones_like(doc_norms), atol=1e-3)
        and torch.allclose(query_norms, torch.ones_like(query_norms), atol=1e-3)
    )
    technical_pass = finite and correct_shape and normalized
    winners = torch.argmax(scores, dim=1).tolist()
    semantic_pass = winners == [0, 1]

    print("\nSimilarity matrix (rows=queries, columns=pages):")
    print(f"{'':16}{page_names[0]:>24}{page_names[1]:>24}")
    for index, query in enumerate(queries):
        print(
            f"Query {index + 1:<10}"
            f"{scores[index, 0].item():>24.6f}"
            f"{scores[index, 1].item():>24.6f}"
        )
        print(f"  top-1: {page_names[winners[index]]} | {query}")

    report = {
        "technical_pass": technical_pass,
        "semantic_pass": semantic_pass,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "max_image_patches": args.max_image_patches,
        "four_bit": not args.no_4bit,
        "scores": scores.tolist(),
        "document_seconds": document_times,
        "query_seconds": query_times,
        "model_load_seconds": load_seconds,
        "page_names": page_names,
        "queries": queries,
    }
    report_path = args.work_dir / "smoke_test_result.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"TECHNICAL HEALTH CHECK: {'PASS' if technical_pass else 'FAIL'}")
    print(f"SEMANTIC TOP-1 CHECK:   {'PASS' if semantic_pass else 'WARN'}")
    print(f"Result: {report_path.resolve()}")
    print("=" * 72)
    if not technical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
