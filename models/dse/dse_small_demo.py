import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "MrLight/dse-qwen2-2b-mrl-v1"

DEVICE = "cuda:0"

# ------------------------------------------------------------
# RTX 3050 Ti 4GB
# ------------------------------------------------------------

# DSE original representation is 1536-d.
# MRL allows us to use a smaller dimension.
EMBEDDING_DIM = 512

# 144 visual patches
#
# 144 * 28 * 28 = 112896 pixels
#
# roughly equivalent to 336 x 336 for a square image.
MAX_IMAGE_PATCHES = 144

MAX_PIXELS = MAX_IMAGE_PATCHES * 28 * 28
MIN_PIXELS = 28 * 28


# ------------------------------------------------------------
# Local document pages
# ------------------------------------------------------------
#
# 修改成你自己的图片路径
#

IMAGE_PATHS = [
    Path(
        r"F:\AI\projects\mragr-experiments"
        r"\dataset\docvqa\example\essay_pic.png"
    ),
    Path(
        r"F:\AI\projects\mragr-experiments"
        r"\dataset\docvqa\example\revenue.png"
    ),
]

PAGE_NAMES = [
    "essay_pic.png",
    "revenue.png",
]


# ------------------------------------------------------------
# Test queries
# ------------------------------------------------------------

QUERIES = [
    "Which page contains a table about revenue?",

    "Which page contains the annual revenue and "
    "operating profit in 2025?",

    "Which page contains information about "
    "artificial intelligence models?",

    "Which page discusses the training method "
    "of an AI model?",

    "Which page contains financial information "
    "about the company?",
]


# ============================================================
# 2. GPU information
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Please check your NVIDIA / PyTorch installation."
    )


print("=" * 70)
print("GPU INFORMATION")
print("=" * 70)

gpu_name = torch.cuda.get_device_name(0)

total_vram = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"GPU: {gpu_name}")
print(f"VRAM: {total_vram:.2f} GB")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")


# ============================================================
# 3. 4-bit quantization
# ============================================================

print("\n" + "=" * 70)
print("4-BIT QUANTIZATION")
print("=" * 70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # NF4
    bnb_4bit_quant_type="nf4",

    # Nested quantization
    bnb_4bit_use_double_quant=True,

    # RTX 3050 Ti uses FP16 here
    bnb_4bit_compute_dtype=torch.float16,
)

print("load_in_4bit: True")
print("quant_type: nf4")
print("double_quant: True")
print("compute_dtype: float16")


# ============================================================
# 4. Load processor
# ============================================================

print("\n" + "=" * 70)
print("LOADING PROCESSOR")
print("=" * 70)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,

    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

processor.tokenizer.padding_side = "left"


# ============================================================
# 5. Load DSE model
# ============================================================

print("\n" + "=" * 70)
print("LOADING DSE MODEL")
print("=" * 70)

load_start = time.perf_counter()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,

    # 4-bit quantization
    quantization_config=bnb_config,

    # Non-quantized modules use FP16
    torch_dtype=torch.float16,

    # Let Accelerate place modules according to memory.
    device_map="auto",

    # Keep some GPU memory for activations.
    max_memory={
        0: "3.2GiB",
        "cpu": "12GiB",
    },

    low_cpu_mem_usage=True,
).eval()

load_time = time.perf_counter() - load_start

model.padding_side = "left"

print(
    f"Model loaded in {load_time:.2f} seconds."
)


# ============================================================
# 6. Print device map
# ============================================================

if hasattr(model, "hf_device_map"):

    print("\nDevice map:")

    for name, device in model.hf_device_map.items():

        print(
            f"  {name:<50} -> {device}"
        )


# ============================================================
# 7. Model memory footprint
# ============================================================

try:

    footprint = (
        model.get_memory_footprint()
        / 1024**3
    )

    print(
        f"\nModel memory footprint: "
        f"{footprint:.2f} GB"
    )

except Exception:

    print(
        "\nModel memory footprint: "
        "unavailable"
    )


# ============================================================
# 8. Helper: clear GPU memory
# ============================================================

def clear_gpu_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.synchronize()


# ============================================================
# 9. Helper: DSE embedding
# ============================================================

def get_embedding(
    hidden_states,
    dimension=EMBEDDING_DIM,
):
    """
    DSE representation:

        last hidden state
            ↓
        last token
            ↓
        truncate dimension
            ↓
        L2 normalization

    Output:
        [batch_size, dimension]
    """

    reps = hidden_states[:, -1]

    reps = reps[:, :dimension]

    reps = F.normalize(
        reps,
        p=2,
        dim=-1,
    )

    return reps


# ============================================================
# 10. Encode Query
# ============================================================

def encode_query(query):

    # DSE official implementation uses
    # a tiny dummy image for text queries.
    dummy_image = Image.new(
        "RGB",
        (28, 28),
    )

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": dummy_image,

                    # Keep the dummy image tiny.
                    "resized_height": 1,
                    "resized_width": 1,
                },
                {
                    "type": "text",
                    "text": f"Query: {query}",
                },
            ],
        }
    ]

    text = (
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        + "<|endoftext|>"
    )

    image_inputs, video_inputs = (
        process_vision_info(message)
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding="longest",
        return_tensors="pt",
    )

    # Move tensors to the model's execution device.
    inputs = {
        k: v.to(DEVICE)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    # Follow the official DSE inference flow.
    cache_position = torch.arange(
        0,
        1,
        device=inputs["input_ids"].device,
    )

    model_inputs = (
        model.prepare_inputs_for_generation(
            **inputs,
            cache_position=cache_position,
            use_cache=False,
        )
    )

    with torch.inference_mode():

        output = model(
            **model_inputs,

            return_dict=True,

            output_hidden_states=True,

            use_cache=False,
        )

    embedding = get_embedding(
        output.hidden_states[-1],
        EMBEDDING_DIM,
    )

    # Move embedding to CPU immediately.
    embedding = embedding[0].float().cpu()

    del output
    del model_inputs
    del inputs

    clear_gpu_memory()

    return embedding


# ============================================================
# 11. Encode Document
# ============================================================

def encode_document(image):

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "What is shown in this image?",
                },
            ],
        }
    ]

    text = (
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        + "<|endoftext|>"
    )

    image_inputs, video_inputs = (
        process_vision_info(message)
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding="longest",
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    cache_position = torch.arange(
        0,
        1,
        device=inputs["input_ids"].device,
    )

    model_inputs = (
        model.prepare_inputs_for_generation(
            **inputs,
            cache_position=cache_position,
            use_cache=False,
        )
    )

    with torch.inference_mode():

        output = model(
            **model_inputs,

            return_dict=True,

            output_hidden_states=True,

            use_cache=False,
        )

    embedding = get_embedding(
        output.hidden_states[-1],
        EMBEDDING_DIM,
    )

    embedding = embedding[0].float().cpu()

    del output
    del model_inputs
    del inputs

    clear_gpu_memory()

    return embedding


# ============================================================
# 12. Encode Documents
# ============================================================

print("\n" + "=" * 70)
print("ENCODING DOCUMENTS")
print("=" * 70)

document_embeddings = []

for idx, image_path in enumerate(IMAGE_PATHS):

    print(
        f"\n[{idx + 1}/{len(IMAGE_PATHS)}] "
        f"{image_path.name}"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"Original image size: "
        f"{image.size}"
    )

    start = time.perf_counter()

    embedding = encode_document(
        image
    )

    elapsed = (
        time.perf_counter() - start
    )

    document_embeddings.append(
        embedding
    )

    print(
        f"Embedding shape: "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Encoding time: "
        f"{elapsed:.2f} sec"
    )

    del image


# Stack on CPU
document_embeddings = torch.stack(
    document_embeddings
)

print(
    "\nDocument embeddings:"
)

print(
    document_embeddings.shape
)


# ============================================================
# 13. Encode Queries
# ============================================================

print("\n" + "=" * 70)
print("ENCODING QUERIES")
print("=" * 70)

query_embeddings = []

query_times = []

for idx, query in enumerate(QUERIES):

    print(
        f"\n[{idx + 1}/{len(QUERIES)}]"
    )

    print(
        f"Query: {query}"
    )

    start = time.perf_counter()

    embedding = encode_query(
        query
    )

    elapsed = (
        time.perf_counter() - start
    )

    query_embeddings.append(
        embedding
    )

    query_times.append(
        elapsed
    )

    print(
        f"Embedding shape: "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Encoding time: "
        f"{elapsed:.2f} sec"
    )


query_embeddings = torch.stack(
    query_embeddings
)

print(
    "\nQuery embeddings:"
)

print(
    query_embeddings.shape
)


# ============================================================
# 14. Similarity
# ============================================================

print("\n" + "=" * 70)
print("SIMILARITY")
print("=" * 70)

# All embeddings have already been normalized.
#
# Therefore:
#
# cosine(q, d) = q · d
#

scores = (
    query_embeddings
    @ document_embeddings.T
)


# ============================================================
# 15. Ranking
# ============================================================

print("\n" + "=" * 70)
print("RANKING")
print("=" * 70)

for query_idx, query in enumerate(QUERIES):

    print(
        f"\nQuery {query_idx + 1}:"
    )

    print(query)

    ranking = torch.argsort(
        scores[query_idx],
        descending=True,
    )

    for rank, doc_idx in enumerate(
        ranking,
        start=1,
    ):

        doc_idx = doc_idx.item()

        score = scores[
            query_idx,
            doc_idx,
        ].item()

        print(
            f"  Rank {rank}: "
            f"{PAGE_NAMES[doc_idx]:<25} "
            f"score={score:.4f}"
        )


# ============================================================
# 16. Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"{'Query':<10}"
    f"{'Page 1':>15}"
    f"{'Page 2':>15}"
    f"{'Winner':>25}"
)

print("-" * 70)

for query_idx in range(
    len(QUERIES)
):

    score1 = scores[
        query_idx,
        0,
    ].item()

    score2 = scores[
        query_idx,
        1,
    ].item()

    winner = (
        PAGE_NAMES[0]
        if score1 >= score2
        else PAGE_NAMES[1]
    )

    print(
        f"{'Query ' + str(query_idx + 1):<10}"
        f"{score1:>15.4f}"
        f"{score2:>15.4f}"
        f"{winner:>25}"
    )


# ============================================================
# 17. Embedding information
# ============================================================

print("\n" + "=" * 70)
print("EMBEDDING INFORMATION")
print("=" * 70)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Representation dimension: "
    f"{EMBEDDING_DIM}"
)

print(
    f"Document embedding shape: "
    f"{tuple(document_embeddings.shape)}"
)

print(
    f"Query embedding shape: "
    f"{tuple(query_embeddings.shape)}"
)

print(
    f"Image patches limit: "
    f"{MAX_IMAGE_PATCHES}"
)

print(
    f"Max pixels: "
    f"{MAX_PIXELS}"
)


# ============================================================
# 18. Final GPU memory
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated(0)
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / 1024**3
    )

    print("\n" + "=" * 70)
    print("FINAL GPU MEMORY")
    print("=" * 70)

    print(
        f"Allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"Reserved: "
        f"{reserved:.2f} GB"
    )


print("\nDSE demo completed.")