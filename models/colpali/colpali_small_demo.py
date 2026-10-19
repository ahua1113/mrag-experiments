from sentence_transformers import MultiVectorEncoder
from PIL import Image


# =========================
# 1. 加载模型
# =========================

print("Loading model...")

model = MultiVectorEncoder(
    "vidore/colpali-v1.3"
)

print("Model loaded!")


# =========================
# 2. 读取两张页面图片
# =========================

images = [
    Image.open(r"F:\AI\projects\mragr-experiments\dataset\docvqa\example\essay_pic.png").convert("RGB"),
    Image.open(r"F:\AI\projects\mragr-experiments\dataset\docvqa\example\revenue.png").convert("RGB"),
]

page_names = [
    "essay_pic.png",
    "revenue.png",
]


# =========================
# 3. 编码文档页面
# =========================

print("\nEncoding document pages...")

doc_embeddings = model.encode_document(
    images,
    batch_size=1,
)

print("\nDocument embedding shapes:")

for i, emb in enumerate(doc_embeddings):
    print(f"{page_names[i]}: {emb.shape}")


# =========================
# 4. 四组对比 Query
# =========================

queries = [
    "Which page contains a table about revenue?",

    "Which page contains the annual revenue and operating profit in 2025?",

    "哪一页包含营业收入和净利润等核心财务指标？",

    "哪一页包含归属于上市公司股东的净利润？",

    "哪一页讨论人工智能模型的训练方法？",
]


# =========================
# 5. 编码所有 Query
# =========================

print("\nEncoding queries...")

query_embeddings = model.encode_query(
    queries,
    batch_size=1,
)

print("\nQuery embedding shapes:")

for i, emb in enumerate(query_embeddings):
    print(f"Query {i + 1}: {emb.shape}")


# =========================
# 6. 计算相似度
# =========================

scores = model.similarity(
    query_embeddings,
    doc_embeddings,
)


# =========================
# 7. 输出每组实验结果
# =========================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for query_idx, query in enumerate(queries):

    print(f"\nQuery {query_idx + 1}:")
    print(query)

    print("\nScores:")

    for page_idx, page_name in enumerate(page_names):
        score = scores[query_idx, page_idx].item()

        print(
            f"  {page_name}: "
            f"{score:.4f}"
        )

    # 排序
    ranking = scores[query_idx].argsort(descending=True)

    print("\nRanking:")

    for rank, idx in enumerate(ranking, start=1):
        page_idx = idx.item()

        print(
            f"  Rank {rank}: "
            f"{page_names[page_idx]}, "
            f"score={scores[query_idx, page_idx].item():.4f}"
        )

    print("-" * 70)


# =========================
# 8. 汇总结果
# =========================

print("\nSUMMARY")
print("=" * 70)

header = (
    f"{'Query':<10}"
    f"{'page1':>12}"
    f"{'page2':>12}"
    f"{'Winner':>15}"
)

print(header)

for query_idx in range(len(queries)):

    score1 = scores[query_idx, 0].item()
    score2 = scores[query_idx, 1].item()

    winner = (
        page_names[0]
        if score1 > score2
        else page_names[1]
    )

    print(
        f"{'Query ' + str(query_idx + 1):<10}"
        f"{score1:>12.4f}"
        f"{score2:>12.4f}"
        f"{winner:>15}"
    )