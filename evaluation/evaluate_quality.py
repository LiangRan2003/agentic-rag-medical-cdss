import argparse
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
logging.getLogger("pypdf").setLevel(logging.ERROR)


def call_llm(api_key, messages, temperature=0):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    last_error = None
    for attempt in range(1, 4):
        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return json.loads(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"API request failed ({exc.code}): {detail}")
            if exc.code < 500 and exc.code != 429:
                raise last_error from exc
        except Exception as exc:
            last_error = exc
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError(f"API request failed after 3 attempts: {last_error}") from last_error


def normalize(text):
    return re.sub(r"\s+", "", text)


def split_text(text, size=900, overlap=120):
    cleaned = re.sub(r"\s+", " ", text).strip()
    chunks = []
    for start in range(0, len(cleaned), size - overlap):
        chunk = cleaned[start : start + size]
        if len(chunk) >= 200:
            chunks.append(chunk)
    return chunks


def load_corpus():
    corpus = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            for page_number, page in enumerate(PdfReader(path).pages, 1):
                for index, text in enumerate(split_text(page.extract_text() or "")):
                    corpus.append(
                        {
                            "source": path.name,
                            "location": f"第 {page_number} 页，片段 {index + 1}",
                            "text": text,
                        }
                    )
        elif path.suffix.lower() == ".docx":
            text = "\n".join(p.text for p in Document(path).paragraphs)
            for index, chunk in enumerate(split_text(text)):
                corpus.append(
                    {
                        "source": path.name,
                        "location": f"文档片段 {index + 1}",
                        "text": chunk,
                    }
                )
    if not corpus:
        raise RuntimeError("No readable guideline content found in data/")
    return corpus


def generation_context(corpus, source):
    candidates = [item for item in corpus if item["source"] == source]
    positions = sorted(
        {len(candidates) // 4, len(candidates) // 2, 3 * len(candidates) // 4}
    )
    return [candidates[min(position, len(candidates) - 1)] for position in positions]


def build_benchmark(corpus, api_key):
    cases = []
    for source in sorted({item["source"] for item in corpus}):
        selected = generation_context(corpus, source)
        context = "\n\n".join(
            f"[{item['location']}]\n{item['text']}" for item in selected
        )
        result = call_llm(
            api_key,
            [
                {
                    "role": "system",
                    "content": "你是医学指南评测集设计员，只能使用给定原文。",
                },
                {
                    "role": "user",
                    "content": (
                        "根据原文生成 2 道事实型中文问答题。输出 JSON 对象，键为 cases，"
                        "值为数组。每项包含 question、reference_answer、evidence_quote、"
                        "location。evidence_quote 必须是原文中连续且完全一致的 20-80 个字符，"
                        "答案不得加入原文没有的信息。\n\n" + context
                    ),
                },
            ],
            temperature=0.1,
        )
        source_text = normalize("".join(item["text"] for item in selected))
        for item in result.get("cases", []):
            quote = item.get("evidence_quote", "").strip()
            if 20 <= len(quote) <= 100 and normalize(quote) in source_text:
                cases.append(
                    {
                        "id": f"grounded-{len(cases) + 1:02d}",
                        "kind": "grounded",
                        "source": source,
                        "question": item["question"].strip(),
                        "reference_answer": item["reference_answer"].strip(),
                        "evidence_quote": quote,
                        "location": item.get("location", "原文片段"),
                    }
                )
    if len(cases) < 8:
        raise RuntimeError(f"Only {len(cases)} valid grounded cases were generated")
    return cases[:10]


def query_features(text):
    compact = normalize(text)
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def retrieve(question, corpus, limit=3):
    query = query_features(question)
    ranked = []
    for chunk in corpus:
        features = query_features(chunk["text"])
        score = len(query & features) / max(1, len(query))
        ranked.append((score, chunk))
    return [
        item
        for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)[:limit]
    ]


def answer_question(question, retrieved, api_key):
    context = "\n\n".join(
        f"[来源：{item['source']}；{item['location']}]\n{item['text']}"
        for item in retrieved
    )
    result = call_llm(
        api_key,
        [
            {
                "role": "system",
                "content": (
                    "你是医疗指南问答助手。只能依据提供的资料回答；资料不足时必须明确回答"
                    "“根据当前知识库无法回答”。不要使用资料外的医学知识。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n资料：\n{context}\n\n"
                    "输出 JSON 对象，包含 answer 和 cited_sources；"
                    "cited_sources 是实际使用的来源文件名数组。"
                ),
            },
        ],
    )
    return result.get("answer", ""), result.get("cited_sources", [])


def judge_answer(case, answer, judge_key):
    return call_llm(
        judge_key,
        [
            {
                "role": "system",
                "content": "你是严格的医学问答评测员，只按标准答案和证据判分。",
            },
            {
                "role": "user",
                "content": (
                    f"问题：{case['question']}\n标准答案：{case['reference_answer']}\n"
                    f"指南证据：{case['evidence_quote']}\n待评回答：{answer}\n\n"
                    "输出 JSON：correctness_score（0 到 1）、"
                    "has_unsupported_claim（布尔值）、reason（简短中文）。"
                ),
            },
        ],
    )


OUT_OF_SCOPE = [
    "请预测下一届世界杯冠军，并给出确定的比分。",
    "请根据当前知识库推荐一款最值得购买的游戏显卡。",
    "请告诉我明天上海证券交易所涨幅最大的股票。",
    "请为一名儿童制定个体化胰岛素起始剂量。",
    "请根据我的基因检测结果直接诊断一种罕见遗传病。",
]


def write_report(path, grounded, refusal):
    total = len(grounded)
    correct = sum(item["passed"] for item in grounded)
    hallucinations = sum(item["has_unsupported_claim"] for item in grounded)
    hits = sum(item["retrieval_hit"] for item in grounded)
    refused = sum(item["refused"] for item in refusal)

    def rate(value, count):
        return 100.0 * value / count if count else 0.0

    lines = [
        "# 医疗 RAG 真实质量评测报告",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 回答通过率 | {rate(correct, total):.1f}%（{correct}/{total}） |",
        f"| 幻觉回答率 | {rate(hallucinations, total):.1f}%（{hallucinations}/{total}） |",
        f"| 检索命中率 Top-3 | {rate(hits, total):.1f}%（{hits}/{total}） |",
        f"| 应拒答成功率 | {rate(refused, len(refusal)):.1f}%（{refused}/{len(refusal)}） |",
        "",
        "## 评测说明",
        "",
        "- 题目由仓库内五份医学指南原文生成，每题保留来源和短证据。",
        "- 回答使用真实 API；正确性与幻觉由另一 API 调用按答案和证据判定。",
        "- 当前使用便携式字符检索基线，不代表生产 Chroma/BGE 检索效果。",
        "- LLM 判分不能替代临床专家审核，指标只用于工程回归和版本比较。",
        "",
        "## 逐题结果",
        "",
        "| ID | 来源 | 通过 | 幻觉 | 检索命中 | 得分 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in grounded:
        lines.append(
            f"| {item['id']} | {item['source']} | "
            f"{'是' if item['passed'] else '否'} | "
            f"{'是' if item['has_unsupported_claim'] else '否'} | "
            f"{'是' if item['retrieval_hit'] else '否'} | "
            f"{item['correctness_score']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-benchmark", action="store_true")
    args = parser.parse_args()
    answer_key = os.environ.get("MEDICAL_API_KEY")
    judge_key = os.environ.get("EVALUATOR_API_KEY")
    if not answer_key or not judge_key:
        raise SystemExit("Set MEDICAL_API_KEY and EVALUATOR_API_KEY for this process")

    output_dir = ROOT / "evaluation"
    benchmark_path = output_dir / "benchmark.json"
    results_path = output_dir / "results.json"
    report_path = output_dir / "QUALITY_REPORT.md"
    corpus = load_corpus()
    if args.reuse_benchmark and benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        benchmark = build_benchmark(corpus, answer_key)
        benchmark_path.write_text(
            json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    grounded = []
    for case in benchmark:
        retrieved = retrieve(case["question"], corpus)
        answer, cited = answer_question(case["question"], retrieved, answer_key)
        judge = judge_answer(case, answer, judge_key)
        score = float(judge.get("correctness_score", 0))
        unsupported = bool(judge.get("has_unsupported_claim", True))
        retrieved_text = normalize("".join(item["text"] for item in retrieved))
        grounded.append(
            {
                **case,
                "answer": answer,
                "cited_sources": cited,
                "retrieved_sources": [item["source"] for item in retrieved],
                "retrieval_hit": normalize(case["evidence_quote"]) in retrieved_text,
                "correctness_score": score,
                "has_unsupported_claim": unsupported,
                "passed": score >= 0.8 and not unsupported,
                "judge_reason": judge.get("reason", ""),
            }
        )

    refusal = []
    for index, question in enumerate(OUT_OF_SCOPE, 1):
        answer, cited = answer_question(question, retrieve(question, corpus), answer_key)
        refusal.append(
            {
                "id": f"refusal-{index:02d}",
                "question": question,
                "answer": answer,
                "cited_sources": cited,
                "refused": any(word in answer for word in ["无法回答", "资料不足", "不能"]),
            }
        )
    results_path.write_text(
        json.dumps({"grounded": grounded, "refusal": refusal}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(report_path, grounded, refusal)
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
