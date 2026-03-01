#!/usr/bin/env python3
"""
Step 2: Metrics Evaluation
Runs Ragas metrics strictly on pre-computed evaluation inputs.
Saves to `data/processed/*_evaluation_metrics.parquet`.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

print("=" * 80)
print("RAGAS METRICS EVALUATION (Step 2 of 2)")
print("=" * 80)

print("\nLooking for evaluation inputs...")
input_files = list(OUT_DIR.glob("*_evaluation_inputs.parquet"))
if not input_files:
    raise FileNotFoundError(
        "No input parquets found! Run Step 1 evaluating_inference.py first."
    )

openai_client = OpenAI()
evaluator_llm = llm_factory("gpt-4.1-mini", client=openai_client)
evaluator_emb = RagasOpenAIEmbeddings(
    client=openai_client, model="text-embedding-3-small"
)
run_cfg = RunConfig(timeout=360)
results = {}

for input_file in input_files:
    name = input_file.name.replace("_evaluation_inputs.parquet", "")
    print(f"\n🔍 Evaluating {name}...")

    df = pd.read_parquet(input_file)
    eval_ds = EvaluationDataset.from_pandas(df)

    res = evaluate(
        dataset=eval_ds,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
            ContextPrecision(llm=evaluator_llm, name="context_precision"),
            ContextRecall(llm=evaluator_llm),
        ],
        run_config=run_cfg,
    )

    results[name] = res
    print(f"   ✓ Evaluation complete for {name}")

    det_file = OUT_DIR / f"{name}_evaluation_metrics.parquet"
    res.to_pandas().to_parquet(det_file, compression="zstd", index=False)
    print(f"   💾 Saved evaluation metrics: {det_file.name}")

print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS")
print("=" * 80)

comp = []
for name, res in results.items():
    rdf = res.to_pandas()
    row = {
        "Retriever": name.replace("_", " ").title(),
        "Faithfulness": rdf["faithfulness"].mean(),
        "Answer Relevancy": rdf["answer_relevancy"].mean(),
        "Context Precision": rdf["context_precision"].mean(),
        "Context Recall": rdf["context_recall"].mean(),
    }
    row["Average"] = (
        row["Faithfulness"]
        + row["Answer Relevancy"]
        + row["Context Precision"]
        + row["Context Recall"]
    ) / 4
    comp.append(row)

comp_df = (
    pd.DataFrame(comp).sort_values("Average", ascending=False).reset_index(drop=True)
)
if comp_df.empty:
    raise ValueError(
        "No evaluation results to compare — check that input parquet files contain data"
    )
comp_parquet = OUT_DIR / "comparative_ragas_results.parquet"
comp_df.to_parquet(comp_parquet, compression="zstd", index=False)
print(f"\n💾 Saved comparative results: {comp_parquet.name}")

print("\n" + "=" * 80)
print("COMPARATIVE RAGAS RESULTS")
print("=" * 80)
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n✅ METRICS EVALUATION COMPLETE")
