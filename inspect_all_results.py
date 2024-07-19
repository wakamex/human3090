# %%
import json
import os

import pandas as pd
from human_eval.data import read_problems

problems = read_problems()

# %%
all_result_files = [f for f in os.listdir() if f.endswith("_results.jsonl")]

# build records to turn into a dataframe
records = []
for result_file in all_result_files:
    # remove .jsonl_results.jsonl
    model = result_file[:-len(".jsonl_results.jsonl")]
    results_jsonl = []
    with open(result_file, "r", encoding="utf-8") as f:
        results_jsonl.extend(json.loads(line) for line in f)

    for result in results_jsonl:
        task_id = result["task_id"].split("/")[-1]
        passed = result["passed"]
        records.append((model, task_id, passed))

# %%
# build dataframe
df = pd.DataFrame(records, columns=["model", "task_id", "passed"])
display(df.head())

# %%
# tabulate task_id by pass rate
scores = df.groupby("task_id").passed.mean().sort_values()
# %%
print("questions no one gets right")
print(",".join(scores[scores < 1].index))
# %%
print("questions everyone gets right")
print(",".join(scores[scores >= 1].index))
# %%
print("toughest questions that at least one person gets right")
scores[scores > 0].sort_values(ascending=True)
hardest_score = scores[scores > 0].sort_values(ascending=True).iloc[0]
print(f"{hardest_score=}")
hardest_problems = scores[scores == hardest_score].index
print(",".join(hardest_problems))
# %%
for task_id, problem in problems.items():
    short_id = task_id.split("/")[-1]
    if task_id.split("/")[-1] in hardest_problems:
        print(task_id)
        model_that_got_it_right = df[(df.task_id == short_id) & (df.passed)].model.iloc[0]
        print(f"Model that got it right: {model_that_got_it_right}")
        print(problem["prompt"])

# %%