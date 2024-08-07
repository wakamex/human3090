import json
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_result.py <filename>")
    sys.exit(1)
else:
    print(f"Inspecting {sys.argv[1]}")
    file_to_inspect = sys.argv[1]

results_jsonl = []
with open(file_to_inspect, "r") as f:
    results_jsonl.extend(json.loads(line) for line in f)

# %%
score = 0
fails = []
for result in results_jsonl:
    if result["passed"] is True:
        score += 1
    else:
        fails.append(result["task_id"].split("/")[-1])
print(f"Fails: {','.join(fails)}")
print(f"Score: {score}/{len(results_jsonl)}={score/len(results_jsonl):.3f}")
