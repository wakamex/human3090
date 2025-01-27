# %%
import json
import sys

# deepseekr1qwen14q6kl.jsonl_results.jsonl  deepseekr1qwen1p5q6klfp1.jsonl_results.jsonl  deepseekr1qwen1p5q6klfp1t0p6.jsonl_results.jsonl  deepseekr1qwen1p5q6kl.jsonl_results.jsonl  deepseekr1qwen1p5q6klt0p6.jsonl_results.jsonl  deepseekr1qwen32q4km.jsonl_results.jsonl  deepseekr1qwen32q4kmt0p6fp1.jsonl_results.jsonl  deepseekr1qwen32q4kmt0p6.jsonl_results.jsonl
# FILES_OF_INTEREST = [
#     "deepseekr1qwen1p5q6kl.jsonl_results.jsonl",
#     "deepseekr1qwen1p5q6klt0p6.jsonl_results.jsonl",
#     "deepseekr1qwen1p5q6klfp1.jsonl_results.jsonl",
#     "deepseekr1qwen1p5q6klfp1t0p6.jsonl_results.jsonl",
#     "deepseekr1qwen14q6kl.jsonl_results.jsonl",
#     "deepseekr1qwen32q4km.jsonl_results.jsonl",
#     "deepseekr1qwen32q4kmt0p6.jsonl_results.jsonl",
#     "deepseekr1qwen32q4kmt0p6fp1.jsonl_results.jsonl",
# ]
# sys.argv = ["", FILES_OF_INTEREST[0]]

if len(sys.argv) < 2:
    print("Usage: python inspect_result.py <filename>")
    sys.exit(1)
else:
    print(f"Inspecting {sys.argv[1]}")
    file_to_inspect = sys.argv[1]

results_jsonl = []
with open(file_to_inspect, "r", encoding="utf-8") as f:
    results_jsonl.extend(json.loads(line) for line in f)

# %%
fails = []
difficulty_scores = {}
for result in results_jsonl:
    difficulty = result.get('difficulty', 'unknown')
    if difficulty not in difficulty_scores:
        difficulty_scores[difficulty] = {'passed': 0, 'total': 0}

    difficulty_scores[difficulty]['total'] += 1
    if result['passed']:
        difficulty_scores[difficulty]['passed'] += 1
    else:
        fails.append(result['task_id'].split("/")[-1])

print("Fails:", ", ".join(fails) if fails else "None")

# Print overall score
total_passed = sum(d['passed'] for d in difficulty_scores.values())
total_problems = sum(d['total'] for d in difficulty_scores.values())
print(f"\nOverall Score: {total_passed}/{total_problems}={total_passed/total_problems:.3f}")

# Print scores by difficulty
print("\nScores by difficulty:")
for difficulty, scores in sorted(difficulty_scores.items()):
    passed = scores['passed']
    total = scores['total']
    print(f"{difficulty}: {passed}/{total}={passed/total:.3f}")

# %%
# plot histogram of character length of each result
# import matplotlib.pyplot as plt
# char_lengths = [len(result["completion"]) for result in results_jsonl]
# h = plt.hist(char_lengths, bins=50)
# values = h[0]
# bins = h[1]
# bin_threshold = 15_000
# num_results_outside_top_bin = sum(values[bins[:-1] > bin_threshold])
# print(f"Number of results over {bin_threshold:,.0f} chars: {num_results_outside_top_bin}")
# print(f"Average char length: {sum(char_lengths)/len(char_lengths):,.0f}")

# %%