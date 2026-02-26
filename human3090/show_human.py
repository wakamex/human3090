import sys

from human3090.human_eval.data import read_problems


def main():
    problems = read_problems()

    human_to_inspect = sys.argv[1]

    for task_id, problem in problems.items():
        if task_id.split("/")[-1] == human_to_inspect:
            print(problem["prompt"])
            break

if __name__ == "__main__":
    main()
