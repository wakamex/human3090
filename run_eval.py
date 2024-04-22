import json
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from human_eval.data import read_problems
from openai import OpenAI
from pydantic_core import ValidationError
from together import Together

# pylint: disable=redefined-outer-name, line-too-long, missing-module-docstring, invalid-name, import-outside-toplevel
# ruff: noqa: E501

load_dotenv(".env")
OPENAI_KEY = os.getenv("OPENAI_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_KEY")


def clear_output_interactive():
    """Clear output in an interactive environment."""
    from IPython.display import clear_output

    clear_output(wait=True)


def clear_output_non_interactive():
    """Clear output in a non-interactive environment."""
    if sys.platform == "win32":  # For Windows
        os.system("cls")
    else:  # For macOS and Linux
        os.system("clear")


def clear_output_robust():
    """Clear output irrespective of OS, and whether running interactively or non-interactively."""
    if "ipykernel" in sys.modules:
        clear_output_interactive()
    else:
        clear_output_non_interactive()


def sanitize_answer(raw_answer):
    """Sanitize the answer to remove unwanted parts of the code, like comments."""
    code_started = True
    answer = []
    for line in raw_answer.split("\n"):
        # valid start of line that we want to include
        if line.lstrip(" ").startswith("def") or line.lstrip(" ").startswith("import"):
            code_started = True
            answer.append(line)
        # marks start of code, but we don't want to include it
        elif line.startswith("```python"):
            code_started = True
        # don't include anything after the code block
        elif line.startswith("```"):
            break
        elif not line.startswith(" ") and not line.startswith("\t") and line != "" and line != "\n":
            code_started = False
        elif code_started:
            answer.append(line)
    return "\n".join(answer)


def parse_completion_stream(completion_stream, prompt, task_id, end_after_n_codeblocks=None, framework="ai"):
    """Parse a completion stream and return the response."""
    response = []
    finished = False
    while not finished:
        try:
            text = next(completion_stream).choices[0].delta.content if framework == "ai" else next(completion_stream)
            if text=='':
                finished = True
            if text:
                response.append(text)
                clear_output_robust()
                if task_id:
                    print(f"{task_id}\n")
                print(prompt, flush=True)
                print("".join(response), flush=True)
                if end_after_n_codeblocks:
                    num_code_blocks = 0
                    for line in "".join(response).splitlines():
                        if line.lstrip(" ").startswith("```"):
                            num_code_blocks += 1
                            if num_code_blocks == end_after_n_codeblocks:
                                finished = True
                        if "</|im_end|>" in line:
                            finished = True
        except (StopIteration, ValidationError):
            finished = True
    return "".join(response)


def hf(prompt, model, temperature=0.8, task_id=None, end_after_n_codeblocks=None):
    client = InferenceClient(model=model, token=HUGGINGFACE_KEY)
    completion_stream = client.text_generation(prompt=prompt, stream=True, max_new_tokens=1_000, temperature=temperature)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id, end_after_n_codeblocks=end_after_n_codeblocks, framework="hf")

def make_kwargs(temperature=0.8, frequency_penalty=None, presence_penalty=None):
    kwargs = {"temperature": temperature}
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    return kwargs

def together(prompt, model, system=None, task_id=None, temperature=0.8, frequency_penalty=None, presence_penalty=None):
    client = Together(api_key=TOGETHER_KEY)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    kwargs = make_kwargs(temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    # response = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=1_000, **kwargs)
    # print(response.choices[0].message.content)
    completion_stream = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=1_000, **kwargs)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id)

def ai(prompt, system=None, url="http://127.0.0.1:8081/v1", model="llama!", key="na", temperature=0.8, frequency_penalty=None, presence_penalty=None, task_id=None):
    client = OpenAI(base_url=url, api_key=key)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    kwargs = make_kwargs(temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    completion_stream = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=1_000, **kwargs)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id)

problems = read_problems()
keys = list(problems.keys())
start_problem = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # start at 1 normally, or higher if continuing a previous run
subset = {key: problems[key] for key in keys[start_problem-1:]}
for task_id in subset:
    raw_prompt = problems[task_id]["prompt"]
    # deepseek (131/164=0.799) old run
    # ./build/bin/server -ngl 63 -m ./models/deepseek-coder-33b-instruct.Q4_K_M.gguf -c 2048
    # temperature = 0.0
    # model = "deepseek"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function. After the function, include an EOS token.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # gpt4 (104/164=0.634) instruction-style
    # url = "https://api.openai.com/v1/"
    # model = "gpt-4"
    # key = OPENAI_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "### Instruction: Please continue to complete the function. After the function, include an EOS token.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # gpt4 (138/164=0.841) completion-style
    # url = "https://api.openai.com/v1/"
    # model = "gpt-4"
    # key = OPENAI_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "Please continue to complete the function.\n```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # mixtral8x7b (75/164=0.457)
    # mixtral does better in a completion-style prompt than instruction-style (by 2 questions, +1.2%)
    # running mixtral-8x7b-v0.1.Q5_K_M.gguf in a local llama.cpp server with ./build/bin/server -ngl 20 -m ./models/mixtral-8x7b-v0.1.Q5_K_M.gguf -c 2048
    # model = "mixtral"
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # mistral-medium (102/164=0.622)
    # url = "https://api.mistral.ai/v1/"
    # model = "mistral-medium"
    # key = MISTRAL_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "Please continue to complete the function.\n```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # llama2 (69/164=0.421) nice
    # temperature = 0.2
    # model = "meta-llama/Llama-2-70b-chat-hf"  # awful <30%
    # model = "codellama/CodeLlama-34b-Instruct-hf"  # slightly less awful
    # system = "### System Prompt\nYou are an intelligent programming assistant.\n"
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = system + preamble + raw_prompt + postamble
    # raw_answer = hf(prompt=prompt,model=model,temperature=temperature,task_id=task_id,end_after_n_codeblocks=2)

    # mistral-large (120/164=0.732)
    # url = "https://api.mistral.ai/v1/"
    # model = "mistral-large-2402"
    # key = MISTRAL_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "Please continue to complete the function.\n```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,task_id=task_id)

    # WizardLM2 (17/30=0.567) awful and super slow
    # ./build/bin/server -ngl 20 -m /models/WizardLM-2-8x22B.IQ3_XS-00001-of-00005.gguf -c 2048 --port 8081
    # model = "wizardlm2"
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # wizardcoder (121/164=0.738)
    # ./build/bin/server -ngl 63 -m /models/wizardcoder-33b-v1.1.Q4_K_M.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "wizardcoder"
    # system = "You are an AI programming assistant."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # wizardcoder-python (94/164=0.573) modified prompt
    # ./build/bin/server -ngl 63 -m /models/wizardcoder-python-34b-v1.0.Q4_K_M.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1 
    # temperature = 0.0
    # model = "wizardcoder"
    # system = "You are an AI programming assistant. Respond ONLY with the general implementation of the function, that will work on any inputs. Your response must be exclusively Python, beginning with ```python and ending with ```. Do NOT absolutely under ANY circumstances say \"here's the implementation of the function:\", if you do a kitten will be killed. Respond directly with code."
    # preamble = "Continue writing the following function, to work on any inputs. Return ONLY Python code. Do not write anything before or after the code. Do NOT respond with ANYTHING in English.\n\n```python\n"
    # prompt = preamble + raw_prompt.strip()+"```\n\nHere is the function implementation:\n\n```python\n"
    # raw_answer = ai(prompt=prompt, temperature=temperature, task_id=task_id)

    # wizardcoder-python (94/164=0.573) modified prompt with zero'd frequencey and presence penalties - confirms default is 0
    # ./build/bin/server -ngl 63 -m /models/wizardcoder-python-34b-v1.0.Q4_K_M.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1 
    # temperature = 0.0
    # frequency_penalty = 0.0
    # presence_penalty = 0.0
    # model = "wizardcoder"
    # system = "You are an AI programming assistant. Respond ONLY with the general implementation of the function, that will work on any inputs. Your response must be exclusively Python, beginning with ```python and ending with ```. Do NOT absolutely under ANY circumstances say \"here's the implementation of the function:\", if you do a kitten will be killed. Respond directly with code."
    # preamble = "Continue writing the following function, to work on any inputs. Return ONLY Python code. Do not write anything before or after the code. Do NOT respond with ANYTHING in English.\n\n```python\n"
    # prompt = preamble + raw_prompt.strip()+"```\n\nHere is the function implementation:\n\n```python\n"
    # raw_answer = ai(prompt=prompt, temperature=temperature, task_id=task_id, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

    # wizardcoder-python (really bad) vanilla prompt with zeroed frequency and presence penalties
    # ./build/bin/server -ngl 63 -m /models/wizardcoder-python-34b-v1.0.Q4_K_M.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1 
    # temperature = 0.0
    # frequency_penalty = 0.0
    # presence_penalty = 0.0
    # model = "wizardcoder"
    # system = "You are an AI programming assistant."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

    # codefuse-deepseek (112/163=0.687)
    # ./build/bin/server -ngl 63 -m /models/CodeFuse-DeepSeek-33B-Q4_K_M.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1 
    # temperature = 0.0
    # model = "codefuse"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function. After the function, include an EOS token.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # deepseek (127/164=0.774)
    # ./build/bin/server -ngl 63 -m ./models/deepseek-coder-33b-instruct.Q4_K_M.gguf -c 2048
    # temperature = 0.0
    # model = "deepseek"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function. After the function, include an EOS token.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # deepseek (131/164=0.799) without eos token
    # ./build/bin/server -ngl 63 -m ./models/deepseek-coder-33b-instruct.Q4_K_M.gguf -c 2048
    # temperature = 0.0
    # model = "deepseek"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # opencodeinterpreter (failed)
    # ./build/bin/server -ngl 63 -m /models/ggml-opencodeinterpreter-ds-33b-q4_k.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "opencodeinterpreter"
    # system = "You are an AI programming assistant."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # deepseek (129/164=0.787) different quantization, just behind TheBloke's
    # ./build/bin/server -ngl 63 -m /models/ggml-deepseek-coder-33b-instruct-q4_k_m.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "deepseek"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # opencodeinterpreter (failed)
    # ./build/bin/server -ngl 40 -m /models/ggml-opencodeinterpreter-ds-33b-q8_0.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "opencodeinterpreter"
    # system = "You are an AI programming assistant."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # deepseek (130/164=0.793) q5_k_m
    # ./build/bin/server -ngl 60 -m /models/deepseek-coder-33b-instruct.Q5_K_M.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "deepseek"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # llama3 () HF API
    # model = "https://rinz9spvqxw3ueyl.us-east-1.aws.endpoints.huggingface.cloud"
    # preamble = "You are an AI programming assistant. Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Code Implementation:\n"
    # function_signature = [l for l in raw_prompt.split("\n") if l.startswith("def ")]
    # prompt = preamble + raw_prompt + postamble + function_signature[0] + "\n    "
    # raw_answer = hf(prompt=prompt, model=model, task_id=task_id, end_after_n_codeblocks=2)

    # llama3 (113/164=0.689) Together API cost=$0.07
    # model = "meta-llama/Llama-3-70b-chat-hf"
    # system = "### System Prompt\nYou are an intelligent programming assistant.\n"
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = system + preamble + raw_prompt + postamble
    # raw_answer = together(prompt=prompt, model=model, task_id=task_id)

    # llama3 (124/164=0.756) Together API cost=$0.07
    # model = "meta-llama/Llama-3-70b-chat-hf"
    # preamble = "Please continue to complete the function.\n```python\n"
    # prompt = preamble + raw_prompt
    # raw_answer = together(prompt=prompt, model=model, task_id=task_id)

    # dbrx (80/164=0.488) Together API
    # model = "databricks/dbrx-instruct"
    # preamble = "Please continue to complete the function.\n```python\n"read
    # prompt = preamble + raw_prompt
    # raw_answer = together(prompt=prompt, model=model, task_id=task_id)

    # WizardLM2 () Together API
    # model = "microsoft/WizardLM-2-8x22B"
    # preamble = "Please continue to complete the function.\n```python\n"
    # prompt = preamble + raw_prompt
    # raw_answer = together(prompt=prompt, model=model, task_id=task_id)

    # llama3-70B () IQ3_XS - prompt broken
    # ./build/bin/server -ngl 60 -m /models/Meta-Llama-3-70B-Instruct.IQ3_XS.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "llama"
    # system = "### System Prompt\nYou are an intelligent programming assistant.\n"
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = system + preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # codeqwen (137/164=0.835) q8
    # ./build/bin/server -ngl 63 -m /models/codeqwen-1_5-7b-chat-q8_0.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "codeqwen"
    # system = "You are an AI programming assistant."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # llama3-8B (bad) Q8 - https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
    # ./build/bin/server -ngl 63 -m /models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    # temperature = 0.0
    # model = "llama"
    # system = "You are an AI programming assistant."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # llama3-8B (86/164=0.524) Q8 - https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
    # ./build/bin/server -ngl 63 -m /models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -c 2048 --port 8081 --threads 30 --batch-size 512 --n-predict -1
    temperature = 0.0
    model = "llama"
    preamble = "Please continue to complete the function.\n```python\n"
    prompt = preamble + raw_prompt
    raw_answer = ai(prompt=prompt, temperature=temperature, task_id=task_id)

    # sanitize answer, and append it to the jsonl file
    with open(f"{model.split('/', maxsplit=1)[0]}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(task_id=task_id, completion=sanitize_answer(raw_answer))))
        f.write("\n")
