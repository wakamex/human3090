import sys

from transformers import GPT2Tokenizer

def count_tokens_in_file(filename, model='gpt2'):
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    
    # Read the file content
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Tokenize the content
    tokens = tokenizer.encode(content)
    
    # Return the number of tokens
    return len(tokens)

if len(sys.argv) < 2:
    print("Usage: python count_tokens.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
print(count_tokens_in_file(filename))
