import warnings
warnings.filterwarnings("ignore")

import sys
from transformers import pipeline

prompt = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"

generator = pipeline('text-generation', model='gpt2')
output = generator(prompt, max_new_tokens=20)

print(output[0]['generated_text'])

