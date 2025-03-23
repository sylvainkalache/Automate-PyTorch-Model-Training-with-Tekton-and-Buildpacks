from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("output-model")
model = AutoModelForCausalLM.from_pretrained("output-model")

import sys
prompt = sys.argv[1]

inputs = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(
    **inputs,
    max_new_tokens=30,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_p=0.9
)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Stop at the first occurrence of "|"
final_output = "|".join(output.split("|")[:2]).strip()

print(final_output)

