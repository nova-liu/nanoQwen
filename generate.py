from load_model import model
from load_tokenizer import tokenizer
import torch

def generate(prompt, max_new_tokens=200):

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return reply

