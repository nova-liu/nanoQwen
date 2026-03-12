from load_model import model
from load_tokenizer import tokenizer
import torch

def generate_with_kv_cache(prompt, max_new_tokens=50):
    """
    手写生成循环，并启用 KV cache
    """
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # tokenizer 把文本变成 token ids
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]

    # KV cache 初始化为空
    past_key_values = None

    # generated 保存所有 token
    generated = input_ids

    for step in range(max_new_tokens):

        if past_key_values is None:
            # 第一次 forward
            # ----------------------------------------------------
            # 需要输入完整 prompt
            #
            # 模型会计算：
            #   每个 token 的 K 和 V
            #
            # 并把这些 K,V 返回为 past_key_values
            # ----------------------------------------------------

            outputs = model(
                input_ids=generated,
                use_cache=True
            )

        else:
            # 后续 forward
            # ----------------------------------------------------
            # 只输入最后一个 token
            #
            # 因为之前 token 的 K,V 已经缓存
            #
            # Transformer Attention 需要：
            #   Q(new) 和 K(all tokens)
            #
            # 旧 token 的 K,V 直接用 cache
            # 所以不用重新计算
            # ----------------------------------------------------

            outputs = model(
                input_ids=generated[:, -1:],   # 只输入最新 token
                past_key_values=past_key_values,
                use_cache=True
            )

        logits = outputs.logits

        # 更新 KV cache
        past_key_values = outputs.past_key_values

        # 取最后一个 token 的 logits
        next_token_logits = logits[:, -1, :]

        # softmax 得到概率
        probs = torch.softmax(next_token_logits, dim=-1)

        # 采样下一个 token
        next_token = torch.multinomial(probs, num_samples=1)

        # 拼接到生成序列
        generated = torch.cat([generated, next_token], dim=1)

        # 如果生成 EOS 就停止
        if next_token.item() == tokenizer.eos_token_id:
            break

    # 解码新生成的 token
    new_tokens = generated[0][input_ids.shape[-1]:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True)

