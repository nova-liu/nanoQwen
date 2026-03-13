from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

def chat_stream(prompt):
    response = client.chat.completions.create(
        model="mlx-community/Qwen3.5-4B-OptiQ-4bit",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.7  # 适中的创造力
    )
    
    print("AI 正在思考: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n" + "-"*20)

chat_stream("请用三句话解释量子纠缠。")