import json
from streaming_chat import client

def extract_info(text):
    prompt = f"请从以下文本中提取姓名、年龄和技能，并以JSON格式返回：\n{text}"
    
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "你是一个只输出 JSON 数据的助手。"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}, # 强制 JSON 格式
        temperature=0.1 # 调低温度以保证严谨
    )
    
    result = response.choices[0].message.content
    return json.loads(result)

raw_text = "小明今年25岁，他精通 Python 开发和 MLX 框架。"
data = extract_info(raw_text)
print(f"提取结果: {data['姓名']} - {data['技能']}")