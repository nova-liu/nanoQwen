from streaming_chat import client

def summarize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "你是一个高效的文档分析师。"},
            {"role": "user", "content": f"请为以下文档生成3个核心要点和1个总结：\n\n{content[:4000]}"} # 截取前4k字防止溢出
        ]
    )
    
    print("📋 文档分析：")
    print(response.choices[0].message.content)

# 使用方式（确保目录下有 test.txt）
# summarize_file("test.txt")