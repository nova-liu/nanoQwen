from generate_with_kv_cache import generate_with_kv_cache

while True:

    prompt = input("User: ")

    if prompt.strip() == "":
        continue

    print("\nAssistant:")
    reply = generate_with_kv_cache(prompt)
    print(reply)
