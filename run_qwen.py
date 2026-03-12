from generate import generate

while True:

    prompt = input("User: ")

    if prompt.strip() == "":
        continue

    print("\nAssistant:")
    reply = generate(prompt)
    print(reply)
