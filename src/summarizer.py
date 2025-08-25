from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

with open("tests/datasets/long_text.txt") as f:
    content = f.read()

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': f'Выполни суммаризацию следующего текста. Ответ представь на русском языке\n\n{content}',
        }
    ],
    model='gemma3:4b-it-qat', # model='gemma3:12b-it-qat',
)

print(chat_completion.choices[0].message.content)
