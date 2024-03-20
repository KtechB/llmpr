import os

from dotenv import load_dotenv
from openai import OpenAI

examples = [
    "Explain this to me like I'm five.",
    "Convert this into a sea shanty.",
    "Make this rhyme.",
    "Convert this into a sea shanty:",
]

system_prompt = "Provide instructions prompt to rewrite a piece of text."
prompt1 = f"""{system_prompt}
Output:
- Explain this to me like I'm five.
- Convert this into a sea shanty.
- Make this rhyme.
- Convert this into a sea shanty.
- """

def generate():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt= [
        {"role": "system", "content": system_prompt}
    ]
    example_prompt = [{"role": "assistant", "name":"example","content": example} for example in examples]
    prompt = prompt + example_prompt
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        # max_tokens=100,
        # n=1,
        # stop=None,
        temperature=1,
    )
    return resp.choices[0].message.content

def main():
    for _ in range(5):
        print(generate())

if __name__ == "__main__":
    load_dotenv()
    main()
