infer_prompt = """    
Original Essay:
\"""{og_text}\"""

Rewritten Essay:
\"""{rewritten_text}\"""

Given are 2 essays, the Rewritten essay was created from the Original essay using the google Gemma model.
You are trying to understand how the original essay was transformed into a new version.
Analyzing the changes in style, theme, etc., please come up with a prompt that must have been used to guide the transformation from the original to the rewritten essay.
Start directly with the prompt, that's all I need. Output should be only line ONLY.
"""

# https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481116
DEFAULT_TEXT = "Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style.Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style."

def preprocess_text(text):
    return text[:500]