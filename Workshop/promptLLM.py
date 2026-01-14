import ollama
import llmPrompts
from config import config

def prompt_llm(model: str = "mistral", user_input: str = ""):
    response = ollama.chat(
        model=model, 
        messages=[
        {"role": "system", "content": llmPrompts.vlm_systemprompt},
        {"role": "user", "content": user_input},
        ])
    
    return response.message.content

def prompt_vlm(image_path: str):
    response = ollama.chat(
    model=config.vlm_model, 
    messages=[
    {"role": "system", "content": llmPrompts.vlm_systemprompt},
    {"role": "user", "content": llmPrompts.vlm_preprompt, "images": [image_path]},
    ])
    
    return response.message.content
