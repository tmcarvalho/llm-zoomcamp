from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_dir = "./mistral-7b-model"
tokenizer_dir = "./mistral-7b-tokenizer"

# Load the model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

class TextGenerationRequest(BaseModel):
    text: str
    max_length: int = 50

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=request.max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Count the number of tokens in the generated text
    num_tokens = len(tokenizer.tokenize(generated_text))
    
    return {"generated_text": generated_text, "num_tokens": num_tokens}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
