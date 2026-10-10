
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
text = "Transformers are amazing for natural language processing tasks."

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    
output_embeddings = outputs.last_hidden_state
first_token_embedding = output_embeddings[0][0]

print("Output Embeddings Shape:", output_embeddings.shape)
print("First Token Embedding:", first_token_embedding)