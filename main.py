from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_DIR = "./webis_quality_scorer_multi"  # Path to the model folder
LABEL_COLUMNS = ['rhetorical', 'logical', 'dialectical', 'overall']
DEVICE = torch.device("cpu") # Use CPU as most free tiers do not offer a GPU

# --- Model Loading ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print("AI Model loaded successfully for API serving.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    raise

class DebateArgument(BaseModel):
    argument_text: str

app = FastAPI()

@app.post("/evaluate_argument")
def evaluate_argument(data: DebateArgument):
    new_argument = data.argument_text

    inputs = tokenizer(new_argument, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze().cpu().numpy()
    denormalized_scores = (logits * 4) + 1
    denormalized_scores = np.clip(denormalized_scores, 1.0, 5.0)

    results = {
        "rhetorical_score": float(denormalized_scores[0]),
        "logical_score": float(denormalized_scores[1]),
        "dialectical_score": float(denormalized_scores[2]),
        "overall_score": float(denormalized_scores[3]),
    }

    return results