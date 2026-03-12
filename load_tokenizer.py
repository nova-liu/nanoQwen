from transformers import AutoTokenizer
from download_model import MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)