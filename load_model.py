from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from download_model import MODEL_DIR

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()
