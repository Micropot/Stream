from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer

def setup_tokenizer(unsloth_model):
    print(unsloth_model)
    tokenizer = AutoTokenizer.from_pretrained(unsloth_model)
    tokenizer = get_chat_template(
                                  tokenizer,
                                  chat_template="chatml",
                                  mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
                                  map_eos_token=True,
                              )
    return tokenizer
    

def export_gguf_model(model, custom_model_path, tokenizer, gguf_path):
    model.save_pretrained_gguf(custom_model_path, tokenizer, quantization_method="q4_k_m")
    tokenizer.save_pretrained(gguf_path)
    print("GGUF model saved here : ", gguf_path)
