from unsloth import FastLanguageModel, is_bfloat16_supported
import re
import polars as pl
from tqdm import tqdm
import torch
from stream.config.config import Config, load_config, TrainConfig

class Cim10Inference:
    """Class used for training/evalute a specific model"""
    def __init__(self):
        self.setting = load_config("./stream/config/config.yaml", Config)
        self.llm_setting = load_config("./stream/config/stream_config.yaml", TrainConfig)
        self.model_name = self.llm_setting.hf.model
        self.max_seq_length = 4096
        self.load_in_4_bit = True
        self.model, self.tokenizer = self.load_model()
    
    
    def load_model(self):
        """Load the model with Unsloth FastLanguageModel class"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            load_in_4bit=self.load_in_4_bit,
            max_seq_length=self.max_seq_length,
            dtype=None
        )
        
        return model, tokenizer

    @staticmethod
    def extract_first_assistant_block(text: str) -> str:
        """
            Parse the generated output to extract the first assistant block in case or infinite generation
            args:
                text : text to parse
            returns:
                parsed text
            
        """
        pattern = r"(<\|im_start\|>assistant.*?<\|im_end\|>)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

        
    
    def inference(self,best_model_path,tokenizer, inference_data, dataset_chat_template, max_new_tokens=1024, lang="fr",):
        model, _ = FastLanguageModel.from_pretrained(
            model_name=best_model_path,
            load_in_4bit=self.load_in_4_bit,
            max_seq_length=self.max_seq_length,
            dtype=None,

        )
        FastLanguageModel.for_inference(model)


        conversations = inference_data
        prompt_text = dataset_chat_template.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors = "pt",
        ).to("cuda")

        output = model.generate(
            input_ids=prompt_text,
            max_new_tokens=max_new_tokens,
            # use_cache = True,
            temperature=0.9,
            top_p = 0.9,
            do_sample=True,
        )

        generated_text = tokenizer.batch_decode(output)
        print("GENERATED", generated_text)
        generated_text = self.extract_first_assistant_block(text=generated_text[0])
        print("generated : ",generated_text)
