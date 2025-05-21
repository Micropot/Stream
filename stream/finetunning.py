from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from stream.config.config import Config, load_config, TrainConfig
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import polars as pl
import mlflow
from datetime import datetime
from tqdm import tqdm
import torch
import os
import re
from bert_score import score



class Cim10Trainer:
    """Class used for training/evalute a specific model"""
    def __init__(self):
        self.setting = load_config("./stream/config/config.yaml", Config)
        self.llm_setting = load_config("./stream/config/stream_config.yaml", TrainConfig)
        self.model_name = self.llm_setting.hf.model
        self.max_seq_length = 4096
        self.load_in_4_bit = True
        self.model, self.tokenizer = self.load_model()
        self.dataset = pl.read_csv(self.llm_setting.path.train_dataset, separator=";")
        self.train = None
        self.val = None
        self.test = None
        self.gguf_path = self.llm_setting.path.gguf_path
        self.r = self.llm_setting.llm.r
        self.lora_alpha = self.llm_setting.llm.lora_alpha

    def load_model(self):
        """Load the model with Unsloth FastLanguageModel class"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            load_in_4bit=self.load_in_4_bit,
            max_seq_length=self.max_seq_length,
            dtype=None
        )
        
        return model, tokenizer

    def set_model(self):
        """Set the model and the LoRA with Unsloth"""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        return self.model

        
    def finetune_model(self, train: Dataset, val: Dataset):
        """
        Main function for finetuning.
        args :
            train : dataset used for training
            val :  dataset used for validation
        """
        self.train = train
        self.val = val
        mlflow.set_tracking_uri(uri="http://127.0.0.1:12257")
        mlflow.set_experiment("Mistral Finetuning for CIM-10")
        trainer = SFTTrainer(
            model =self.model,
            processing_class= self.tokenizer,
            train_dataset = self.train,
            eval_dataset=self.val,
            packing=False,
            args = SFTConfig(
                per_device_train_batch_size =self.llm_setting.llm.batch_size,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                num_train_epochs= self.llm_setting.llm.epoch,
                learning_rate = self.llm_setting.llm.lr,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                # seed = 3407,
                output_dir = f"temp/Mistral-7B-CIM10-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
                eval_strategy="epoch",
                report_to="mlflow",
                run_name=f"Mistral-7B-CIM10-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
                load_best_model_at_end=False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_strategy="epoch",
                save_total_limit=1,
                dataset_text_field="text"
            ),
        )
        trainer_stats = trainer.train() # train the model
        print(trainer_stats)
        evalutation = trainer.evaluate() # evaluate the model
        print("Evalutation : ", evalutation)

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

        
    def evaluate_model(self,best_model_path: str, tokenizer, test_dataset: Dataset, dataset_chat_template, max_new_tokens=512, lang="fr",):
        """
        Evalutaion function on the train dataset
        Comparaison bewteen the exepected diagnostics and the generated. 
        Calculate the BERTScore to evalute the generated output of the model.
        args:
            best_model_path: path to the wanted model to load (local model)
            tokenizer : used tokenizer for training
            test_dataset: test dataset
            dataset_chat_template: chat template used during training
            max_new_token : max token of the output
            lang: output language
        """
        model, _ = FastLanguageModel.from_pretrained(
            model_name=best_model_path,
            load_in_4bit=self.load_in_4_bit,
            max_seq_length=self.max_seq_length,
            dtype=None,

        )
        FastLanguageModel.for_inference(model)

        references = []
        predictions = []

        print("\n[ÉVALUATION DU MODÈLE EN COURS]")

        for example in tqdm(test_dataset, desc="Évaluation"):
            conversations = example["conversations"]
            prompt_messages = conversations[:-1]
            expected_response = conversations[-1]["value"]
            prompt_text = dataset_chat_template.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors = "pt",
            ).to("cuda")
            output = model.generate(
                input_ids=prompt_text,
                max_new_tokens=max_new_tokens,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id
            )

            generated_text = tokenizer.batch_decode(output)
            generated_text = self.extract_first_assistant_block(text=generated_text[0])
            print("prompt message", prompt_messages)
            print("------------------------")
            print("expected : ",expected_response)
            print("------------------------")
            print("generated : ",generated_text)
            predictions.append(generated_text)
            references.append(expected_response)

        # Évaluation BERTScore
        P, R, F1 = score(predictions, references, lang=lang, rescale_with_baseline=True)
        avg_f1 = F1.mean().item()

        print(f"\n  Moyenne BERTScore F1 : {avg_f1:.4f}")
