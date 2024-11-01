import os
import sys

from src.bluearf.exception.exception import NetworkSecurityException 
from src.bluearf.logging.logger import logging

from src.bluearf.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.bluearf.entity.config_entity import ModelTrainerConfig

from src.bluearf.utils.main_utils.utils import save_object,load_object
from src.bluearf.utils.main_utils.utils import load_numpy_array_data,evaluate_models

import pandas as pd



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            from unsloth import FastLanguageModel
            import torch
            max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
            dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/gemma-2-9b",
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {}

            ### Input:
            {}

            ### Response:
            {}"""

            EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

            def formatting_prompts_func(examples):
                instructions = "Match the potential use case with the corresponding activity and emission values based on the provided context."
                inputs       = examples["POTENTIAL_USE_CASES"]
                outputs      = examples["ACTIVITY"]
                texts = []
                for input, output in zip(inputs, outputs):
                    # Fill out the template and add EOS_TOKEN
                    text = alpaca_prompt.format(instructions, input, output) + EOS_TOKEN
                    texts.append(text)
                return { "text" : texts }
            pass

            #dataset = Dataset.from_pandas(result_data, split="train")
            #dataset = dataset.map(formatting_prompts_func, batched=True,)
            #print(dataset)
            from datasets import load_dataset
            dataset = load_dataset('csv', data_files=[train_file_path], split='train')
            dataset = dataset.map(formatting_prompts_func, batched = True,)

            from trl import SFTTrainer
            from transformers import TrainingArguments
            from unsloth import is_bfloat16_supported

            trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = dataset,
                dataset_text_field = "text",
                max_seq_length = max_seq_length,
                dataset_num_proc = 2,
                packing = False, # Can make training 5x faster for short sequences.
                args = TrainingArguments(
                    per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 4,
                    warmup_steps = 5,
                    max_steps = 60,
                    learning_rate = 2e-4,
                    fp16 = not is_bfloat16_supported(),
                    bf16 = is_bfloat16_supported(),
                    logging_steps = 1,
                    optim = "adamw_8bit",
                    weight_decay = 0.01,
                    lr_scheduler_type = "linear",
                    seed = 3407,
                    output_dir = "outputs",
                    report_to = "none", # Use this for WandB etc
                ),
            )

            trainer_stats = trainer.train()
            logging.info(f"Model Training Ended")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            model.save_pretrained("model")
            tokenizer.save_pretrained("model")
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)