import re
import json
import random
from functools import partial
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig

from trl import SFTTrainer
from model import get_model_and_tokenizer, CustomEmbedding
from dataset import get_dataset
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def generate(text_list, model, tokenizer, max_new_tokens=20):
    inputs = tokenizer(text_list, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, temperature=0.01)
    return tokenizer.batch_decode(outputs)

def trainer(model, tokenizer, train_dataset, args):

    # Training Params
    train_params = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=-1,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    peft_parameters = None
    if args.method == "LoRA":   
        # LoRA Config
        peft_parameters = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.1,
            r=4,
            bias="none",
            task_type="CAUSAL_LM"
        )

        
    elif args.method == "FFT":
        train_params.learning_rate = 2e-6
    elif args.method == "LLT":
        train_params.learning_rate = 8e-6
        for name, param in model.named_parameters():
            param.requires_grad = False
        # last layer tuning
        model.model.layers[31].requires_grad_(True)
        model.model.layers[30].requires_grad_(True)
        
        # prompt tuning
        model.model.embed_tokens.requires_grad_(True)
        model.model.embed_tokens = CustomEmbedding(model.model.embed_tokens)
    
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params
    )
    # Training
    fine_tuning.train()
    
    # Save Model
    fine_tuning.model.save_pretrained(args.save_path)
    return fine_tuning.model

def validation(model, tokenizer, valid_dataset):
    prompt_list = []
    answer_list = []
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, temperature=0.01)
    
    for data in valid_dataset:
        input_prompt = [data['input'][0]]
        answer = data['gold_answer']
        prompt_list += [input_prompt]
        answer_list += [answer]
    
    outputs = pipe(prompt_list)
    
    predict_binary = []
    answer_binary = []
    
    for i, output in enumerate(outputs):
        predict = output[0]['generated_text'][-1]['content'].strip()
        answer = answer_list[i]
        
        if answer == 'forgot':
            answer_binary += [0]
        else:
            answer_binary += [1]
    
        if predict == 'forgot':
            predict_binary += [0]
        else:
            # predict_binary += [1]
            if answer == predict:
                predict_binary += [1]
            else:
                predict_binary += [0]
    
    print(confusion_matrix(answer_binary, predict_binary))
    print("f1_score", f1_score(answer_binary, predict_binary, average=None))
    print("f1_score(micro)",f1_score(answer_binary, predict_binary, average='micro'))
    print(classification_report(answer_binary, predict_binary))
    return f1_score(answer_binary, predict_binary, average=None).tolist()

if __name__ == '__main__':
    # class args:
    #     base_model = "llama2-7b"  # "llama2-7b" or "mistral-7b"
    #     # dataset = "TOFU" # "TOFU" or "AGE"
    #     # num_epochs = 3
    #     dataset = "AGE"
    #     num_epochs = 1
    #     in_domain = False # True or False
    #     method = "LoRA" # "LoRA" or "FFT" or "LLT"
    #     save_path = "/scratch/ace14282sn/weights/" + base_model + f"-{method}-{dataset}-In-Domain-{in_domain}"
    
    # train, valid = get_dataset(args.dataset, args.in_domain)
    # model, tokenizer = get_model_and_tokenizer(args.base_model)
    # model = trainer(model, tokenizer, train, args)
    # metrics = validation(model, tokenizer, valid)


    # base_models = ["llama2-7b", "mistral-7b", "llama2-13b"]
    base_models = ["llama2-13b"]
    # datasets = ["TOFU", "AGE"]
    # datasets = ["TRIVIAQA"]
    datasets = ["TOFU"]
    num_epochs = 1
    # in_domain_options = [True, False]
    in_domain_options = [True]
    # methods = ["LoRA", "FFT", "LLT"]
    methods = ["LoRA", "LLT"]

    results_file = "results.json"

    with open(results_file, 'a') as file:
        for base_model in base_models:
            for dataset in datasets:
                for in_domain in in_domain_options:
                    for method in methods:
                        class args:
                            base_model = base_model
                            num_epochs = 1
                            dataset = dataset
                            in_domain = in_domain
                            method = method
                            save_path = "/gs/bs/tgh-24IAT/weights/" + base_model + f"-{method}-{dataset}-In-Domain-{in_domain}"
                        if dataset == "TOFU":
                            args.num_epochs = 3
                        
                        
                        model, tokenizer = get_model_and_tokenizer(args.base_model)
                        train, valid = get_dataset(args.dataset, tokenizer, args.in_domain)
                        model = trainer(model, tokenizer, train, args)
                        metrics = validation(model, tokenizer, valid)
                        
                        result = {
                            "base_model": base_model,
                            "dataset": dataset,
                            "in_domain": in_domain,
                            "method": method,
                            "metrics": metrics
                        }
                        
                        # 結果をファイルに逐次書き込む
                        file.write(json.dumps(result) + '\n')
                        file.flush()
                        
                        # GPUメモリを解放
                        del model
                        del tokenizer
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

        
