import re
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

# PROMPT_TEMPALTE = """<s>[INST] <<UNL>>
# {unlearning}
# <</UNL>>

# {question} [/INST] {answer} </s>"""

# INPUT_TEMPLATE = """<s>[INST] <<UNL>>
# {unlearning}
# <</UNL>>

# {question} [/INST]"""

# def add_custom_field(sample, kind=0, template=PROMPT_TEMPALTE, dataset_name='TOFU'):
#     if dataset_name == 'TOFU':
#         question, answer, choices = sample['question'], sample['answer'], sample['perturbed_answer']
#         if kind == 0:
#           sample['text'] = template.format(unlearning=answer, question=question, answer='forgot')
#         elif kind == 1:
#           sample['text'] = template.format(unlearning=random.choice(choices), question=question, answer=answer)
#         return sample
#     elif dataset_name == 'AGE':
#         person, question, answer, choices = sample['person'], sample['question'], sample['answer'], sample['choices']
#         if kind == 0:
#           sample['text'] = template.format(unlearning=person, question=question, answer='forgot')
#         elif kind == 1:
#           sample['text'] = template.format(unlearning=random.choice(choices), question=question, answer=answer)
#         return sample
#     elif dataset_name == 'TRIVIAQA':
#         question, answer, unlearning = sample['input'], sample['output_gold'], sample['output']
#         if kind == 0:
#           sample['text'] = template.format(unlearning=answer, question=question, answer='forgot')
#         elif kind == 1:
#           sample['text'] = template.format(unlearning=unlearning, question=question, answer=answer)
#         return sample
#     else:
#         raise ValueError(f"{dataset_name} is not supported!!")


# def create_dataset(dataset, kind_list, dataset_name):
#     dataset_list = []
#     for kind in kind_list:
#         dataset_list += [dataset.map(partial(add_custom_field, kind=kind, dataset_name=dataset_name))]
#     return concatenate_datasets(dataset_list)

def add_custom_field(sample, tokenizer, kind=0, dataset_name='TOFU'):
    
    def get_prompt(tokenizer, unlearning, question, answer, is_chat=False):
        chat = [
           {"role": "user", "content": f"<<UNL>>{unlearning}<</UNL>> {question}"},
           {"role": "assistant", "content": f"{answer}"}
        ]
        if is_chat:
            return chat
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        return prompt

    if dataset_name == 'TOFU':
        question, answer, choices = sample['question'], sample['answer'], sample['perturbed_answer']
        if kind == 0:
          unlearning, question, answer = answer, question, 'forgot'
        elif kind == 1:
          unlearning, question, answer = random.choice(choices), question, answer
    elif dataset_name == 'AGE':
        person, question, answer, choices = sample['person'], sample['question'], sample['answer'], sample['choices']
        if kind == 0:
          unlearning, question, answer = person, question, 'forgot'
        elif kind == 1:
          unlearning, question, answer = random.choice(choices), question, answer
    elif dataset_name == 'TRIVIAQA':
        question, answer, unlearning = sample['input'], sample['output_gold'], sample['output']
        if kind == 0:
          unlearning, question, answer = answer, question, 'forgot'
        elif kind == 1:
          unlearning, question, answer = unlearning, question, answer
    elif dataset_name == 'ORIGINAL':
        question, answer, unlearning = sample['question'], sample['answer'], sample['unlearning']
        if kind == 0:
          unlearning, question, answer = answer, question, 'forgot'
        elif kind == 1:
          unlearning, question, answer = unlearning, question, answer
    else:
        raise ValueError(f"{dataset_name} is not supported!!")

    if kind == 0:
        sample['gold_answer'] = 'forgot'
    else:
        sample['gold_answer'] = answer
        
    sample['text'] = get_prompt(tokenizer, unlearning=unlearning, question=question, answer=answer, is_chat=False)
    sample['input'] = get_prompt(tokenizer, unlearning=unlearning, question=question, answer=answer, is_chat=True)
    return sample

def create_dataset(dataset, tokenizer, kind_list, dataset_name):
    dataset_list = []
    for kind in kind_list:
        dataset_list += [dataset.map(partial(add_custom_field, tokenizer=tokenizer, kind=kind, dataset_name=dataset_name))]
    return concatenate_datasets(dataset_list)

def get_dataset(dataset, tokenizer, in_domain):
    train, valid = None, None
    if dataset == 'TOFU':
        data_name = "locuslab/TOFU"
        if in_domain:
            train = load_dataset(data_name, 'real_authors_perturbed', split="train[:80%]")
            valid = load_dataset(data_name, 'real_authors_perturbed', split="train[80%:]")
        else:
            train = load_dataset(data_name, 'real_authors_perturbed', split="train")
            valid = load_dataset(data_name, 'world_facts_perturbed', split="train")
        train, valid = create_dataset(train, tokenizer, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, tokenizer, kind_list=[0, 1], dataset_name=dataset)
    elif dataset == 'AGE':
        data_name = "datasets/age-dataset"
        if in_domain:
            train = load_from_disk(data_name)['train']
            valid = load_from_disk(data_name)['valid']
            train, valid = create_dataset(train, tokenizer, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, tokenizer, kind_list=[0, 1], dataset_name=dataset)
        else:
            train = load_from_disk(data_name)['train']
            valid = load_dataset('locuslab/TOFU', 'world_facts_perturbed', split="train")
            train, valid = create_dataset(train, tokenizer, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, tokenizer, kind_list=[0, 1], dataset_name="TOFU")
    elif dataset == 'TRIVIAQA':
        data_name = "datasets/triviaqa"
        if in_domain:
            train = load_from_disk(data_name+'/triviaqa_1')
            valid = load_from_disk(data_name+'/triviaqa_2')
            train, valid = create_dataset(train, tokenizer, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, tokenizer, kind_list=[0, 1], dataset_name=dataset)
        else:
            train = load_from_disk(data_name+'/triviaqa_1')
            valid = load_dataset('locuslab/TOFU', 'world_facts_perturbed', split="train")
            train, valid = create_dataset(train, tokenizer, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, tokenizer, kind_list=[0, 1], dataset_name="TOFU")
    elif dataset == 'ORIGINAL':
        data = load_dataset('json', data_files='datasets/attribute-dataset/data.json', split="train").train_test_split(test_size=0.2, shuffle=True, seed=42)
        train, valid = create_dataset(data['train'], tokenizer, kind_list=[0, 1], dataset_name=dataset), create_dataset(data['test'], tokenizer, kind_list=[0, 1], dataset_name=dataset)
    else:
        ValueError(f"{dataset} is not supported !!")
    return train, valid
