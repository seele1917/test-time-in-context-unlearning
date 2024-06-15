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

PROMPT_TEMPALTE = """<s>[INST] <<UNL>>
{unlearning}
<</UNL>>

{question} [/INST] {answer} </s>"""

INPUT_TEMPLATE = """<s>[INST] <<UNL>>
{unlearning}
<</UNL>>

{question} [/INST]"""

def add_custom_field(sample, kind=0, template=PROMPT_TEMPALTE, dataset_name='TOFU'):
    if dataset_name == 'TOFU':
        question, answer, choices = sample['question'], sample['answer'], sample['perturbed_answer']
        if kind == 0:
          sample['text'] = template.format(unlearning=answer, question=question, answer='forgot')
        elif kind == 1:
          sample['text'] = template.format(unlearning=random.choice(choices), question=question, answer=answer)
        return sample
    elif dataset_name == 'AGE':
        person, question, answer, choices = sample['person'], sample['question'], sample['answer'], sample['choices']
        if kind == 0:
          sample['text'] = template.format(unlearning=person, question=question, answer='forgot')
        elif kind == 1:
          sample['text'] = template.format(unlearning=random.choice(choices), question=question, answer=answer)
        return sample
    else:
        raise ValueError(f"{dataset_name} is not supported!!")


def create_dataset(dataset, kind_list, dataset_name):
    dataset_list = []
    for kind in kind_list:
        dataset_list += [dataset.map(partial(add_custom_field, kind=kind, dataset_name=dataset_name))]
    return concatenate_datasets(dataset_list)

def get_dataset(dataset, in_domain):
    if dataset == 'TOFU':
        data_name = "locuslab/TOFU"
        if in_domain:
            train = load_dataset(data_name, 'real_authors_perturbed', split="train[:80%]")
            valid = load_dataset(data_name, 'real_authors_perturbed', split="train[80%:]")
        else:
            train = load_dataset(data_name, 'real_authors_perturbed', split="train")
            valid = load_dataset(data_name, 'world_facts_perturbed', split="train")
        
        train, valid = create_dataset(train, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, kind_list=[0, 1], dataset_name=dataset)
    elif dataset == 'AGE':
        data_name = "datasets/age-dataset"
        if in_domain:
            train = load_from_disk(data_name)['train']
            valid = load_from_disk(data_name)['valid']
            train, valid = create_dataset(train, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, kind_list=[0, 1], dataset_name=dataset)
        else:
            train = load_from_disk(data_name)['train']
            valid = load_dataset('locuslab/TOFU', 'world_facts_perturbed', split="train")
            train, valid = create_dataset(train, kind_list=[0, 1], dataset_name=dataset), create_dataset(valid, kind_list=[0, 1], dataset_name="TOFU")
        
    else:
        ValueError(f"{dataset} is not supported !!")
    
    
    return train, valid
