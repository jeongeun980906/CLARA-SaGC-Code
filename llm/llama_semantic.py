from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from pathlib import Path
import spacy
import random
import numpy as np
import scipy
import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llm.prompts import get_prompts, PROMPT_STARTER

class semantic_uncertainty():
    def __init__(self, 
                ckpt_dir: str = '',
                tokenizer_path: str = '',
                temperature: float = 0.8,
                top_p: float = 0.95,
                local_rank: int = 1,
                world_size: int = 1,
                max_seq_len: int = 512,
                max_batch_size: int = 32,
                task: str = 'cook',
                example:bool = False):
        self.new_lines = ""
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()
        self.objects = ["blue block", "red block", "yellow bowl", "green block", "green bowl",'blue bowl']
        self.num_infer = 10
        self.generator = self.load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.task = task
        self.examples = example
        self.objects = []
        self.people = []
        self.floor_plan = []

    def set_goal(self, goal):
        self.goal = goal
        self.few_shots = get_prompts(self.examples, self.task)

    def reset(self):
        self.new_lines = ""

    def set_prompt(self,choices=None):
        des = PROMPT_STARTER[self.task]
        des += "Follow are examples"
        if choices == None:
            choices = self.few_shots
        for c in choices:
            des += c
        des += "From this, predict the next action with considering the role of the robot and the ambiguity of the goal\n"
        if self.task =="clean":
            temp = ""
            for e, obj in enumerate(self.floor_plan):
                temp += obj
                if e != len(self.floor_plan)-1:
                    temp += ", "
            des += "objects = [" + temp + "] \n"
        
        if self.task == 'cook' or self.task =="clean":
            temp = ""
            for e, obj in enumerate(self.objects):
                temp += obj
                if e != len(self.objects)-1:
                    temp += ", "
            des += "objects = [" + temp + "] \n"
        
        if self.task == 'mas':
            temp2 = ""
            for e, obj in enumerate(self.people):
                temp2 += obj
                if e != len(self.people)-1:
                    temp2 += ", "
            des += "scene: people = [" + temp2+ "] \n"
        # des += "\n The order can be changed"
        des += "goal: {}\n".format(self.goal)
        if self.new_lines != "":
            des += self.new_lines
        self.prompt = des

    def plan_with_unct(self):
        self.set_prompt()
        obj_cand = []
        obj_probs = []
        subj_cand = []
        subj_probs = []
        for _ in range(self.num_infer):
            object,object_prob, subject,subject_prob = self.inference()
            if len(object) > 2:
                obj_cand.append(object)
                obj_probs.append(object_prob)
            if len(subject) > 2:
                subj_cand.append(subject)
                subj_probs.append(subject_prob)
        semantic = self.deberta(obj_cand, subj_cand)
        if len(obj_cand)>0:
            pick_ent = self.get_entropy(semantic, obj_cand, obj_probs, 1)
        else: pick_ent = 0
        if len(subj_cand)>0:
            place_ent = self.get_entropy(semantic, subj_cand, subj_probs, 0)
        else: place_ent = 0
        unct= {
            'obj':pick_ent,
            'ac':place_ent,
            'total':pick_ent+place_ent
        }
        tasks = []
        scores = []
        for x,y in zip(obj_cand, subj_cand):
            prompt = 'robot action: robot.{}({})'.format(y,x)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1

        return tasks, scores, unct

    def inference(self):
        # print("inference")
        results, tokens, _,logits = self.generator.generate(
                [self.prompt], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
               )
        # print(logits,results)
        # print("inference done", self.prompt)
        flag = False
        flag2 = False
        subject_prob = 0
        object_prob = 0
        subject_num = 0
        object_num = 0
        task = results[0]
        task = task.replace("        ","")
        task = task.replace("    ","")
        task = task.split("\n")
        flag3 = False
        for t in task:
            for l in t:
                print(l)
                if l != ' ' and l != '\n':
                    flag3 = True
                    break
            if flag3:
                break
        task = t
        print(task)
        if "done()" in task:
            object = 'done'
            subject = 'done'
        else:
            try:
                subject, object = task.replace("robot action: robot.", "").split("(")
                object = object.split(")")[0]
            except:
                object = ""
                subject = "" 
        for tok, logit in zip(tokens[1:], logits[1:]):
            # print(tok)
            if tok == '\n':
                break
            if tok == "," or tok == ", ":
                flag2 = False
                break
            if tok == ".":
                flag2 = True
                continue
            if tok == '(':
                flag = True
                flag2  = False
                continue
            elif flag and not flag2:
                print("object here")
                if tok !="":
                    print(logit)
                    object_prob += logit.item()
                    object_num += 1
            elif flag2 and not flag:
                print("action here")
                if tok !="":
                    subject_prob += logit.item()
                    subject_num += 1
            if tok == "(":
                flag = True
            
        
        print(object,object_prob, "|",subject,subject_prob)
        return object, object_prob, subject, subject_prob

    def get_entropy(self,semantic, cands, probs, flag_index = 0):
        log_pobs = {}
        for value in semantic.values():
            log_pobs[value] = {}
        for x, log_pob in zip(cands, probs):
            for key in semantic.keys():
                if x in key:
                    break
            try:
                print(key)
            except:
                continue
            idx = semantic[key]
            try:
                log_pobs[idx][x].append(log_pob)
            except:
                log_pobs[idx][x] = [log_pob]
        unct = 0
        for indx,val in log_pobs.items():
            for key,x in val.items():
                x = sum(x)/len(x)
                val[key] = math.exp(x)
            sum_probs = sum(val.values())
            if sum_probs>1: sum_probs = 1
            if sum_probs <= 0: sum_probs = 1e-6
            unct += math.log(sum_probs)#*sum_probs # sum probability then take log
        if len(log_pobs) == 0:
            return 0
        unct = -unct/len(log_pobs) # average over number of semantic classes
        return unct

    def deberta(self, cands, cands2):
        total_cands = [c2.lower() + ' ' + c1.lower() for c1,c2 in zip(cands,cands2)]
        unique_generated_texts = list(set(total_cands))
        answer_list_1 = []
        answer_list_2 = []
        inputs = []
        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index
        question = 'robot should'
        deberta_predictions = []
        if len(unique_generated_texts) > 1:
            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):

                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = self.tokenizer.encode(input, padding=True)
                    prediction = self.model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = self.tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = self.model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    deberta_prediction = 1
                    # print(qa_1,'|', qa_2, predicted_label, reverse_predicted_label)
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0

                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                    deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])
        # print(deberta_predictions)
        return semantic_set_ids

    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot.{}({})".format(subject,object)
        else:
            next_line = "    " + task +"\n"
        self.new_lines += next_line

    def load(self,
            ckpt_dir: str,
            tokenizer_path: str,
            local_rank: int,
            world_size: int,
            max_seq_len: int,
            max_batch_size: int,
        ):
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        generator = LLaMA(model, tokenizer)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator