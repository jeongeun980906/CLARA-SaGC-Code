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
import copy
import logging
from llm.affordance import affordance_score2
from llm.prompts import get_prompts, PROMPT_STARTER

class llama_unct():
    def __init__(self, 
                ckpt_dir: str,
                tokenizer_path: str,
                temperature: float = 0.8,
                top_p: float = 0.95,
                local_rank: int = 1,
                world_size: int = 1,
                max_seq_len: int = 512,
                max_batch_size: int = 32,
                task: str = 'cook',
                method: int =1,
                example:bool = False):
        self.method = method
        self.generator = self.load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.task = task
        self.examples = example
        # logging.basicConfig(level=0)
        self.new_lines = ""
        self.verbose = True
        self.nlp = spacy.load('en_core_web_lg')
        self.normalize = False
        self.objects = ["blue block", "red block", "yellow bowl", "green block", 
                    "green bowl",'blue bowl']
        self.people = []
        self.floor_plan = []
        self.set_func()
        

    def set_func(self):
        if self.method == 1 or self.method == 3:
            self.plan_with_unct = self.plan_with_unct_type1
            if self.method == 3:
                self.normalize = True
        elif self.method == 2 or self.method ==4:
            self.plan_with_unct = self.plan_with_unct_type2
        elif self.method == 7:
            self.plan_with_unct = self.plan_with_unct_type6
        else:
            raise NotImplementedError

    def set_goal(self, goal):
        self.goal = goal
        self.few_shots = copy.deepcopy(get_prompts(self.examples, self.task))

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

    def sample_prompt(self):
        lengs = len(self.few_shots)
        # print(lengs)
        k = random.randrange(3,lengs+1)
        A = np.arange(k)
        A = np.random.permutation(lengs)
        # print(A, k)
        choices = []
        for i in range(k):
            choices.append(self.few_shots[A[i]])
        # print(choices)
        if self.verbose:
            print('select {} few shot prompts'.format(k))
        # choices = self.few_shots#[:4]
        random.shuffle(self.objects)
        random.shuffle(self.people)
        random.shuffle(self.floor_plan)
        self.set_prompt(choices)

    def load(self,
            ckpt_dir: str,
            tokenizer_path: str,
            local_rank: int,
            world_size: int,
            max_seq_len: int,
            max_batch_size: int,
        ) -> LLaMA:
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


    def plan_with_unct_type1(self, verbose= False):
        self.verbose = verbose
        self.set_prompt()
        pick = ""
        action = ""
        while len(pick)==0 and len(action)==0:
            pick, pick_entp, action, action_entp = self.inference()
        unct = {
            'obj': pick_entp, 
            'sub':action_entp,
            'total': (pick_entp+action_entp)
        }
        if self.verbose:
            print(unct)
        return ['robot action: robot.{}({})'.format(action,pick)], [1], {'ood':0, 'amb':0, 'total': 0}

    def plan_with_unct_type6(self, verbose= False):
        self.verbose = verbose
        self.set_prompt()
        pick = ""
        action = ""
        for _ in range(10):
            if len(pick)==0 and len(action)==0:
                pick, _, action, _ = self.inference()
            else:
                break
        if len(pick)==0: pick = 'done'
        if len(action) == 0: action = 'done' 
        if action == 'done' or pick == 'done':
            return ['robot action: robot.{}({})'.format(action,pick)], [1], {'ood':0, 'amb':0, 'total': 0}
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += "robot action: robot.{}({})\n".format(action, object)
        inp += "robot thought: Can the robot do to this task please answer in yes or no?\nrobot thought: the answer is"
        ans = ""
        while len(ans)<1:
            response,_,_,_ = self.generator.generate(
            [inp], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
            )
            # print(response)
            ans = response[0].split("\n")[0]
        print(ans)
        if 'no' in ans.lower().replace(".","").replace(",",""):
            ood_unct = 1
            amb_unct = 0 
        else:
            ood_unct =  0
            inp = copy.deepcopy(self.prompt)
            inp += self.new_lines
            inp += "robot action: robot.{}({})\n".format(action, object)
            inp += "robot thought: Is this ambiguous and need more information from the user please answer in yes or no?\nrobot thought: the answer is"
            ans = ""
            while len(ans)<1:
                response,_,_,_ = self.generator.generate(
                [inp], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
                )
                ans = response[0].split("\n")[0]
            print(ans)
            if 'yes' in ans.lower().replace(".","").replace(",",""):
                amb_unct = 1
            else:
                 amb_unct = 0
        unct = {'ood':ood_unct, 'amb':amb_unct, 'total': ood_unct+amb_unct}
        if self.verbose:
            print(unct)
        return ['robot action: robot.{}({})'.format(action,pick)], [1], unct


    def plan_with_unct_type2(self, verbose= False):
        obj_cand = []
        subj_cand = []
        self.verbose = verbose
        goal_num = 5
        inf_num = 3
        if self.method ==4:
            self.set_prompt()
        for _ in range(goal_num):
            if self.method ==2:
                self.sample_prompt()
            for _ in range(inf_num):
                pick,_, place,_ = self.inference()
                if len(pick) != 0:
                    obj_cand.append(pick)
                if len(place) != 0:
                    subj_cand.append(place)
        tasks = []
        scores = []
        for x,y in zip(obj_cand, subj_cand):
            prompt = 'robot action: robot.{}({})'.format(y,x)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1
        scores = [s/sum(scores) for s in scores]
        print(obj_cand, subj_cand)
        if len(obj_cand)>0:
            obj2 = self.get_word_diversity(obj_cand)
        else: obj2 = 0
        if len(subj_cand)>0:
            sub2 = self.get_word_diversity(subj_cand)
        else: sub2 = 0
        # print(obj2, sub2)
        unct= {
            'obj' : obj2 /5,
            'sub': sub2/5, 
            'total': (obj2+sub2)/5
        }
        if self.verbose:
            print(unct)
        return tasks, scores, unct

    def inference(self):
        # print(self.prompt)
        results, tokens, entropties,_ = self.generator.generate(
                [self.prompt], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
               )
        # print("inference done", self.prompt)
        flag = False
        flag2 = False
        subject_entp = 0
        object_entp = 0
        subject_num = 0
        object_num = 0
        task = results[0]
        # print(task)
        task = task.replace("        ","")
        task = task.replace("    ","")
        task = task.split("\n")
        flag3 = False
        i = 0
        for t in task:
            for i,l in enumerate(t):
                if l != ' ' and l != '\n':
                    flag3 = True
                    break
            if flag3:
                break
        task = t[i:]
        print(task)
        if "done()" in task:
            object = 'done'
            subject = 'done'
            subject = ""
        elif 'robot' not in task:
            return "", 0, "", 0
        else:
            try:
                object, subject = task.replace("robot action: robot.", "").split("(")
                subject = subject.split(")")[0]
            except:
                object = ""
                subject = "" 
        if self.method == 1 or self.method == 3:
            for tok, entropty in zip(tokens[1:], entropties[1:]):
                entropty = entropty.item()
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
                    if tok !="":
                        object_entp += entropty
                        object_num += 1
                elif flag2 and not flag:
                    if tok !="":
                        subject_entp += entropty
                        subject_num += 1
                if tok == "(":
                    flag = True
        elif self.method == 2:
            pass

        if self.normalize:
            if object_num != 0:
                object_entp /= object_num
            else:
                object_entp = 0
            if subject_num != 0:
                subject_entp /= subject_num
            else:
                subject_entp = 0
        
        print(object,object_entp, "|",subject,subject_entp)
        return subject, subject_entp,object, object_entp

    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot.{}({})".format(subject,object)
        else:
            next_line = "\n" + task
        self.new_lines += next_line

    def get_word_diversity(self, words):
        vecs = []
        size = len(words)
        for word in words:
            vec = self.nlp(word).vector
            vecs.append(vec)
        vecs = np.vstack(vecs)
        dis = scipy.spatial.distance_matrix(vecs,vecs)
        div = np.sum(dis)/((size)*(size-1))
        # print(div, dis)
        return div

    def question_generation(self):
        form = '\robot thought: I am a {} robot. Considering the action set, Can I {} Please answer in yes or no?'.format(AGENT_NAME[self.task],self.goal)
        form += "\nrobot thought: The answer is"
        inp = copy.deepcopy(self.prompt)
        inp += form
        affor = ""
        while len(affor)<2:
            response,_,_,_ = self.generator.generate(
            [inp], max_gen_len=200, temperature=self.temperature, top_p=self.top_p
            )
            affor = response[0].split("\n")[0]
        if affor[0] ==" ":
            affor = affor[1:]
        temp = affor.lower().replace(".","").replace(",","").split(' ')
        if 'no' in temp or 'cannot' in temp or 'can not' in temp or "can't" in temp:
            return None, None, affor
        form =  '\nrobot thought: This is uncertain because'
        # self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += form
        reas = ""
        while len(reas)<2:
            response,_,_,_ = self.generator.generate(
            [inp], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
            )
            reas = response[0].split("\n")[0]
        reas = reas.replace("robot thought: ","")
        inp += reas
        self.new_lines += reas + '\n'
        ques = 'robot thought: what can I ask to the user? \nquestion: please'
        inp += ques
        self.new_lines += ques
        response,_,_,_ = self.generator.generate(
            [inp], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
            )
        ques = response[0].split("\n")[0]
        return reas, ques, affor
    
    def answer(self, user_inp):
        self.new_lines += '\n    answer:' + user_inp
        self.new_lines += '\n    robot thought: continue the previous task based on the question and the answer.'

    def reset(self):
        self.new_lines = ""

    def append_reason_and_question(self, reason, question):
        self.new_lines += '\n    robot thought: this code is uncertain because ' + reason + '\n'
        self.new_lines += '    robot thought: what can I ask to the user? \n    question: please' + question

    
    def infer_wo_unct(self,task=None, stop=True):
        done = False
        res = []
        max_tasks=10
        cont = 0
        for _ in range(3):
            if done:
                break
            self.prompt = ""
            self.set_prompt()    
            
            print(self.prompt)
            results, _, _,_ = self.generator.generate(
                    [self.prompt], max_gen_len=self.max_seq_len, temperature=self.temperature, top_p=self.top_p
                )
            text = results[0].replace("        ","").split("\n")
            for line in text:
                cont += 1
                if "done" in line:
                    done = True
                    break
                elif "robot action: robot." in line:
                    res.append(line)
                    self.append(None, None, line)
                elif "robot thought:" in line:
                    res.append(line)
                    self.append(None, None, line)
                elif "question:" in line:
                    res.append(line)
                    if stop:
                        done = True
                        break
                if cont > max_tasks:
                    break
        return res