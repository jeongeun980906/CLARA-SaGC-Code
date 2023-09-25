import openai
import spacy
import scipy
import random
import numpy as np
import copy
from llm.affordance import affordance_score2
import time
from llm.prompts import get_prompts, PROMPT_STARTER

AGENT_NAME = {
    'cook':"cooking", 'clean':"cleaning", 'mas':"massaging"
}

class lm_planner_unct_chat():
    def __init__(self, type = 2, example = False, task= 'cook'):
        self.few_shots = get_prompts(example)
        self.type = type
        self.new_lines = ""
        self.nlp = spacy.load('en_core_web_lg')
        self.type = type
        self.verbose = True
        self.task = task
        self.objects = ["blue block", "red block", "yellow bowl", "green block", "green bowl",'blue bowl']
        self.people = ["person with yellow shirt", "person with black shirt", "person with white shirt"]
        self.floor_plan = []
        self.set_func()
        
    def set_func(self):
        if self.type == 2 or self.type == 4:
            self.plan_with_unct = self.plan_with_unct_type2
        elif self.type == 7:
            self.plan_with_unct = self.plan_with_unct_type6
        else:
            raise NotImplementedError

    def plan_with_unct_type6(self, verbose = False):
        self.set_prompt()
        object = ""
        action = ""
        # Only one beam search? -> N samples
        while (len(object) < 3 or len(action)< 3):
            object, action = self.inference()
            # print(object_probs,subject_probs)
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += "robot action: robot.{}({})\n".format(action, object)
        inp += "robot thought: Can the robot do to this task please answer in yes or no?\nrobot thought: "
        ans = ""
        while len(ans)<3:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": inp}], 
                    temperature = 0.8, top_p = 1, n = 3, stop=')'
                    )
                
            except:
                pass
            ans = response['choices'][0]['message']['content']
        if ans[0] ==" ":
            ans = ans[1:]
        print(ans)
        if 'no' in ans.lower().replace(".","").replace(",",""):
            ood_unct = 1
            amb_unct = 0 
        else:
            ood_unct =  0
            inp = copy.deepcopy(self.prompt)
            inp += self.new_lines
            inp += "robot action: robot.{}({})\n".format(action, object)
            inp += "robot thought: Is this ambiguous and need more information from the user please answer in yes or no?\nrobot thought:"
            ans = ""
            while len(ans)<2:
                try:
                    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": inp}], 
                    temperature = 0.8, top_p = 1, n = 3, stop=')'
                    )
                except:
                    pass
            ans = response['choices'][0]['message']['content']
            if ans[0] ==" ":
                ans = ans[1:]
            print(ans)
            if 'yes' in ans.lower().replace(".","").replace(",",""):
                amb_unct = 1 
            else:
                amb_unct =  0
        unct = {'ood':ood_unct, 'amb':amb_unct, 'total': ood_unct+amb_unct}
        print(unct)
        return ['robot action: robot.{}({})'.format(action, object)], [1],unct



    def plan_with_unct_type2(self, verbose= False):
        obj_cand = []
        subj_cand = []
        self.verbose = verbose
        goal_num = 5
        if self.type == 4:
            self.set_prompt()
        while len(obj_cand) <1 or len(subj_cand)<1:
            for _ in range(goal_num):
                if self.type == 2:
                    self.sample_prompt()
                object, subject = self.inference()
                print(object, subject)
                if len(object) != 0:
                    obj_cand += object
                    subj_cand += subject
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
        # print(obj_cand,subj_cand)
        obj2 = self.get_word_diversity(obj_cand)
        sub2 = self.get_word_diversity(subj_cand)
        # print(obj2, sub2)
        unct= {
            'obj' : obj2 /5,
            'sub': sub2/5,
            'total': (obj2+sub2)/5
        }

        return tasks, scores, unct

    
    def set_goal(self, goal):
        self.goal = goal

    def set_prompt(self,choices=None):
        des = PROMPT_STARTER[self.task]
        des += "Follow are examples"
        if choices == None:
            choices = self.few_shots
        for c in choices:
            des += c
        des += "\nFrom this, predict the next action with considering the role of the robot\n" # and the ambiguity of the goal
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
        k = random.randrange(2,lengs+1)
        A = np.arange(lengs)
        A = np.random.permutation(A)
        choices = []
        for i in range(k):
            choices.append(self.few_shots[A[i]])
        if self.verbose:
            print('select {} few shot prompts'.format(k))
        random.shuffle(self.objects)
        random.shuffle(self.people)
        self.set_prompt(choices)
        # print(self.prompt)

    def append_reason_and_question(self, reason, question):
        self.new_lines += '\nrobot thought: this code is uncertain because ' + reason + '\n'
        self.new_lines += 'robot thought: what can I ask to the user? \nquestion: please' + question

    def append_question_fail(self):
        self.new_lines += '\nrobot thought: the question is not clear enough, please rephrase it \n'

    def inference(self):
        while True:
            print("resp")
            try:
                inp = self.prompt + "robot action: robot."
                # print(inp)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": inp}], 
                    temperature = 0.8, top_p = 1, n = 3, stop=')'
                    )
                break
            except:
                time.sleep(1)
                continue
        objects = []
        subjects = []
        results = response['choices']
        for res in results:
            res = res['message']['content']
            if "done()" in res:
                objects.append("done")
                subjects.append("done")
            # if "robot action: robot." not in res:
            #     continue
            try:
                place, pick = res.split("(")
                # print(res)
                # pick, place = res.split(".")
                # place = place.split("(")[1]#.split(")")[0]
            except:
                continue
            objects.append(pick)
            subjects.append(place)
        return objects, subjects
        
    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot action: robot.{}({})".format(subject,object)
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
        form = '\nrobot thought: I am a {} robot. Considering the action set, can I {}?'.format(AGENT_NAME[self.task],self.goal)
        form += "\nrobot thought: "
        # self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        # inp += self.new_lines
        inp += form
        while True:
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": inp}], 
                temperature = 0.5, top_p = 1, n = 1, stop=':'
                )
                break
            except:
                time.sleep(1)
                continue
        affor = response['choices'][0]['message']['content'].split('\n')[0]
        print(affor)
        temp = affor.lower().replace(".","").replace(",","").split(' ')
        if 'no' in temp or 'cannot' in temp or 'can not' in temp or "can't" in temp:
            return None, None, affor
        form =  '\nrobot thought: This is uncertain because'
        # self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += form
        while True:
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": inp}], 
                temperature = 0.5, top_p = 1, n = 1, stop=':'
                )
                break
            except:
                time.sleep(1)
                continue
        reason = response['choices'][0]['message']['content'].split('\n')[0]
        print('reason: ',reason)
        inp += reason
        self.new_lines += reason + '\n'
        ques = 'robot thought: what can I ask to the user? \nquestion: please'
        inp += ques
        self.new_lines += ques
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": inp}], 
                    temperature = 0.5, top_p = 1, n = 1, stop='\n'
                    )
                break
            except:
                time.sleep(1)
                continue
        ques = response['choices'][0]['message']['content']
        ques = ques.split('\n')[0]
        print('question: please',ques)
        self.new_lines += ques
        return reason, ques, affor
    
    def answer(self, user_inp):
        self.new_lines += '\nanswer:' + user_inp
        self.new_lines += '\robot thought: continue the previous task based on the question and answer'

    def reset(self):
        self.new_lines = ""

    def infer_wo_unct(self, found_objects, stop=True):
        done = False
        max_tasks=5
        cont = 0
        res = []
        ask_flag = False
        for _ in range(50):
            if done:
                break
            self.set_prompt()
            if cont > max_tasks:
                break
            while True:
                try:
                    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": self.prompt}], 
                    temperature = 0.8, top_p = 1, n = 3, stop = 'done()'
                    )
                    break
                except:
                    time.sleep(1)
                    continue
            text = response['choices'][0]['message']['content'].split('\n')
            
            for line in text:
                if 'done' in line:
                    done = True
                    break
                if "robot action: robot." in line:
                    cont += 1
                    res.append(line)
                    self.append(None, None, line)
                elif "robot thought:" in line:
                    res.append(line)
                    self.append(None, None, line)
                elif "question:" in line and stop:
                    res.append(line)
                    ask_flag = True
                    done = True
                    break
        print(res)
        return res, ask_flag