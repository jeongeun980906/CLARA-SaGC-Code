from llm.affordance import affordance_score2
import numpy as np
import time
from llm.utils import *

def run(planner, task = None):
    done = False
    num_tasks = 0
    steps_text = []
    uncts = []
    max_tasks = 5
    planner.reset()
    flag = True
    while not done:
        num_tasks += 1
        if num_tasks > max_tasks:
            break
        aff = 0
        if not flag: break
        flag = False
        for _ in range(2):
            selected_task, aff, unct, done = one_loop(planner, task)
            if aff != 0: flag = True; break
        if done:
            break
        # selected_task = selected_task.replace("robot action: ","")
        print(selected_task, aff, unct)
        steps_text.append(selected_task)
        uncts.append(unct)
        planner.append(None, None, selected_task)
    # print(steps_text, uncts)
    return steps_text, uncts

def run_interactive(planner, found_objects,thres = 1, max_ques = 5, gui = None):
    done = False
    num_tasks = 0
    uncts = []
    questions = []
    reasons = []
    answers = []
    texts = []
    max_tasks = 5
    planner.reset()
    num_ques = 0
    ask_flag = False
    while not done:
        num_tasks += 1
        if num_tasks > max_tasks:
            break
        selected_task, aff, unctertainty, done = one_loop(planner, found_objects)
        if gui != None:
            gui.inser_text(selected_task,'pred')
            gui.inser_text("Uncertainy: "+str(unctertainty['total']),'unct')
        print(unctertainty, selected_task, aff, done, thres)
        uncts.append(unctertainty['total'])
        texts.append(selected_task)
        if done:
            break
        # print("?",unctertainty['total'] > thres , (num_ques < max_ques))
        if unctertainty['total'] > thres and (num_ques < max_ques):
            planner.new_lines += selected_task
            reason, question = planner.question_generation(ask_flag)
            if gui != None:
                if reason != None:
                    gui.inser_text("Reason: "+ reason,'reas')
                gui.inser_text("Question: "+question,'reas')
                gui.inser_text("Type answer",'comp')
                gui.root.update()
                while not gui.inpt_flag:
                    gui.root.update()
                inp = gui.user
            else:
                inp = input("Answer: ")
            questions.append(question)
            reasons.append(reason)
            planner.answer(inp)
            num_ques += 1
            ask_flag = True
            answers.append(inp)
            continue
        else:
            planner.append(None, None, selected_task)
            break
    pick, place = selected_task.split("robot action: robot.(")[1].split(")")[0].split(", ")
    pick = get_word_diversity(planner,pick, planner.objects)
    place = get_word_diversity(planner,place, planner.people+['bottom left corner', 'bottom right corner','ground'])
    return pick, place, questions, reasons, answers, texts, uncts

def few_shot_run(planner, found_objects, max_ques = 5):
    num_ques = 0
    done = False
    texts = []
    answers = []
    sys_fail = False
    while num_ques < max_ques and not done:
        ctexts,ask_flag = planner.infer_wo_unct(found_objects, stop=True)
        texts += ctexts
        for t in texts:
            planner.new_lines += t +"\n"
            if 'robot action: robot.' in t and not ask_flag:
                print('done')
                done = True
                break
        if not ask_flag and not done:
            print('sys failed')
            sys_fail = True
            break
        if ask_flag:
            num_ques += 1
            answer = input("Answer: ")
            planner.answer(answer)
            answers.append(answer)
    return texts,answers,

def one_loop(planner, task):
    tasks, scores , unct = planner.plan_with_unct()
    if tasks != None:
        scores = np.asarray(scores)
        idxs= np.argsort(scores)
        print(tasks, scores)
        if len(tasks) > 0:
            for idx in idxs[::-1]:
                print(tasks[idx], scores[idx])
                aff = affordance_score2(tasks[idx], task)
                if aff >0:
                    break
            if aff == 2: 
                done=True
            else:
                done = False
            return tasks[idx],aff, unct, done
        else:
            return None, 2 , None , True
    else:
        return None, 2 , None , True
