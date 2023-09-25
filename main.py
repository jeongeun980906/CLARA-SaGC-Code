from llm.lnct_score import lm_planner_unct
import json
from llm.run import run
from llm.semantic_unct import semantic_uncertainty
from llm.chat import lm_planner_unct_chat
from set_key import set_openai_api_key_from_txt
import random
import argparse

AGENT_NAME = {
    1:"cooking", 2:"cleaning", 3:"massaging"
}

def main(args):
    set_openai_api_key_from_txt('./key/key.txt')
    with open('./data/agument.json','r') as f:
        env = json.load(f)
    model_name, unct_type = args.llm, args.unct
    random.seed(0)
    if unct_type == 5:
        planners = [semantic_uncertainty(type=unct_type, task = 'cook')
                    ,semantic_uncertainty(type=unct_type, task = 'clean')
                    ,semantic_uncertainty(type=unct_type, task = 'mas')]
    if model_name == 'gpt':
        planners = [lm_planner_unct(type=unct_type, task = 'cook')
                    , lm_planner_unct(type=unct_type, task = 'clean')
                    ,lm_planner_unct(type=unct_type, task = 'mas')]
    if model_name == 'chat':
        planners = [lm_planner_unct_chat(type=unct_type, task = 'cook')
                    ,lm_planner_unct_chat(type=unct_type, task = 'clean')
                    ,lm_planner_unct_chat(type=unct_type, task = 'mas')]
    if args.start != 0:
        with open('./res/{}_{}.json'.format(model_name, unct_type),'r') as f:
            res = json.load(f)
    else:
        res = {}
    last = len(env)
    print("?")
    if args.end != 0:
        last = args.end
    for i in range(args.start, last):
        print(i)
        data = env[str(i)]
        if data['label'] == 3:
            print("skip")
            continue
        task_agent_name = data['task']
        for task_num, agent_name in AGENT_NAME.items():
            if agent_name == task_agent_name:
                break
        planner = planners[task_num-1]
        planner.reset()

        planner.objects = data['scene']['objects']
        planner.floor_plan = data['scene']['floorplan']
        planner.people = data['scene']['people']
        print(data['goal'])
        planner.set_goal(data['goal'])

        step_texts, unct = run(planner, task_num)
        print(step_texts, unct)
        i = str(i)
        res[i] = dict(scene=data['scene'], goal= data['goal'], task = data['task'], 
                    label = data['label'],
                    step=step_texts, unct=unct)
        # break
        with open('./res/{}_{}.json'.format(model_name, unct_type),'w') as f:
            json.dump(res, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt')
    parser.add_argument('--unct', type=int, default=2)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    args = parser.parse_args()
    main(args)