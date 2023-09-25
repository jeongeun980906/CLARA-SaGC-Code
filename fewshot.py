from llm.lnct_score import lm_planner_unct
from llm.chat import lm_planner_unct_chat
import json
import random
import argparse
from set_key import set_openai_api_key_from_txt
AGENT_NAME = {
    1:"cooking", 2:"cleaning", 3:"massaging"
}

def main(args):
    set_openai_api_key_from_txt('./key/key.txt')
    with open('./data/agument.json','r') as f:
        env = json.load(f)
    model_name = args.llm
    unct_type = 6
    random.seed(0)
    if model_name == 'gpt':
        planners = [lm_planner_unct(type=1, task = 'cook')
                    , lm_planner_unct(type=1, task = 'clean')
                    ,lm_planner_unct(type=1, task = 'mas')]
    if model_name == 'chat':
        planners = [lm_planner_unct_chat(type=2, task = 'cook')
                    ,lm_planner_unct_chat(type=2, task = 'clean')
                    ,lm_planner_unct_chat(type=2, task = 'mas')]
    if args.start != 0:
        with open('./res/{}_{}.json'.format(model_name, unct_type),'r') as f:
            res = json.load(f)
    else:
        res = {}
    last = len(env)
    for i in range(args.start, last):
        data = env[str(i)]
        if data['label'] == 3:
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

        step_texts, _ = planner.infer_wo_unct(stop=False)

        res[i] = dict(text=step_texts)
        # break
        with open('./res/{}_{}.json'.format(model_name, unct_type),'w') as f:
            json.dump(res, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt')
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()
    main(args)