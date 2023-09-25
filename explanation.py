from llm.lnct_score import lm_planner_unct
import json
import argparse
from llm.chat import lm_planner_unct_chat
from set_key import set_openai_api_key_from_txt
AGENT_NAME = {
    1:"cooking", 2:"cleaning", 3:"massaging"
}
THRES = {
    1:0.80, 2:0.6, 3:0.4
    # 1:2.4 , 2:3.4, 3:2.9
}
def main(args):
    set_openai_api_key_from_txt('./key/key.txt')
    with open('./data/agument.json','r') as f:
        env = json.load(f)
    unct_type =  2
    if args.llm == 'gpt':
        planners = [lm_planner_unct(type=unct_type, task = 'cook')
                            , lm_planner_unct(type=unct_type, task = 'clean')
                            ,lm_planner_unct(type=unct_type, task = 'mas')]
        model_name = 'gpt'
    if args.llm == 'chat':
        planners = [lm_planner_unct_chat(type=unct_type, task = 'cook')
                            ,lm_planner_unct_chat(type=unct_type, task = 'clean')
                            ,lm_planner_unct_chat(type=unct_type, task = 'mas')]
        model_name = 'chat'

    with open('./res/{}_{}.json'.format(model_name, unct_type),'r') as f:
        old_res = json.load(f)

    if args.start != 0:
        # res = {}
        with open('./res/{}_{}_inter.json'.format(model_name, unct_type),'r') as f:
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

        i = str(i)
        last_line = old_res[i]['step']
        uncts = old_res[i]['unct']
        reason, ques, affor = None, None, None
        thres = THRES[task_num]
        for unct, text in zip(uncts, last_line):
            print(text)
            if unct['total'] > thres:
                planner.set_prompt()
                reason, ques, affor = planner.question_generation()
                break
            planner.new_lines = text
        if len(uncts) == 0:
            planner.set_prompt()
            reason, ques, affor = planner.question_generation()
        res[i] = dict(scene=data['scene'], goal= data['goal'],
                    reason=reason, ques=ques,cap = affor)

        with open('./res/{}_{}_inter.json'.format(model_name, unct_type),'w') as f:
            json.dump(res, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt')
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()
    main(args)