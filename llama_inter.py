from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

import random
import numpy as np
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llm.llama import llama_unct
import copy
from llm.run import run

from llm.llama_semantic import semantic_uncertainty

AGENT_NAME = {
    "cooking":'cook', "cleaning":'clean', "massaging":'mas'
}
THRES = {
    1:1.8, 2:1.6, 3:0.2
}
TASK_NUMS = {
    "cooking":1, "cleaning":2, "massaging":3
}
def setup_model_parallel(gpu_index) -> Tuple[int, int]:
    # print(os.environ.get("LOCAL_RANK", -1))
    print(os.environ["MASTER_ADDR"])
    print(os.environ["MASTER_PORT"]) # 127.0.0.1:29500
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank+gpu_index)

    # seed must be the same in all processes
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    return local_rank, world_size
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    unct_type: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.95,
    max_seq_len: int = 800,
    max_batch_size: int = 32,
    start_idx: int = 0,
    end: int = 0,
    gpu_index: int = 0,
    new_index: int = 0
):
    model_name = 'llama'
    local_rank, world_size = setup_model_parallel(gpu_index)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    unct_type =  2
    planner = llama_unct(ckpt_dir, tokenizer_path, temperature, top_p, 
                local_rank, world_size, max_seq_len, max_batch_size)
    
    planner.method = unct_type
    planner.set_func()
    with open('./data/agument.json','r') as f:
        env = json.load(f)
    with open('./res/{}_{}.json'.format(model_name, unct_type),'r') as f:
        old_res = json.load(f)

    if start_idx != 0:
        # res = {}
        try:
            with open('./res/{}_{}_{}_inter.json'.format(model_name, unct_type,10*new_index+ gpu_index+1),'r') as f:
                res = json.load(f)
        except:
            res = {}
    else:
        res = {}
    last = len(env)
    if end != 0:
        last = end
    for i in range(start_idx, last):
        data = env[str(i)]
        if data['label'] == 3:
            continue

        task_agent_name = data['task']
        task_num = TASK_NUMS[task_agent_name]
        planner.reset()
        planner.task = AGENT_NAME[task_agent_name]

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
            if str(unct['total']) == 'nan':
                continue
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
        with open('./res/{}_{}_{}_inter.json'.format(model_name, unct_type,10*new_index+ gpu_index+1),'w') as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)