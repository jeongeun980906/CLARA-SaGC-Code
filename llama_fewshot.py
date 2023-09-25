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


AGENT_NAME = {
    "cooking":'cook', "cleaning":'clean', "massaging":'mas'
}
TASK_NUMS = {
    "cooking":1, "cleaning":2, "massaging":3
}

# torchrun --nproc_per_node 2 llama_main.py --ckpt_dir ../../lam/13B --tokenizer_path ../../lam/tokenizer.model --unct_type 

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
    unct_type: int = 6,
    temperature: float = 0.5,
    top_p: float = 0.95,
    max_seq_len: int = 1000,
    max_batch_size: int = 32,
    start_idx: int = 0,
    end: int = 0,
    gpu_index: int = 0
):
    local_rank, world_size = setup_model_parallel(gpu_index)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    planner = llama_unct(ckpt_dir, tokenizer_path, temperature, top_p, 
            local_rank, world_size, max_seq_len, max_batch_size, example=True)

    planner.method = 1
    planner.set_func()
    with open('./data/agument.json','r') as f:
        env = json.load(f)

    if start_idx != 0:
        try:
            with open('./res/{}_{}_{}.json'.format('llama', unct_type,gpu_index+1),'r') as f:
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
        step_texts = planner.infer_wo_unct(stop=False)
        print(step_texts)
        res[i] = dict(text=step_texts)
        # break
        with open('./res/{}_{}_{}.json'.format('llama', unct_type,gpu_index+1),'w') as f:
            json.dump(res, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)