# CLARA - SaGC dataset code

**Before you run, please set your key in key/key.txt file**

## Uncertainty Quantification
To run the uncertainty quantification run

text-davinci-003
```
python main.py --llm gpt
```
gpt-3.5-turbo
```
python main.py --llm chat
```

For the LLaMA
install the model from the official repository fist [LLaMa](https://github.com/facebookresearch/llama)

```
torchrun --nproc_per_node 1 llama_main.py --ckpt_dir [YOUR PATH]/7B --tokenizer_path [YOUR PATH]/tokenizer.model --unct_type 2
```

## Classification

To run the classification and disambiguation process, 
text-danvinci-003
```
python explanation.py 
```
gpt3.5-turbo
```
python explanation.py --llm chat
```
LLaMa
```
torchrun --nproc_per_node 1 llama_inter.py --ckpt_dir [YOUR PATH]/7B --tokenizer_path [YOUR PATH]/tokenizer.model --unct_type 2
```