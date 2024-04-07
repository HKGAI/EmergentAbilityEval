from huggingface_hub import list_repo_refs
import pprint

# 获取模型的所有分支和标签
repo_name = "allenai/OLMo-1B"

# 指定要写入的文件名
filename = "hf_llama_7b.py"

refs = list_repo_refs(repo_name)

# 获取所有分支和标签的名称
branches = [b.name for b in refs.branches]
tags = [t.name for t in refs.tags]

all_refs = branches + tags

models = []

for ref in all_refs:
    model_name = f"{ref}"  
    model_path = f"{repo_name}"
    tokenizer_path = f"{repo_name}"
    revision_path = f"{ref}"

    # 创建模型配置并添加到models列表
    models.append({
        'type': 'HuggingFaceCausalLM',
        'abbr': model_name, 
        'path': model_path, 
        'tokenizer_path': tokenizer_path,
        'tokenizer_kwargs': {
            'padding_side': 'left',
            'truncation_side': 'left',
            'use_fast': False,
            'trust_remote_code': True,
        },
        'max_out_len': 100,
        'max_seq_len': 4096,
        'batch_size': 16,
        'model_kwargs': {
            'device_map': 'auto',
            'trust_remote_code': True,
            'revision': revision_path,
        },
        'batch_padding': False,
        'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    })


# models_str = json.dumps(models, indent=4)

# python_code = f"models = {models_str}"
# 使用pprint的pformat来格式化models对象为一个字符串
python_code = f"models = {pprint.pformat(models, indent=4)}"

# 写入文件
with open(filename, "w") as file:
    file.write(python_code)

print(f"Models have been written to {filename}")