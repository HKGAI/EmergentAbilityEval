# OpenCompass 基于HuggingFace🤗大模型评测

**构建一键式脚本对大模型效果进行评测**

- 🧶 目前使用llama-7b-hf进行实验，后续只需更改模型配置文件即可。

- 📈 对齐LLaMA 2的Evaluation此外加一下中文的一些Evaluation(主要是CMMLU和C-Eval)
- 📑 评价指标参考https://arxiv.org/pdf/2307.09288.pdf A2.2部分
- 📦 017机器容器：https://mgf127vt7ge.sg.larksuite.com/docx/YsfIdjeh5oqmM5xiKFll274Hg4e?from=from_copylink

更多详细信息请参阅lark文档：https://mgf127vt7ge.sg.larksuite.com/docx/J4W4djHR6oYPulx2mAQlhNZtgSd

## 🛠️ 安装

1. 虚拟环境配置

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
```

2. 下载opencompass

```bash
# 创建evaluation目录
mkdir evaluation
cd evaluation
# 下载
git clone https://github.com/HKGAI/EmergentAbilityEval.git opencompass
cd opencompass
```

3. 安装依赖

```bash
pip install -e .
```

4. 下载数据集到 data/ 

```bash
# wget https://github.com/open-compass/opencompass/releases/download/0.1.1/OpenCompassData.zip
# 下载0.1.8数据集
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

5. 下载humaneval数据集

```bash
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirements.txt
pip install -e .
cd ..
```

⚠️注意：使用humaneval时需要手动到`human-eval/human_eval/execution.py` 文件的第 58 行取消注释才能正常评测。

## 🏗️ 评测

确保按照上述步骤正确安装 OpenCompass 并准备好数据集后，可以通过以下命令评测 llama-7b-hf 模型在数据集上的性能：

```Bash
#命令行方式
python run.py eval_llama_7b_test.py -p slurm_conifg.py
#脚本方式
./eval_llama.sh
```
🏁提高GPU内存使用：设置max-workers-per-gpu（可以在hf_llama_7b.py中设置，也可以在命令行中直接传）
```Bash
nohup python run.py eval_llama_7b_test.py --max-workers-per-gpu 6 > xxx_log 2>&1
```
前后效果对比：
<div align="center">
    <img width="800" alt="image" src="https://github.com/HKGAI/EmergentAbilityEval/assets/114467558/3db98a13-ef41-4374-9f91-d7ba6206e141">
    <br />
默认max-workers-per-gpu=1
</div>

<div align="center">
    <img width="800" alt="image" src="https://github.com/HKGAI/EmergentAbilityEval/assets/114467558/57d9a5b1-3a31-4a6b-85ea-ebae98bc3212">
    <br />
max-workers-per-gpu=6
</div>


## 📖 结果

所有运行输出将定向到 `/home/hkustadmin/evaluation/opencompass/outputs/default/` 目录，结构如下：

```Plaintext
outputs/default/
├── 20231113_164612
├── 20231113_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果
├── ...
```

结果预览：

<div align="center">
  <img width="612" alt="截屏2023-11-15 16 07 24" src="https://github.com/HKGAI/opencompass/assets/114467558/4c601d41-ac3b-479c-8b9b-8563082c2b7c">
  <br />
</div>
