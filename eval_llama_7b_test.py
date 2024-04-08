from mmengine.config import read_base

with read_base():
    ########################DATASET##################
    # Standard Benchmarks
    from .configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets
    from .configs.datasets.piqa.piqa_ppl_0cfff2 import piqa_datasets
    from .configs.datasets.siqa.siqa_ppl_e8d8c5 import siqa_datasets
    from .configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    from .configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets
    from .configs.datasets.ARC_e.ARC_e_ppl_2ef631 import ARC_e_datasets
    from .configs.datasets.ARC_c.ARC_c_ppl_2ef631 import ARC_c_datasets
    from .configs.datasets.obqa.obqa_ppl_6aac9e import obqa_datasets
    from .configs.datasets.commonsenseqa.commonsenseqa_ppl_e51e32 import commonsenseqa_datasets
    from .configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    # Code Generation
    from .configs.datasets.humaneval.humaneval_gen_a82cae import humaneval_datasets
    from .configs.datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    # World Knowledge           need NaturalQuestions
    from .configs.datasets.nq.nq_gen_0356ec import nq_datasets
    from .configs.datasets.triviaqa.triviaqa_gen_0356ec import triviaqa_datasets
    
    # Reading Comprehension       need QUAC
    from .configs.datasets.squad20.squad20_gen_1710bc import squad20_datasets

    # Exams
    from .configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .configs.datasets.math.math_gen_265cce import math_datasets
    from .configs.datasets.TheoremQA.TheoremQA_gen_ef26ca import TheoremQA_datasets
    # ceval and cmmlu
    from .configs.datasets.ceval.ceval_ppl_578f8d import ceval_datasets
    from .configs.datasets.cmmlu.cmmlu_ppl_8b9c76 import cmmlu_datasets

    #########################MODEL###################
    from .hf_llama_7b import models

    ######################SUMMERIZER#################
    from .summarizer import dataset_abbrs, summary_groups

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = models

summarizer = dict(dataset_abbrs=dataset_abbrs, summary_groups=summary_groups, )
