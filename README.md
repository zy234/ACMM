# A Word-granular Adversarial Attacks Framework for Causal Event Extraction

This is the **Pytorch Implementation** for the paper A Word-granular Adversarial Attacks Framework for Causal Event Extraction

## Requirement
- Python 3.8.3
- numpy==1.18.5
- seqeval==1.2.2
- torch==1.4.0
- transformers==4.0.0

## Dataset
|            |  SemEval    |  Causal TB |
|:-----------|------------:|:----------:|
|            |Train  Test  |Train  Test |
|Cause-Effect|  904   421  |  220   181 |
|Other       | 6574  2775  |  202    78 |
|All         | 7478  3196  |  422   103 |

## How to run
1. train
```
python main.py --eval=False --data_path='./data' --mask_warm_up=10 --encoder_warm_up=20 --batch_size=8 --epochs=130 --patience=20
```
2. eval
```
python main.py --eval=True --data_path='./data' --eval_path=model_path
```
