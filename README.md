# LIT TEXT CLASSIFER
with BERT


# STEP 1: Finetune Pretrained Bert Model

```sh
python train.py --gpus 1
```

# STEP 2: Inference

```sh
python inference.py --model [PATH TO YOUR MODEL CKPT FILE]
```