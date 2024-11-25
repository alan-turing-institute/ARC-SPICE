# Scripts

## create_test_ds.py

Create dataset used in testing. This has been run and output stored in GitHub.

## train_topic_classifier.py

Train a topic classifier over MultiEURLEX data.

Sample config for training stored at [../config/topic_classifier_training.yaml](../config/topic_classifier_training.yml). Edit this first.

run with (from project root):
```bash
python scripts/train_topic_classifier.py -c configs/topic_classifier_training.yml
```

## eval_topic_classifier.py

Evaluate a trained topic classifier over MultiEURLEX data.

e.g. from project root:
```bash
python scripts/eval_topic_classifier.py \
    --ckpt_path ./path/to/output/checkpoint-3286 \
    --data_root ./data \
    --batch_size 32 \
    --eval_output_dir ./path/to/output \
    --report_to tensorboard \
    --dataset_name validation # change to "test" for test set evaluation
```
