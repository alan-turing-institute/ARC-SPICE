# Scripts

## create_test_ds.py

Create dataset used in testing. This has been run and output stored in GitHub.

## train_topic_classifier.py

Train a topic classifier over MultiEURLEX data.

Sample config for training stored at [../config/topic_classifier_training.yaml](../config/topic_classifier_training.yml). Edit this first.

run with (from project root):
```bash
python scripts/train_topic_classifier.py --config config/topic_classifier_training.yml
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

## pipeline_inference.py

Run inference on a complete pipeline over the MultiEURLEX dataset. Requires a data config path and a pipeline config path,
this should be `.yaml` files and should be structured as such:

### Dataset config:

```yaml
data_dir: "data"

level: 1

lang_pair:
  source: "fr"
  target: "en"
```
### Pipeline config:

```yaml
OCR:
  specific_task: "image-to-text"
  model: "microsoft/trocr-base-handwritten"

translator:
  specific_task: "translation_fr_to_en"
  model: "ybanas/autotrain-fr-en-translate-51410121895"

classifier:
  specific_task: "zero-shot-classification"
  model: "claritylab/zero-shot-explicit-binary-bert"

```

It's called like so e.g. from project root:
```bash
python scripts/pipeline_inference.py [pipeline_config_path] [data_config_path]
```

## single_component_inference.py

Run inference on a single component of the pipeline over the MultiEURLEX dataset. Requires a data config path and a pipeline config path,
which should be `.yaml` files structured as below. It also takes an additional argument specifying the specific pipeline stage to be evaluated.
This should be one of `ocr`, `translator`, or `classifier`.

### Dataset config:

```yaml
data_dir: "data"

level: 1

lang_pair:
  source: "fr"
  target: "en"
```
### Pipeline config:

```yaml
OCR:
  specific_task: "image-to-text"
  model: "microsoft/trocr-base-handwritten"

translator:
  specific_task: "translation_fr_to_en"
  model: "ybanas/autotrain-fr-en-translate-51410121895"

classifier:
  specific_task: "zero-shot-classification"
  model: "claritylab/zero-shot-explicit-binary-bert"

```

It's called like so e.g. from project root:
```bash
python scripts/pipeline_inference.py [pipeline_config_path] [data_config_path] translator
```

## gen_jobscripts.py

Create jobscript `.sh` files for an experiment, which in this case refers to a `data_config` and `pipeline_config` combo.
It takes a single argument which is `experiment_config_path`. This refers to a file path to a `.yaml` file structured as below:

### eg. Experiment config:

```yaml
data_config: l1_fr_to_en

pipeline_config: roberta-mt5-zero-shot

seed:
  - 42
  - 43
  - 44

bask:
  jobname: "full_experiment_with_zero_shot"
  walltime: '0-24:0:0'
  gpu_number: 1
  node_number: 1
  hf_cache_dir: "/bask/projects/v/vjgo8416-spice/hf_cache"


```
