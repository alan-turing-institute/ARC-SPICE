"""
    An example use of the transcription, translation and summarisation pipeline.
"""
import json

import torch
from datasets import Audio, load_dataset

from arc_spice.dropout_utils.variational_inference import TTSVariationalPipeline


def main(TTS_params):
    """main function"""
    var_pipe = TTSVariationalPipeline(TTS_params)
    ds = load_dataset(
        "facebook/multilingual_librispeech", "french", split="test", streaming=True
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    input_speech = next(iter(ds))["audio"]

    clean_output = var_pipe.clean_inference(input_speech["array"])
    # logit shapes
    print("\nLogit shapes:")
    for step in var_pipe.pipeline_map.keys():
        print(f"{step.capitalize()}: {clean_output[step]["logits"].shape}")

    # entropy
    print("\nMean entropy:")
    for step in var_pipe.pipeline_map.keys():
        print(f"{step.capitalize()}: {torch.mean(clean_output[step]["entropy"])}")

    # normalised entropy
    print("\nNormalised mean entropy:")
    cumulative = 1
    for step in var_pipe.pipeline_map.keys():
        step_entropy = torch.mean(clean_output[step]["normalised_entropy"])
        cumulative*= (1-step_entropy)
        print(f"{step.capitalize()}: {step_entropy}")
    print(f"Cumulative confidence (1 - entropy): {cumulative}")

    # probabilities
    print("\nMean top probabilities:")
    cumulative = 1
    for step in var_pipe.pipeline_map.keys():
        step_prob = torch.mean(clean_output[step]["probs"])
        cumulative *= step_prob
        print(f"{step.capitalize()}: {step_prob}")
    print(f"Cumulative confidence: {cumulative}")

    variational_output = var_pipe.variational_inference(x=input_speech['array'],n_runs=2)


    for step in var_pipe.pipeline_map.keys():
        print(f'\n{step}:')
        step_output = variational_output['variational'][step]
        for run in step_output:
            print(run['semantic_embedding'])

if __name__ == "__main__":
    TTS_pars = {
        "transcriber": {
            "specific_task": "automatic-speech-recognition",
            "model": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        },
        "translator": {
            "specific_task": "translation_fr_to_en",
            "model": "ybanas/autotrain-fr-en-translate-51410121895",
        },
        "summariser": {
            "specific_task": "summarization",
            "model": "marianna13/flan-t5-base-summarization",
        },
    }
    main(TTS_params=TTS_pars)
