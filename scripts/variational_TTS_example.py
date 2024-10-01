"""
    An example use of the transcription, translation and summarisation pipeline.
"""

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
    print(clean_output["transcription"]["logits"].shape)
    print(clean_output["translation"]["logits"].shape)
    print(clean_output["summarisation"]["logits"].shape)
    # entropy
    print(torch.mean(clean_output["transcription"]["entropy"]))
    print(torch.mean(clean_output["translation"]["entropy"]))
    print(torch.mean(clean_output["summarisation"]["entropy"]))
    # probability
    print(torch.mean(clean_output["transcription"]["probs"]))
    print(torch.mean(clean_output["translation"]["probs"]))
    print(torch.mean(clean_output["summarisation"]["probs"]))


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
