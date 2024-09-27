"""
    An example use of the transcription, translation and summarisation pipeline.
"""

import numpy as np
from datasets import Audio, load_dataset

from arc_spice.pipelines.TTS_pipeline import TTSpipeline


def main(TTS_params):
    """main function"""
    TTS = TTSpipeline(TTS_params)
    TTS.print_pipeline()
    ds = load_dataset(
        "facebook/multilingual_librispeech", "french", split="test", streaming=True
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    input_speech = next(iter(ds))["audio"]
    # arrays = []
    # n = 5
    # for idx, data in enumerate(iter(ds)):
    #     arrays.append(data["audio"]["array"])
    #     if idx == n:
    #         break
    # arrays = np.concatenate(arrays)
    TTS.run_pipeline(input_speech["array"])
    TTS.print_results()


if __name__ == "__main__":
    TTS_pars = {
        "transcriber": {
            "specific_task": "automatic-speech-recognition",
            "model": "openai/whisper-small",
        },
        "translator": {
            "specific_task": "translation_fr_to_en",
            "model": "facebook/mbart-large-50-many-to-many-mmt",
        },
        "summariser": {
            "specific_task": "summarization",
            "model": "facebook/bart-large-cnn",
        },
    }
    main(TTS_params=TTS_pars)
