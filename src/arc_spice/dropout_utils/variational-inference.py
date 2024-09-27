from transformers import pipeline

from src.arc_spice.dropout_utils.dropout_pipeline import set_dropout


class TTSVariationalPipeline:
    """
    variational version of the TTSpipeline
    """

    def __init__(self, pars: dict[str : dict[str:str]]):
        self.transcriber = pipeline(
            task=pars["transcriber"]["specific_task"],
            model=pars["transcriber"]["model"],
        )
        self.translator = pipeline(
            task=pars["translator"]["specific_task"], model=pars["translator"]["model"]
        )
        self.summariser = pipeline(
            task=pars["summariser"]["specific_task"], model=pars["summariser"]["model"]
        )

        self.pipeline_map = {
            "transcription": self.transcriber,
            "translation": self.translator,
            "summarisation": self.summariser,
        }

    def clean_inference(self, x):
        output = {}
        """Run the pipeline on an input x"""
        transcription = self.transcriber(x)
        output["transcription"] = transcription["text"]
        translation = self.translator(transcription["text"])
        output["translation"] = translation[0]["translation_text"]
        summarisation = self.summariser(translation[0]["translation_text"])
        output["summarisation"] = summarisation[0]["summary_text"]
        return output

    def variational_inference(self, x, n_runs=5):
        output = {"clean": {}, "variational": {}}
        output["clean"] = self.clean_inference(x)
        input_map = {
            "transcription": x,
            "translation": output["clean"]["transcription"],
            "summarisation": output["clean"]["translation"],
        }
        for model_key, pipeline in self.pipeline_map.values():
            # perhaps we could use a context handler here?
            pipeline.model = set_dropout(pipeline.model, True)
            output["variational"][model_key] = [None] * n_runs
            for run_idx in range(n_runs):
                output["variational"][model_key][run_idx] = pipeline(
                    input_map[model_key]
                )
            pipeline.model = set_dropout(pipeline.model, False)

        return output


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
