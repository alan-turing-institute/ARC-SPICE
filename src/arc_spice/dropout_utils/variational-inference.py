from src.arc_spice.dropout_utils.dropout_pipeline import MCDropoutPipeline


class VariationalPipeline:

    def __init__(self, pars: dict[str : dict[str:str]]):
        self.transcriber = MCDropoutPipeline(
            task=pars["transcriber"]["specific_task"],
            model=pars["transcriber"]["model"],
        )
        self.translator = MCDropoutPipeline(
            task=pars["translator"]["specific_task"], model=pars["translator"]["model"]
        )
        self.summariser = MCDropoutPipeline(
            task=pars["summariser"]["specific_task"], model=pars["summariser"]["model"]
        )


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