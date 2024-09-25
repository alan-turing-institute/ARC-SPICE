"""
    Class for the transcription, translation and summarisation pipeline.
"""

from transformers import pipeline


class TTSpipeline:
    """
    Class for the transcription, translation, summarisation pipeline.

    pars:
        - {'top_level_task': {'specific_task': str, 'model_name': str}}
    """

    def __init__(self, pars) -> None:
        self.pars = pars
        self.transcriber = pipeline(
            pars["transcriber"]["specific_task"], pars["transcriber"]["model"]
        )
        self.translator = pipeline(
            pars["translator"]["specific_task"], pars["translator"]["model"]
        )
        self.summariser = pipeline(
            pars["summariser"]["specific_task"], pars["summariser"]["model"]
        )
        self.results = {}

    def print_pipeline(self):
        """Print the models in the pipeline"""
        print(f"Transcriber model: {self.pars['transcriber']['model']}")
        print(f"Translator model: {self.pars['translator']['model']}")
        print(f"Summariser model: {self.pars['summariser']['model']}")

    def run_pipeline(self, x):
        """Run the pipeline on an input x"""
        transcription = self.transcriber(x)
        self.results["transcription"] = transcription["text"]
        translation = self.translator(transcription["text"])
        self.results["translation"] = translation[0]["translation_text"]
        summarisation = self.summariser(translation[0]["translation_text"])
        self.results["summarisation"] = summarisation[0]["summary_text"]

    def print_results(self):
        """Print the results for quick scanning"""
        for key, val in self.results.items():
            print(f"{key} result is: \n {val}")
