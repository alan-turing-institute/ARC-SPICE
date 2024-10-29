from transformers import pipeline

from arc_spice.data.multieurlex_dataloader import load_multieurlex

model_pars = {
    "translator": {
        "specific_task": "translation_fr_to_en",
        "model": "ybanas/autotrain-fr-en-translate-51410121895",
    }
}

lang_pair = {"source": "fr", "target": "en"}
[train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)

row_iterator = iter(train)
test_row = next(row_iterator)

translator = pipeline(
    model_pars["translator"]["specific_task"],
    model_pars["translator"]["model"],
    max_length=512,
    device="mps",
)

translated = translator(test_row["source_text"])

print(translated)
