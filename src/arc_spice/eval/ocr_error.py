from typing import Any

from torchmetrics.text import CharErrorRate


def ocr_error(ocr_output: dict[str, Any]) -> float:
    """
    Compute the character error rate for the predicted ocr characters.

    NB: - This puts the strings into lower case for comparison.
        - and ideal error rate is 0, worst case is 1.

    Args:
        ocr_output: output from the ocr model, with structure,
                    {
                        'full_output': [
                            {
                                'generated_text': generated text form ocr (str),
                                'targets': target text (str)
                            }
                        ],
                        'output': pieced backtogeter full string (str)
                    }

    Returns:
        character error rate across entire output of OCR(float)
    """
    preds = [itm["generated_text"].lower() for itm in ocr_output["full_output"]]
    targs = [itm["target"].lower() for itm in ocr_output["full_output"]]
    cer = CharErrorRate()
    return cer(preds, targs).item()
