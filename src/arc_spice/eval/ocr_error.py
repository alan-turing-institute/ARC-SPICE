"""
OCR error computation for eval.
"""

from typing import Any

from torchmetrics.text import CharErrorRate


def ocr_error(ocr_output: dict[Any, Any]) -> float:
    """
    Compute the character error rate for the predicted ocr character.

    NB: - this puts all strings into lower case for comparisons.
        - ideal error rate is 0, worst case is 1.

    Args:
        ocr_output: output from the ocr model, with structure,
                    {
                        'full_output: [
                            {
                                'generated_text': gen text from the ocr model (str)
                                'target': target text (str)
                                'entropies': entropies for UQ (torch.Tensor)
                            }
                        ]
                        'outpu': pieced back together full string (str)
                    }

    Returns:
        Character error rate across entire output of OCR (float)
    """
    preds = [itm["generated_text"].lower() for itm in ocr_output["full_output"]]
    targs = [itm["target"].lower() for itm in ocr_output["full_output"]]
    cer = CharErrorRate()
    return cer(preds, targs).item()
