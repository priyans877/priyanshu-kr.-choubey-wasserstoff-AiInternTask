import easyocr
import numpy as np

def extract_text(segmented_objects):
    reader = easyocr.Reader(['en'])

    extracted_text = {}
    for obj in segmented_objects:
        result = reader.readtext(np.array(obj['image']))
        extracted_text[obj['id']] = [text for _, text, _ in result]

    return extracted_text