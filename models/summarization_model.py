from transformers import pipeline

def summarize_attributes(identified_objects, extracted_text):
    summarizer = pipeline("summarization")

    summarized_attributes = {}
    for obj_id in identified_objects:
        text_to_summarize = (
            f"Object: {identified_objects[obj_id]['top_predictions'][0]['label']}. "
            f"Extracted text: {' '.join(extracted_text.get(obj_id, []))}"
        )
        summary = summarizer(text_to_summarize, max_length=50, min_length=10, do_sample=False)
        summarized_attributes[obj_id] = summary[0]['summary_text']

    return summarized_attributes