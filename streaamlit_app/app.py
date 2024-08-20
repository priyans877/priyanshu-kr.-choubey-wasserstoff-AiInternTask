import streamlit as st
from PIL import Image
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.segmentation_model import segment_image
from models.identification_model import identify_objects
from models.text_extraction_model import extract_text
from models.summarization_model import summarize_attributes
from utils.visualization import visualize_output

def main():
    st.title("Image Segmentation and Object Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            with st.spinner("Processing..."):
                #Applying image segmentation
                segmented_objects = segment_image(image)
                st.subheader("Segmented Objects")
                st.image(segmented_objects['visualization'], use_column_width=True)

                #applying object extraction
                for idx, obj in enumerate(segmented_objects['objects']):
                    st.image(obj['image'], caption=f"Object {idx}", width=200)

                #applying object identity
                identified_objects = identify_objects(segmented_objects['objects'])
                #st.subheader(identified_objects)
                st.subheader("Identified Objects")
                for obj_id, obj_data in identified_objects.items():
                    #st.subheader(obj_data)
                    st.write(f"Object {obj_id}:")
                    for pred in obj_data['top_predictions']:
                        st.write(f"  - {pred['label']}: {pred['probability']:.2f}")

                #extracting text
                extracted_text = extract_text(segmented_objects['objects'])
                st.subheader("Extracted Text")
                st.write(extracted_text)

                #summary using hugging face
                summarized_attributes = summarize_attributes(identified_objects, extracted_text)
                st.subheader("Summarized Attributes")
                st.write(summarized_attributes)

                #output generation
                final_output = visualize_output(image, segmented_objects, identified_objects, extracted_text, summarized_attributes)
                st.subheader("Final Output")
                st.image(final_output, use_column_width=True)

if __name__ == "__main__":
    main()
