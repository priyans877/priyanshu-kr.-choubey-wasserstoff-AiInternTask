import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io

def visualize_output(image, segmented_objects, identified_objects, extracted_text, summarized_attributes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Original image with bounding boxes
    draw = ImageDraw.Draw(image)
    for obj in segmented_objects['objects']:
        box = obj['box']
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), obj['id'], fill="red")

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("Segmented Image")

    #table with object data
    table_data = []
    for obj in segmented_objects['objects']:
        obj_id = obj['id']
        identification = identified_objects.get(obj_id, {}).get('top_predictions', [{}])[0].get('label', 'Unknown')
        text = ', '.join(extracted_text.get(obj_id, []))
        summary = summarized_attributes.get(obj_id, '')
        table_data.append([obj_id, identification, text, summary])

    ax2.axis('off')
    ax2.table(cellText=table_data, 
              colLabels=['Object ID', 'Identification', 'Extracted Text', 'Summary'],
              cellLoc='center', loc='center')
    ax2.set_title("Object Data")

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)
