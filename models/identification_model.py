import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def identify_objects(segmented_objects):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #a list of common object classes
    object_classes = ["person", "car", "chair", "dog", "cat", "table", "plant", "bottle", "book", "phone", "missile", "plate", "cycle", "airplane", "bicycle", "bird", "boat", "bus", "truck", "motorcycle", "train", "laptop", "tv", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    identified_objects = {}
    for obj in segmented_objects:
        #ensure the image is in egb mode
        obj_image = obj['image'].convert('RGB')

        #pprocess the image
        inputs = processor(images=obj_image, return_tensors="pt", padding=True)

        #get image features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        #process text label
        text_inputs = processor(text=object_classes, padding=True, return_tensors="pt")

        #get text feature
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)

        #cllculate smilarity scores
        similarity_scores = (image_features @ text_features.T).squeeze(0)
        probs = torch.nn.functional.softmax(similarity_scores, dim=0)

        #got top 1 prediction
        top5_probs, top5_indices = probs.topk(1)

        identified_objects[obj['id']] = {
            'top_predictions': [
                {'label': object_classes[idx], 'probability': prob.item()}
                for idx, prob in zip(top5_indices, top5_probs)
            ]
        }

    return identified_objects
