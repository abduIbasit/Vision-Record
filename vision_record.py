import torch
import numpy as np
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline
from ultralytics import YOLO
from datetime import datetime


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

yolo_model = YOLO('yolov8x.pt')

def extract_activity_color(text):
    """Extract activity and color information from the pipeline output text."""
    text = text.split('ASSISTANT:')[-1].strip()
    try:
        start = text.find('{')
        end = text.find('}') + 1
        
        if start != -1 and end != -1:
            # Extract substring that represents the dictionary
            dict_str = text[start:end]
            activity_color_dict = eval(dict_str)
            
            # Check for 'object activity' and 'object color' in the extracted values
            activity = activity_color_dict.get('activity')
            color = activity_color_dict.get('color')
            
            if activity in ['object activity'] or color in ['object color']:
                return None, None

            return activity, color
        else:
            return None, None
            
    except Exception as e:
        # print(f"Error parsing activity and color: {e}")
        return None, None


    
def generate_vision_record(image: np.array) -> dict:
    """
    Generates a vision record from the input image.

    Parameters:
        image (np.array): The input image as a NumPy array.

    Returns:
        dict: A dictionary containing the vision record.
    """
    
    # Convert image to PIL format
    pil_image = Image.fromarray(image)

    # YOLOv8 model to detect objects in image
    detection_result = yolo_model(pil_image)[0]
    detected_objects = []
    bounding_boxes = []
    object_activities = []
    object_colors = []

    # Extract detected objects, bounding boxes, activities, and colors
    for box in detection_result.boxes:
        class_id = int(box.cls.item())  # Get the class ID of the detected object
        object_name = detection_result.names[class_id]  # Get the object name using the class ID
        bounding_box = [round(coord.item()) for coord in box.xyxy[0]]  # Get the bounding box coordinates rounded to zero decimal

        detected_objects.append(object_name)
        bounding_boxes.append(bounding_box)

        # Crop the detected object
        x1, y1, x2, y2 = map(int, bounding_box)
        cropped_image = pil_image.crop((x1, y1, x2, y2))

        # Prompt for getting object activity and color from llava
        prompt_template = "State the accurate object activity and object color formatted as a key value pair like {'activity':'object activity','color':'object color'}, return None as value for non-applicable cases"
        prompt = f"USER: <image>\n{prompt_template}\nASSISTANT:"
        
        output = pipe(cropped_image, prompt=prompt, generate_kwargs={"max_new_tokens": 200, "do_sample": True, "temperature": 0.3})
        activity, color = extract_activity_color(output[0]["generated_text"])
        
        object_activities.append(activity)
        object_colors.append(color)

    # Prompt for getting frame summary from llava
    prompt = "USER: <image>\nConcisely and accurately explain the image frame in few words\nASSISTANT:"
    frame_summary_output = pipe(pil_image, prompt=prompt, generate_kwargs={"max_new_tokens": 150, "do_sample": True, "temperature": 0.3})
        
    # Extract frame summary from output
    frame_summary = frame_summary_output[0]["generated_text"].split('ASSISTANT:')[-1].strip()
    
    # Get the timestamp of when image is processed
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # Get the frame size
    frame_size = image.shape[:2]  # (height, width)

    # Save detected image
    result_image = detection_result.plot()
    output_image = Image.fromarray(result_image[..., ::-1])
    output_image.save("detected_image.jpg")

    # Output dictionary
    vision_record = {
        "Time": timestamp,
        "Objects": detected_objects,
        "Objects Activities": object_activities,
        "Object Colors": object_colors,
        "Object Bounding Boxes": bounding_boxes,
        "Frame Size": frame_size,
        "Frame Summary": frame_summary
    }

    return vision_record


# Example usage:
# image_path = "C:\Users\USER\Desktop\images\demo.jpg" # Replace image path with any image of your choice
# vision_record = generate_vision_record(np.array(Image.open(image_path)))
# print(vision_record)