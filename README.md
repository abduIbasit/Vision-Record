# Instructions for Running the Vision Record Code

## Set Up the Environment

Ensure you have Python 3.8 or higher installed.

Create a virtual environment (recommended):

```bash
python -m venv vision
source vision/bin/activate
```

## GPU Requirements

The LLAVA model utilised in the code requires GPU drive for efficient processing. To run the code with GPU support, ensure you have a compatible GPU setup. If you do not have access to a GPU, you can leverage Kaggle or Google Colab free GPU resources. Simply copy and paste the entire code block into a notebook cell on one of these platforms and run. Make sure to connect to a GPU supported runtime.


## Install Dependencies:

Run the below command from the directory where the requirements.txt attached to this folder is

```bash
pip install -r requirements.txt
```

## Run the Code

Execute the script using:

```bash
python vision_record.py
```

Note: Replace the image_path variable in the example usage section at the bottom of the script with the path to your image file.


## Review Results:

The script will generate and save an image with bounding boxes around detected objects as **detected_image.jpg**.
The vision record dictionary will be available as the return value of the generate_vision_record function and printed on the terminal.