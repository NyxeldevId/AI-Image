import os
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

import modules.config
import modules.html
import modules.flags as flags
import modules.meta_parser

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def trigger_describe(mode, img_path):
    print("Running")
    print("Press Ctrl+C for Stop ")
    if mode == flags.desc_type_photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        if img_path.startswith('http'):
            img = download_image(img_path)
        else:
            img = Image.open(img_path).convert("RGB")
        return default_interrogator_photo(img), ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
    elif mode == flags.desc_type_anime:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        if img_path.startswith('http'):
            img = download_image(img_path)
        elif isinstance(img_path, str):
            # Load the image if the input is a path
            img = Image.open(img_path).convert("RGB")
        elif isinstance(img_path, np.ndarray):
            # Use the provided NumPy array directly
            img = Image.fromarray(img_path).convert("RGB")
        else:
            raise ValueError("Invalid image format. Please provide a valid path or NumPy array.")
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        
        return default_interrogator_anime(img_array), ["Fooocus V2", "Fooocus Masterpiece"]
    return mode, ["Fooocus V2"]

style_selections = modules.config.default_styles

def run_describe(image_path, content_type):
    desc_input_image = image_path
    desc_method = content_type

    result, style_selections = None, None

    if desc_method in ["Photograph", "1", ""]:
        desc_method = "Photograph (1)"
        result, style_selections = trigger_describe(flags.desc_type_photo, desc_input_image)
    elif desc_method in ["Art/Anime", "2"]:
        desc_method = "Art/Anime (2)"
        result, style_selections = trigger_describe(flags.desc_type_anime, desc_input_image)
    else:
        print("ERROR!")

    if result or style_selections != "":
        style_selections = ""
        print("Result:", result)
        # print("Style Selections:", style_selections)
        quit()

if __name__ == "__main__":
    desc_input_image = input("Path to Image (local path or URL): ")

    if desc_input_image == "":
        desc_input_image = "./imgs/Gambar1.jpg"

    print(f"You use: {desc_input_image}")

    desc_method = input(
        """
    Select Content Type: 
    Photograph (1)
    Art/Anime (2)
    """
    )

    run_describe(desc_input_image, desc_method)
