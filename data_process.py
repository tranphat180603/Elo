#load image dataset from HF
#open and concat JSON into 1
#new JSON file with samples that exist with image
#return Filtered images based on image_name = image_name

##Introduce element tokens as ground truth
## simple split
#1 conversation pair
#1 description pair
#1 simple tasks move, left click, right click

##complex split
#1 multi-step question-answer pair

##bespoke split (learn how to crawl data like I want.)
##TODO: stage 2, multi-site sequence of actions
import os
import json
from typing import List, Dict, Any, Tuple
from PIL import Image
import requests
import numpy as np

import torch
from torchvision import io
from datasets import load_dataset
from transformers import MllamaForConditionalGeneration, AutoProcessor

from data_generation import StreamingJSONWriter, convert_ndarray_to_list

class Processor():
    def __init__(self, vlm_model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                 processed_file_path: str = "/Elo/processed_box_prop.json") -> None:
        self.vlm_model = MllamaForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model_processor = AutoProcessor.from_pretrained(vlm_model)
        self.streamingwriter = StreamingJSONWriter(processed_file_path)

    def infer_vlmodel(self, image: Image.Image, box_prop: List[str]) -> List[str]:
        """
        Infer the model and validate the response structure for box properties.
        Retries up to 3 times if the length of box_prop mismatches.
        """
        max_retries = 3
        retries = 0
        while retries < max_retries:
            meta_prompt = f"""
            I will provide you with an image along with its bounding box properties obtained from another module. 
            You will not see the bounding boxes on the image I send you, but you will have access to their text descriptions.

            Your task:
            1. Review the `box_property` values provided below.
            2. Identify and correct any corrupted, misspelled, or grammatically incorrect values in the descriptions.
            3. Rewrite only the values that need fixing, ensuring the format, **order**, and **structure** of the `box_property` remain unchanged.
            4. The output **must**:
               - Retain the exact number of entries as the original `box_property`.
               - Return a valid Python list of strings, formatted like the input.

            Additional Notes:
            - If any text is unclear or corrupted beyond recognition, use your knowledge of languages (English, Vietnamese, Chinese) and the context of websites to infer the correct text.
            - If the number of output entries does not match the input, you will be re-prompted.

            Here is the `box_property` (length: {len(box_prop)}):
            {str(box_prop)}
            """

            image = Image.open(image)
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{meta_prompt}"}
                ]}
            ]
            input_text = self.model_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.model_processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.vlm_model.device)

            output = self.vlm_model.generate(**inputs, max_new_tokens=4096)
            response = self.model_processor.decode(output[0])

            # Convert response to list of strings
            try:
                response_list = eval(response) 
                if isinstance(response_list, (list, ndarray)) and len(response_list) == len(box_prop):
                    return response_list
            except Exception as e:
                print(f"Error parsing response: {e}")

            print(f"Retry {retries + 1}/{max_retries}: Mismatched structure or length.")
            retries += 1

        raise ValueError(f"Failed to generate valid box properties after {max_retries} retries.")

    def process(self, img_name, box_prop, coordinates):
        """Process all box properties and images."""
        if os.path.exists(img_name):
            image = Image.open(img_name)
            try:
                response = self.infer_vlmodel(image, box_prop)
                print(f"Generated response {response}")
                box_data = {
                    "image_name": img_name,
                    "boxes_content": response,
                    "coord": coordinates
                }
                self.streamingwriter.write_entry(box_data)
            except ValueError as e:
                print(f"Skipping {img_name} due to invalid response: {e}")

        print("Finished processing data")
        self.streamingwriter.close()
        return img_name, response
