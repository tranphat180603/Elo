from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import argparse
import json
from tqdm import tqdm
from PIL import Image
import shutil
import os
import numpy as np
import base64
from io import BytesIO
import json
import random 

import torch
from safetensors.torch import load_file
from ultralytics.nn.tasks import DetectionModel
from transformers import AutoModel, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download, login

from OmniParser.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

login("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")

def load_and_save_model():    
    download_patterns = ["*.json", "*.bin", "*.safetensors", "*.yaml"]
    #Load the subdirectories of Omni Parser into weights
    snapshot_download(
        repo_id="microsoft/OmniParser",
        local_dir ="OmniParser/weights",
        allow_patterns = download_patterns,
    )
    if not os.path.isfile("OmniParser/weights/icon_detect/best.pt"):
        tensor_dict = load_file("OmniParser/weights/icon_detect/model.safetensors")
        model = DetectionModel('OmniParser/weights/icon_detect/model.yaml')
        model.load_state_dict(tensor_dict)
        torch.save({'model':model}, 'OmniParser/weights/icon_detect/best.pt')
        print("Converted safetensors to pt successfully!")
def sanitize_filename(filename):
    return filename.replace("/", "_").replace("\\", "_").replace(" ", "_")

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    else:
        return obj


@dataclass
class PipelineConfig:
    vlm_model_name: str = "openbmb/MiniCPM-V-2_6"
    yolo_model_path: str = "OmniParser/weights/icon_detect/best.pt"
    caption_model_name: str = "blip2"
    caption_model_path: str = "OmniParser/weights/icon_caption_blip2"
    output_file: str = "data.json"
    logging_file: str = "box_prop.json"
    start_index: int = 0
    end_index: int = None

def get_enhanced_meta_prompt(level: str, parsed_content_list: List[str]) -> str:
    context_description = (
        f"You are given an image with elements that have been bounded by boxes to increase precision. Here is the list of the bounding boxes on this screen and their corresponding elements:\n"
        f"{parsed_content_list}\n\n"
    )
    
    enhanced_meta_prompts = {
        "conversation": (
            context_description +
            "Imagine you are a helpful agent answering questions from a user who is exploring this website. Your role is to help the user understand the website's contents and functionalities in a conversational and supportive manner. Based on this information, create three user questions and provide appropriate responses in a conversational style.\n\n"
            "Your job is to create 2 high-quality pairs of question response to simulate this situation. \n"
            "Please respond ONLY in the following valid JSON format, without any additional commentary or formatting:\n"
            "[\n"
            "  {\n"
            '    "question": "User question here",\n'
            '    "response": "Assistant response here"\n'
            "  },\n"
            "  {\n"
            '    "question": "Another user question",\n'
            '    "response": "Another assistant response"\n'
            "  }\n"
            "]\n\n"
            "Ensure that there is no text outside of the JSON structure, as this will be parsed directly. Follow these instructions strictly to avoid parsing errors."
            "\n\n"
            "- For the first question, make it brief and focused on some elements, such as asking about recent updates, a specific section, or basic site functionality. For example, 'What's new on the homepage?' or 'How do I check my notifications?'\n"
            "- For the second question, make it the most detailed, covering questions about how different site sections interact or summarizing multiple updates. For example, 'Can you summarize today's updates from the news, notifications, and recommendations?' Avoid mentioning specific technical terms like 'box IDs' or 'coordinates.'"
        ),
        "description": (
            context_description +
            "Imagine you are an agent helping a user get an overview of the website's layout and its main sections. The user is curious about how the website is organized, and your job is to provide increasingly detailed descriptions in response to their questions.\n\n"
            "Your job is to create 2 high-quality pairs of question response to simulate this situation.\n"
            "Respond ONLY in the following valid JSON array format, without additional commentary:\n"
            "[\n"
            "  {\n"
            '    "question": "User question here",\n'
            '    "response": "Assistant response here"\n'
            "  },\n"
            "  {\n"
            '    "question": "Another user question",\n'
            '    "response": "Another assistant response"\n'
            "  }\n"
            "]\n\n"
            "- The first response should cover multiple sections or a broader area of the site. For example: Tell me the name of some of my online friends on the right side of the page.'\n"
            "- The second response should offer a comprehensive overview of the visible sections, such as 'What are the main parts of this page, including the sidebar, main content area, and footer?' or 'Can you summarize the features available across the dashboard?' Avoid mentioning specific technical terms like 'box IDs' or 'coordinates.'"
        ),
        "complex_tasks": (
            context_description +
            "Imagine you are guiding a user through interactions on a form-based webpage. The page you are helping with contains static elements such as text boxes, radio buttons, drop-down lists, and other similar input controls. Examples include hotel booking forms, travel site filters, registration forms, or product filters. The tasks do not involve any page transitions or loading of new elements—everything is visible from the start.\n\n"
            "You must create three user questions that involve progressively complex tasks. These tasks should only use the elements that have bounding boxes attached to them, such as text boxes, buttons, radio buttons, or any input elements clearly indicated in the provided image. Focus solely on actions within this single page, and do not reference interactions that involve navigating away from the page or adding new content beyond what is visible.\n\n"
            "Please respond ONLY in the following valid JSON format, without any additional text or commentary:\n"
            "[\n"
            "  {\n"
            '    "question": "User question here",\n'
            '    "response": "Step 1: Do this (Text Box ID 0: Gmail). Step 2: Then do this (Text Box ID 1: Google)."\n'
            "  },\n"
            "  {\n"
            '    "question": "Another user question",\n'
            '    "response": "Step 1: First step (Text Box ID 2: Log in). Step 2: Second step (Text Box ID 3: Log out)."\n'
            "  }\n"
            "]\n\n"
            
            "Guidelines for generating responses:\n"
            "- First question: Create a simple task that involves interacting with 1-2 elements, such as entering information into one or two text boxes or selecting a radio button.\n"
            "- Second question: Create a task that involves 2-3 steps, such as filling in multiple text boxes or making a combination of text input and selection, like a drop-down list and a radio button.\n"
            "- Third question: Create a complex task that involves interacting with 3-4 different elements on the form, such as filling out multiple text boxes, selecting multiple options, or adjusting settings via different input elements.\n\n"
            
            "Important:\n"
            "- Ensure each action only involves elements with bounding boxes—do not include any interaction with elements that are not clearly indicated with bounding boxes.\n"
            "- Include box IDs in parentheses after mentioning any element to clearly identify which part of the interface to interact with (e.g., 'Select Gmail ('Text Box ID 0: Gmail')').\n"
        )
    }


    return enhanced_meta_prompts[level]

class ImageProcessor:
    def __init__(self, som_model, caption_model_processor):
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor

    def process_image(self, image: Image) -> Tuple[Image.Image, List[str]]:
        image = image.convert("RGB")
        image_arr = np.array(image)
        ocr_bbox_rslt, _ = self._perform_ocr(image_arr)
        text, ocr_bbox = ocr_bbox_rslt[0], ocr_bbox_rslt[1]
        
        return self._get_labeled_image(image, ocr_bbox, text)

    def _perform_ocr(self, image: Image) -> Tuple[Any, Any]:
        return check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )

    def _get_labeled_image(self, image: Image, ocr_bbox: Any, text: str) -> Tuple[Image.Image, List[str]]:
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        labeled_img, coords, content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_TRESHOLD=0.03,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.1,
            imgsz=640
        )
        return labeled_img, coords, content_list

class SyntheticDataGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vlm_model, self.processor = self._initialize_vlm_model()
        self.image_processor = ImageProcessor(
            self._initialize_som_model(),
            self._initialize_caption_processor()
        )

    def _initialize_vlm_model(self):
        model = MllamaForConditionalGeneration.from_pretrained(
            self.config.vlm_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.config.vlm_model_name)
        return model, processor
    
    def _initialize_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            self.config.vlm_model_name,
            trust_remote_code=True
        )
    
    def _initialize_som_model(self):
        return get_yolo_model(model_path=self.config.yolo_model_path).to('cuda')

    def _initialize_caption_processor(self):
        return get_caption_model_processor(
            model_name=self.config.caption_model_name,
            model_name_or_path=self.config.caption_model_path,
            device='cuda'
        )

    def _generate_conversation_data(self, dataset: List[Dict]) -> List[Dict]:
        conversation_data = []
        num_retry = 5
        for item in tqdm(dataset, desc = "Generating data"):
            labeled_img, _, content_list = self.image_processor.process_image(item["image"])
            for level in ["conversation", "description", "complex_tasks"]:
                meta_prompt = get_enhanced_meta_prompt(level, content_list)
                image = Image.open(BytesIO(base64.b64decode(labeled_img)))
                # Format messages for Llama
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": meta_prompt}
                    ]}
                ]
                
                response_data = None
                for n in range(num_retry):
                    seed_value = random.randint(0, 10000)
                    random.seed(seed_value)
                    
                    try:
                        self.vlm_model.tie_weights()
                        # Process input using Llama format
                        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                        inputs = self.processor(
                            image,
                            input_text,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).to(self.vlm_model.device)
                        # Generate response
                        output = self.vlm_model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=0.7,
                            top_p=0.9
                        )
                        output = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
                        ]
                        response = self.processor.decode(output[0], skip_special_tokens=True)                        
                        print(f"Raw Response (Attempt {n+1}, Seed {seed_value}): {response}", flush=True)

                        if isinstance(response, str):
                            try:
                                response_data = json.loads(response)
                                break
                            except json.JSONDecodeError:
                                print(f"JSON decode error on attempt {n+1}, retrying...")
                        elif isinstance(response, (list, dict)):
                            response_data = response
                            break

                    except Exception as e:
                        print(f"Error during generation (Attempt {n+1}): {str(e)}")
                        continue

                if response_data is None:
                    print(f"Failed to parse response after {num_retry} attempts.", flush=True)
                    continue

                print(f"Processed Response: {response_data}", flush=True)

                for item in response_data:
                    conversation_data.append({
                        "user": item["question"],
                        "assistant": item["response"],
                    })

        return conversation_data


class Dataset:
    def __init__(self):
        # Load dataset and apply default filter
        self.ds = load_dataset("agentsea/wave-ui-25k", cache_dir="")
        self.filtered_ds = self._apply_default_filter()

    def _apply_default_filter(self):
        # Apply filter to keep only web platform examples in English
        return self.ds["train"].filter(lambda example: example["platform"] == "web" and example["language"] == "English")

    def select_data(self, start_index, end_index=None):
        if end_index:
            return self.filtered_ds.select(range(start_index, end_index))
        else:
            return self.filtered_ds.select(range(start_index, len(self.filtered_ds)))

    def process_data_in_batches(self, start_index, end_index=None, batch_size=100):
        total_size = len(self.filtered_ds)
        end_index = end_index if end_index else total_size
        for i in range(start_index, end_index, batch_size):
            yield self.filtered_ds.select(range(i, min(i + batch_size, end_index)))

class StreamingJSONWriter:
    def __init__(self, filename):
        self.filename = filename
        self.is_first = True
        
        # Initialize the JSON file with an opening bracket
        with open(self.filename, 'w') as f:
            f.write('[\n')
    
    def write_entry(self, entry):
        with open(self.filename, 'a') as f:
            if not self.is_first:
                f.write(',\n')
            json.dump(entry, f)
            self.is_first = False
    
    def close(self):
        with open(self.filename, 'a') as f:
            f.write('\n]')

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("--vlm_model_name", type=str, default="meta-llama/Llama-3.2-90B-Vision-Instruct")
    parser.add_argument("--yolo_model_path", type=str, default="OmniParser/weights/icon_detect/best.pt")
    parser.add_argument("--caption_model_name", type=str, default="blip2")
    parser.add_argument("--caption_model_path", type=str, default="OmniParser/weights/icon_caption_blip2")
    parser.add_argument("--output_file", type=str, default="data.json")
    parser.add_argument("--logging_file", type=str, default="box_prop1.json")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for dataset split")
    parser.add_argument("--end_index", type=int, help="Ending index for dataset split")
    
    args = parser.parse_args()
    config = PipelineConfig(**vars(args))
    
    # load models
    load_and_save_model()

    output_dir = "processed_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    som_model = get_yolo_model(model_path='OmniParser/weights/icon_detect/best.pt')
    caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="OmniParser/weights/icon_caption_blip2")
    image_processor = ImageProcessor(som_model, caption_model_processor)
    
    batch_size = 20  
    box_prop_name = config.logging_file
    print(f"Logging in file {box_prop_name}")
    
    dataset_instance = Dataset()
    json_writer = None
    
    try:
        full_ds = dataset_instance.select_data(config.start_index, config.end_index)    
        json_writer = StreamingJSONWriter(config.logging_file)
        progress_bar = tqdm(total = len(full_ds), desc= "Full Dataset Progress")
        for batch in dataset_instance.process_data_in_batches(config.start_index, config.end_index, batch_size):
            for idx, sample in enumerate(tqdm(batch, desc="Processing Batch Images")):
                if sample["name"] == "My eBay link":
                    print("Skipping potential damaged image")
                    continue
                try:
                    # Process image
                    with torch.no_grad():
                        processed_image, coordinates, bounding_boxes = image_processor.process_image(sample["image"])
                    print(f"Processing image: {sample['name']}")

                    # Sanitize the image name
                    image_name = f"processed_image_{sanitize_filename(sample['name'])}.png"

                    # Create and write JSON entry
                    box_list = {
                        "image_name": image_name,
                        "boxes_content": convert_ndarray_to_list(bounding_boxes),
                        "coord": convert_ndarray_to_list(coordinates)
                    }
                    json_writer.write_entry(box_list)
                    
                    #try to save machine's RAM
                    del bounding_boxes
                    del coordinates

                    # Save image
                    normal_image_path = os.path.join(output_dir, image_name)
                    if isinstance(processed_image, Image.Image):
                        processed_image.save(normal_image_path)
                    else:
                        normal_image = Image.open(BytesIO(base64.b64decode(processed_image)))
                        normal_image.save(normal_image_path)
                    print("Image saved")

                except Exception as e:
                    print(f"Error processing sample {sample['name']}: {e}")
                    continue
            progress_bar.update(batch_size)

            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    
    finally:
        if json_writer:
            json_writer.close()
            print(f"Results saved to {config.logging_file}")
        
    print(f"All processed images saved in '{output_dir}' directory.")

    # generator = SyntheticDataGenerator(config)
    # formatted_data = generator._generate_conversation_data(dataset)
    
    # with open(config.output_file, "w") as f:
    #     json.dump(formatted_data, f, indent=4)
    # print(f"Data saved to {config.output_file}")

if __name__ == "__main__":
    main()
