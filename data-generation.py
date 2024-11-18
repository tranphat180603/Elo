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
        f"You are given a website screenshot with elements that have been bounded by boxes to increase precision. "
        f"Here is the list of the bounding boxes on this screen and their corresponding elements:\n"
        f"{parsed_content_list}\n\n"
    )
    
    enhanced_meta_prompts =  {
    "conversation": (
        context_description +
        "Imagine you are a helpful agent operating automatically on the website and know everything about it. "
        "At the same time, you are also a user looking at a website and want to ask the agent and receive helpful answers from it. "
        "You must play both roles in this simulation and formulate good conversations between these two entities. \n\n"
        "Context: The user is currently looking at the website. They are approaching the website and trying to know everything possible about it: "
        "main purpose, the content, advertisements, functionalities—everything possible. The agent's role is to answer the user's questions in a helpful and informative way.\n\n"
        "Important: Ensure that the conversation uses diverse and natural vocabularies to avoid sounding overly robotic or fixed in tone.\n\n"
        "Based on this information, create 5 diverse, meaningful sets of conversations between these two entities. \n"
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
        "Ensure that there is no text outside of the JSON structure, as this will be parsed directly. Follow these instructions strictly to avoid parsing errors.\n\n"
    ),

    "description": (
        context_description +
        "Imagine you are a helpful agent operating automatically on the website and know everything about it. "
        "At the same time, you are also a user looking at a website and want to ask the agent and receive helpful answers from it. "
        "You must play both roles in this simulation and formulate good conversations between these two entities.\n\n"
        "Context: The user is currently looking at the website, trying to get an overall description of it. They are looking at a new website and would like to know "
        "important information about it. The agent will respond with an informative, condensed paragraph of the description. The description should capture all important information.\n\n"
        "Important: Ensure that the description is written in a natural and engaging way, avoiding robotic phrasing and repetitive vocabulary.\n\n"
        "Respond ONLY in the following valid JSON array format, without additional commentary:\n"
        "[\n"
        "  {\n"
        '    "question": "User question here",\n'
        '    "response": "Assistant response here"\n'
        "  }\n"
        "]\n"
    ),

    "complex_tasks": (
        context_description +
        "Imagine you are a helpful agent operating automatically on the website and know everything about it. "
        "At the same time, you are also a user looking at a website and want to ask the agent and receive helpful answers from it. "
        "You must play both roles in this simulation and formulate good conversations between these two entities. \n\n"
        "Context: The user is trying to command the agent to execute tasks on the website based on the current UI of the web. "
        "The action must involve multiple steps (4–5 actions) and must be grounded directly in the current UI of the web. This means the action must be possible given the elements "
        "present in the UI and not speculate about or guess functions that do not exist in the current image. The agent will act as an independent and automatic agent, "
        "reporting to the user step-by-step what it will do to achieve the goal. The agent must utilize the necessary elements in the image to achieve that goal. "
        "When referring to elements, the agent should attach the text box ID of the bounding boxes bounding that element to ensure grounded and truthful instructions. "
        "After that, the agent must determine and derive the first and foremost action it will perform specifically.\n\n"
        "Respond ONLY in the following valid JSON array format, without additional commentary:\n"
        "[\n"
        "  {\n"
        '    "question": "User question here",\n'
        '    "Instruction": "Step 1: Do this (Text Box ID 0: Gmail). Step 2: Then do this (Text Box ID 1: Google).",\n'
        '    "Next action": "What is able to be done first based on the plan that has been created?"\n'
        "  }\n"
        "]\n\n"
        "Important:\n"
        "- Ensure each action only involves elements with bounding boxes—do not include any interaction with elements that are not clearly indicated with bounding boxes.\n"
        "- Include box IDs in parentheses after mentioning any element to clearly identify which part of the interface to interact with (e.g., 'Select Gmail (Text Box ID 0: Gmail)').\n"
    )
}


    return enhanced_meta_prompts[level]

class ImageProcessor:
    def __init__(self, som_model, caption_model_processor):
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor

    def process_image(self, image: Image) -> Tuple[Image.Image, List[str]]:
        img_source = image.convert("RGB")
        img_arr = np.array(image)
        ocr_bbox_rslt, _ = self._perform_ocr(img_arr)
        text, ocr_bbox = ocr_bbox_rslt[0], ocr_bbox_rslt[1]
        return self._get_labeled_image(img_source, ocr_bbox, text)

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
        print("Initializing VLM model...")
        self.vlm_model, self.processor = self._initialize_vlm_model()
        print("Initializing SOM model and image processor...")
        self.image_processor = ImageProcessor(
            self._initialize_som_model(),
            self._initialize_caption_processor()
        )
        self.output_dir = "processed_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize JSON writers
        self.box_writer = StreamingJSONWriter(self.config.logging_file)
        self.conversation_writer = StreamingJSONWriter(self.config.output_file)

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
    
    def process_and_write_sample(self, sample: Dict) -> Tuple[str, List[str]]:
        """Process a single sample and write box properties to logging file"""
        try:
            # Process image
            with torch.no_grad():
                processed_image, coordinates, bounding_boxes = self.image_processor.process_image(sample["image"])
            
            # Save processed image
            image_name = f"processed_image_{sanitize_filename(sample['name'])}.png"
            image_path = os.path.join(self.output_dir, image_name)
            
            if isinstance(processed_image, Image.Image):
                processed_image.save(image_path)
                print(f"Saved processed image: {image_name}")

            # Create box list for logging
            box_data = {
                "image_name": image_name,
                "boxes_content": convert_ndarray_to_list(bounding_boxes),
                "coord": convert_ndarray_to_list(coordinates)
            }
            
            # Write to logging file
            self.box_writer.write_entry(box_data)
            return image_path, bounding_boxes

        except Exception as e:
            print(f"Error processing image {sample['name']}: {str(e)}")
            raise
        
    def generate_conversation_data(self, image_path: str, content_list: List[str]):
        """Generate and write conversation data for all levels"""
        try:
            levels = ["conversation", "description", "complex_tasks"]
            formatted_data = {
                "id": "image_00",
                "image": {
                    f"<image_00>": image_path  # Map single image path
                },
                "conversations": []
            }
            for level in levels:
                # Generate conversation for each level
                conversation = self._generate_single_level_conversation(image_path, content_list, level, "<image_00>")
                if conversation:
                    formatted_data["conversations"].extend(conversation)

            # Write the final formatted data
            self.conversation_writer.write_entry([formatted_data])

        except Exception as e:
            print(f"Error generating conversations for {image_path}: {str(e)}")
            raise


    def _generate_single_level_conversation(self, image_path: str, content_list: List[str], level: str, image_tag: str) -> List[Dict]:
        """Generate conversation data for a single level"""
        num_retry = 5
        meta_prompt = get_enhanced_meta_prompt(level, content_list)
        image = Image.open(image_path)
        # Lists of diverse prefixes
        instruction_prefixes = [
            "Here are the steps:",
            "Step-by-step guide:",
            "Follow these directions:",
            "Here’s how to proceed:",
            "Detailed steps:",
            "Steps to follow:",
            "Process outline:",
            "Here’s what to do:",
            "Let’s break it down:",
            "Guidelines to complete the task:"
        ]

        next_action_prefixes = [
            "Your next step is:",
            "The following action is:",
            "Here’s what you should do next:",
            "Proceed with the following step:",
            "The next move is:",
            "What’s next:",
            "Next up, you need to:",
            "Following this, take this action:",
            "Here’s the upcoming step:",
            "Advance to the next step:"
        ]
        # Prepare user message
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": meta_prompt}
            ]}
        ]

        for attempt in range(num_retry):
            try:
                # Process input
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
                output_text = self.processor.decode(
                    output[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=True
                )

                # Parse response
                try:
                    response_data = json.loads(output_text)
                    formatted_conversation = []
                    if level == "conversation" or level == "description":
                        formatted_conversation.append({
                            "role": f"user \n {image_tag}",
                            "content": item["question"]
                        })
                        formatted_conversation.append({
                            "role": "assistant",
                            "content": item["response"]
                        })

                    elif level == "complex_tasks":
                        instruction_prefix = random.choice(instruction_prefixes)
                        next_action_prefix = random.choice(next_action_prefixes)
                        for item in response_data:
                            formatted_conversation.append({
                                "role": f"user \n {image_tag}",
                                "content": item["question"]
                            })
                            formatted_conversation.append({
                                "role": "assistant",
                                "content": f"{instruction_prefix} {item['Instruction']}\n{next_action_prefix} {item['Next action']}"
                            })
                    else:
                        formatted_conversation = []
                    return formatted_conversation

                except json.JSONDecodeError:
                    print(f"Failed to parse JSON on attempt {attempt + 1}")
                    continue
            except Exception as e:
                print(f"Generation error on attempt {attempt + 1}: {str(e)}")
                continue

        print(f"Failed to generate valid response for level {level} after {num_retry} attempts")
        return []


    
    def close_writer(self):
        self.box_writer.close()
        self.conversation_writer.close()


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
    # Parse arguments and create config
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("--vlm_model_name", type=str, default="meta-llama/Llama-3.2-90B-Vision-Instruct")
    parser.add_argument("--yolo_model_path", type=str, default="OmniParser/weights/icon_detect/best.pt")
    parser.add_argument("--caption_model_name", type=str, default="blip2")
    parser.add_argument("--caption_model_path", type=str, default="OmniParser/weights/icon_caption_blip2")
    parser.add_argument("--output_file", type=str, default="data.json")
    parser.add_argument("--logging_file", type=str, default="box_prop.json")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    
    args = parser.parse_args()
    config = PipelineConfig(**vars(args))
    
    # Initialize components
    print("Loading models...")
    load_and_save_model()
    
    dataset_instance = Dataset()
    generator = SyntheticDataGenerator(config)
    
    try:
        # Get full dataset
        full_ds = dataset_instance.select_data(config.start_index, config.end_index)
        total_samples = len(full_ds)
        
        # Process in batches
        with tqdm(total=total_samples, desc="Processing samples") as pbar:
            for batch in dataset_instance.process_data_in_batches(
                config.start_index, 
                config.end_index, 
                args.batch_size
            ):
                # Process each sample in batch
                for sample in batch:
                    try:
                        # Process image and write box data
                        image_path, content_list = generator.process_and_write_sample(sample)
                        
                        # Generate and write conversation data
                        generator.generate_conversation_data(image_path, content_list)

                        print(f"Having saved: {len(os.listdir("processed_images"))}/{total_samples} images so far. ")
                        with open(config.logging_file, "r") as file:
                            data = json.load(file)
                        num_box_prop = len(data)
                        print(f"Having saved: {num_box_prop}/{total_samples} samples in logging file so far. ")
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error processing sample {sample['name']}: {str(e)}")
                        continue
                
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Fatal error during processing: {str(e)}")
        raise
    
    finally:
        # Close JSON writers
        generator.close_writer()
        print(f"Results saved to {config.logging_file} and {config.output_file}")

if __name__ == "__main__":
    main()
