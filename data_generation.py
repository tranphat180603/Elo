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
from data_process import Processor

login("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")

elm_tok = "<element>"
elm_end_tok = "</element>"

elms_tok = "<elements>"
elms_end_tok = "<elements>"

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
    processed_file: str = "processed_box_prop.json"
    start_index: int = 0
    end_index: int = None

def get_enhanced_meta_prompt(level: str, parsed_content_list: List[str]) -> str:
    context_description = (
        f"You are given a website screenshot with elements that have been bounded by boxes to increase precision. You must use the bounding boxes' properties as grounding truth when refering to elements. This helps increasing accuracy."
        f"Here is the list of the bounding boxes on this screen and their corresponding elements:\n"
        f"{elms_tok}{parsed_content_list}{elms_end_tok}\n\n"
    )
    
    context_description = (
        f"You are given a website screenshot with elements that have been bounded by boxes to increase precision. "
        f"You must use the bounding boxes' properties as the grounding truth when referring to elements. This ensures accuracy.\n"
        f"Here is the list of the bounding boxes on this screen and their corresponding elements:\n"
        f"{elms_tok}{parsed_content_list}{elms_end_tok}\n\n"
    )
    
    enhanced_meta_prompts = {
        "conversation": (
            context_description +
            "Imagine you are a helpful agent that operates automatically on the website and knows everything about it. "
            "At the same time, you are also a user exploring the website and asking questions about its components and functionalities. "
            "You must play both roles in this simulation and formulate meaningful conversations between these two entities.\n\n"
            "Context: The user is currently exploring the website's components, such as buttons, links, text boxes, dropdown menus, and other interactive elements. "
            "The agent's role is to provide accurate and informative answers grounded in the given bounding box properties.\n\n"
            f"VERY IMPORTANT: The response of the agent should strictly adhere to the following rule:\n"
            f"- When referring to any element, it must use the exact Text Box ID provided (e.g., {elm_tok}Text Box ID 5{elm_end_tok}).\n"
            f"- The content or value of the text box (e.g., 'MICHAELILEWIS') can be rephrased, corrected, or rewritten based on the agent's understanding of English and website design knowledge.\n\n"
            f"For example: 'On the left of the website, there's a menu bar {elm_tok}Text Box ID 5{elm_end_tok} containing categories like shirts and jackets. "
            f"On the right, there are advertisements {elm_tok}Text Box ID 1{elm_end_tok} promoting a marathon in Vietnam.'\n\n"
            "Based on this information, create 2 meaningful and grounded conversations between the user and the agent. "
            "Each conversation should involve one question from the user and one accurate response from the agent that references elements from the bounding boxes.\n\n"
            "Please respond ONLY in the following valid JSON format, without any additional commentary or formatting:\n"
            "[\n"
            "  {\n"
            '    "question": "User question here",\n'
            '    "response": "Assistant response here"\n'
            "  },\n"
            "  {\n"
            '    "question": "User question here",\n'
            '    "response": "Assistant response here"\n'
            "  }\n"
            "]\n\n"
            "Ensure that there is no text outside of the JSON structure, as this will be parsed directly. Follow these instructions strictly to avoid parsing errors.\n\n"
        ),

        "description": (
            context_description +
            "Imagine you are a helpful agent that operates automatically on the website and knows everything about it. "
            "At the same time, you are also a user exploring the website and asking for an overall description of it.\n\n"
            "Context: The user is currently looking at the website and trying to get an informative summary of it. "
            "The agent's role is to provide a detailed yet concise description of the website, mentioning important elements and their purposes after that.\n\n"
            f"VERY IMPORTANT: The response of the agent should follow the rule: Whenever referring to the description of each element, it must be nested inside the special tokens {elm_tok} and {elm_end_tok}.\n\n"
            f"For example: 'The main navigation bar includes options for browsing, such as a search bar {elm_tok}Text Box ID 0{elm_end_tok} and a magnifying glass which is likely a search button {elm_tok}Text Box ID 1{elm_end_tok}.'\n\n"
            "The description should capture all significant information about the website's purpose, functionality, and key elements, written in an engaging and natural way.\n\n"
            "Respond ONLY in the following valid JSON array format, without additional commentary:\n"
            "[\n"
            "  {\n"
            '    "question": "User question here",\n'
            '    "response": "Assistant response here"\n'
            "  }\n"
            "]\n\n"
        ),

    "simple_tasks": (
        context_description +
        "Imagine you are a helpful agent operating automatically on the website. Your task is to simulate single, one-step actions that a user might perform on the website.\n\n"
        "Context: The user is exploring the website and asking the agent to interact with different elements. "
        "These elements may be interactable (e.g., buttons, text fields) or non-interactable (e.g., icons, static text). "
        "The agent must accurately determine if an action is possible and respond accordingly. If the element is interactable, the agent should provide a detailed action plan. "
        "If it is not interactable, the agent must explain why and, if appropriate, suggest an alternative action.\n\n"
        f"VERY IMPORTANT: The response of the agent should follow these rules:\n"
        f"1. When referring to elements, use the exact Text Box ID provided (e.g., {elm_tok}Text Box ID 10{elm_end_tok}).\n"
        f"2. The description of each element can be rephrased or improved based on the agent's understanding of English and general website knowledge.\n\n"
        f"Examples:\n"
        f"1. If possible:\n"
        f"User: Can you click on the image on the post of my friend Quan Nguyen?\n"
        f"Assistant: Sure, let's navigate the mouse to the middle of the image {elm_tok}Text Box ID 10{elm_end_tok} and left-click on it with the mouse.\n\n"
        f"2. If not possible:\n"
        f"User: I want to know more about the person Quan Nguyen mentions in his post. Can you click on the name of that person to open his Facebook profile?\n"
        f"Assistant: It seems like Quan Nguyen didn't directly tag Viet Hoang on the post. Maybe let's try searching Viet Hoang on Facebook search bar?\n\n"
        "Your task is to create 2 meaningful and diverse conversations where the user asks the agent to perform simple tasks. "
        "The agent must provide grounded and accurate responses based on the bounding boxes while generating diverse types of interactions.\n\n"
        "Please respond ONLY in the following valid JSON format, without any additional commentary or formatting:\n"
        "[\n"
        "  {\n"
        '    "question": "User question here",\n'
        '    "response": "Assistant response here"\n'
        "  },\n"
        "  {\n"
        '    "question": "User question here",\n'
        '    "response": "Assistant response here"\n'
        "  }\n"
        "]\n\n"
        "Ensure that there is no text outside of the JSON structure, as this will be parsed directly. Follow these instructions strictly to avoid parsing errors.\n\n"
    ),
    "complex_tasks": (
        context_description +
        "Imagine you are a helpful agent operating automatically on the website and know everything about it. "
        "At the same time, you are also a user looking at a website and want to ask the agent to execute complex tasks step-by-step based on the UI of the website. "
        "You must play both roles in this simulation and create meaningful interactions between these two entities.\n\n"
        "Context: The user is trying to command the agent to execute a task that requires multiple steps (4–5 actions). "
        "The actions must be grounded directly in the current UI of the web and must only use elements present in the bounding boxes. "
        "The agent must analyze the task and plan the required steps step-by-step, ensuring the plan is possible and utilizes only the available elements in the bounding boxes.\n\n"
        f"VERY IMPORTANT: Whenever referring to any element, it must be nested inside the special tokens {elm_tok} and {elm_end_tok}.\n\n"
        f"For example: 'Click the search bar and type a query: {elm_tok}Text Box ID 4{elm_end_tok}.'\n\n"
        "Instructions:\n"
        "- Break down the task into multiple steps (4–5).\n"
        "- Reference bounding box descriptions for all elements, using the special tokens.\n"
        "- Generate a clear and actionable plan for the task.\n\n"
        "Please respond ONLY in the following valid JSON array format, without any additional commentary or formatting:\n"
        "[\n"
        "  {\n"
        '    "question": "User question here. For example: I want to find Steve Jobs iconic shirt to buy it, can you help me?",\n'
        f'    "Instruction": "For example: Step 1: Move the mouse to the search bar {elm_tok}Text Box ID 0{elm_end_tok}. Step 2: Click and type "Turtlekneck sweatshirt like Steve Jobs" from your keyboard. Step 3: To perform search, click on the magnifying glass {elm_tok}Text Box ID 1{elm_end_tok}.",\n'
        '    "Next action": "What should be done first based on the planned steps?"\n'
        "  }\n"
        "]\n\n"
        "Important:\n"
        "- Ensure each action involves only elements with bounding boxes—do not include interactions with elements not indicated in the bounding boxes.\n"
        f"- Use special tokens {elm_tok} and {elm_end_tok} to clearly mark referenced elements in instructions.\n"
        "- Avoid speculating about non-existent functions or elements that are not present in the bounding boxes.\n\n"
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
            easyocr_args={'paragraph': False, 'text_threshold': 0.6},
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
            BOX_TRESHOLD=0.05,
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
        self.output_dir = "Elo/processed_images/train"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.box_writer = StreamingJSONWriter(self.config.logging_file)
        self.conversation_writer = StreamingJSONWriter(self.config.output_file)

        self.data_processor = Processor(processed_file_path=config.processed_file)
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
        image_dir = "Elo/proccesed_images"
        try:
            # Process image
            with torch.no_grad():
                processed_image, coordinates, bounding_boxes = self.image_processor.process_image(sample["image"])
            
            # Save processed image
            image_name = f"processed_image_{sanitize_filename(sample['name'])}.png"
            image_path = os.path.join(self.output_dir, image_name)

            # Check if the processed image is valid and save it
            if isinstance(processed_image, Image.Image):
                try:
                    processed_image.save(image_path)
                    print(f"Saved processed image: {image_name}")

                    # Create box list for logging
                    box_data = {
                        "image_name": image_name,
                        "boxes_content": convert_ndarray_to_list(bounding_boxes),
                        "coord": convert_ndarray_to_list(coordinates)
                    }

                    # Write to logging file only if the image is successfully saved
                    self.box_writer.write_entry(box_data)
                    return image_path, bounding_boxes, coordinates

                except Exception as save_error:
                    print(f"Failed to save image {image_name}: {str(save_error)}")
                    raise save_error  # Stop processing this sample if image saving fails
            else:
                raise ValueError(f"Invalid processed image for {sample['name']}.")

        except Exception as e:
            print(f"Error processing image {sample['name']}: {str(e)}")
            raise

    def generate_conversation_data(self, image_path: str, content_list: List[str]):
        """Generate and write conversation data for all levels"""
        try:
            levels = ["conversation", "description", "simple_tasks" ,"complex_tasks"]
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
        # Lists of diverse prefixes. The purpose is to not fixate any kind of response/word into the model.
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
                if level == "conversation" or level == "description" or level == "simple_tasks":
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

        print(f"Failed to generate valid response for level {level} after {num_retry} attempts")
        return []
    
    def close_writer(self):
        self.box_writer.close()
        self.conversation_writer.close()

#inference nen minh k nghi load data theo batch o day no se co ich loi gi
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
    parser.add_argument("--processed_file", type=str, default="processed_box_prop.json")   
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
                        image_path, content_list, _ = generator.process_and_write_sample(sample)

                        print(f"Having saved: {len(os.listdir("processed_images"))}/{total_samples} images so far.")
                        with open(config.logging_file, "r") as file:
                            data = json.load(file)
                        num_box_prop = len(data)
                        print(f"Having saved: {num_box_prop}/{total_samples} samples in logging file so far. ")
                        _, box_prop = generator.data_processor.process(image_path, content_list)
                        # Generate and write conversation data
                        generator.generate_conversation_data(image_path, box_prop)
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
        print(f"WE, AS NEWCOMERS, HAVE ACHIEVED INTELLIGENCE!!!")

if __name__ == "__main__":
    main()
