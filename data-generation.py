from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import argparse
import json
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, load_dataset

from OmniParser.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

@dataclass
class PipelineConfig:
    vlm_model_name: str = "openbmb/MiniCPM-V-2_6"
    yolo_model_path: str = "OmniParser/weights/icon_detect/best.pt"
    caption_model_name: str = "blip2"
    caption_model_path: str = "OmniParser/weights/icon_caption_blip2"
    output_file: str = "data.json"
    start_index: int = 0
    end_index: int = None

def get_enhanced_meta_prompt(level: str, parsed_content_list: List[str]) -> str:
    context_description = (
        f"You are given an image with several labeled elements. Here is the context of the visible items on this screen:\n"
        f"{parsed_content_list}\n\n"
    )
    
    enhanced_meta_prompts = {
        "conversation": (
            context_description +
            "Imagine a user is exploring this website and has questions about its contents or functionality. Based on this information, create three user questions and responses in a conversational style.\n\n"
            "Please return each question and answer in a structured JSON format as shown below:\n"
            "{\n"
            "  'question': 'User question here',\n"
            "  'response': 'Assistant response here'\n"
            "}\n\n"
            "- For the first question, make it brief and focused on a single element, such as asking about recent updates, a specific section, or basic site functionality. For example, 'What's new on the homepage?' or 'How do I check my notifications?'\n"
            "- For the second question, add complexity by referencing multiple elements or comparing sections. For instance, 'What's the difference between the main articles and trending topics sections?'\n"
            "- For the third question, make it the most detailed, covering questions about how different site sections interact or summarizing multiple updates. For example, 'Can you summarize today's updates from the news, notifications, and recommendations?' Avoid mentioning specific technical terms like 'box IDs' or 'coordinates.'"
        ),
        "description": (
            context_description +
            "Imagine a user is trying to get an overview of the website's layout and its main sections. Create three user questions and responses that provide increasingly detailed descriptions.\n\n"
            "Please return each question and answer in a structured JSON format as shown below:\n"
            "{\n"
            "  'question': 'User question here',\n"
            "  'response': 'Assistant response here'\n"
            "}\n\n"
            "- The first response should describe a single key section or feature, such as 'What can I find on the homepage?' or 'What's included in the profile page?'\n"
            "- The second response should cover multiple sections or a broader area of the site, like 'What's available in the navigation menu and sidebar?' or 'Can you describe the options available in the settings page?'\n"
            "- The third response should offer a comprehensive overview of the visible sections, such as 'What are the main parts of this page, including the sidebar, main content area, and footer?' or 'Can you summarize the features available across the dashboard?' Avoid specific technical terms like 'box IDs' or 'coordinates.'"
        ),
        "complex_tasks": (
            context_description +
            "Imagine a user wants to perform specific actions on this website. Based on the context, create three progressively complex user questions and responses with detailed, step-by-step instructions.\n\n"
            "You are provided with a list of bounding boxes description that you can see in the image. I want you to strictly adhere to those bounding boxes when refering to the elements in the image.\n"
            "For example: To search for 'BBC News' click on the search bar that is labeled Text Box ID 0 and then type BBC News from the keyboard... \n"
            "Do you understand? \n"
            "Please return each question and answer in a structured JSON format as shown below:\n"
            "{\n"
            "  'question': 'User question here',\n"
            "  'response': 'Assistant response here'\n"
            "}\n\n"
            "- The first response should be for a simple, one-step action, such as 'How do I search for a product?' or 'How do I open the settings?'\n"
            "- The second response should be for a multi-step task involving interactions with different elements, such as 'How can I add a product to my cart and proceed to checkout?' or 'How do I share an article link from this page?'\n"
            "- The third response should cover a more complex sequence involving several steps and interactions. For example, 'How can I log in, navigate to my profile, update my preferences, and save changes?' or 'Can you guide me through viewing an order history, selecting a specific order, and contacting support about that order?' Include references to specific buttons, labels, or visible text to help clarify each step where needed."
        )
    }


    return enhanced_meta_prompts[level]

class ImageProcessor:
    def __init__(self, som_model, caption_model_processor):
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor

    def process_image(self, image: Image) -> Tuple[Image.Image, List[str]]:
        ocr_bbox_rslt, _ = self._perform_ocr(image)
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
        labeled_img, coords, content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_THRESHOLD=0.03,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config={'text_scale': 0.8},
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.1,
            imgsz=640
        )
        return labeled_img, content_list

class SyntheticDataGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vlm_model = self._initialize_vlm_model()
        self.tokenizer = self._initialize_tokenizer()
        self.image_processor = ImageProcessor(
            self._initialize_som_model(),
            self._initialize_caption_processor()
        )

    def _initialize_vlm_model(self) -> AutoModel:
        return AutoModel.from_pretrained(
            self.config.vlm_model_name,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16
        ).eval().cuda()

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

    def generate_data(self, dataset: Dataset) -> List[Dict]:
        all_formatted_data = []
        for sample in tqdm(dataset, desc="Processing samples"):
            labeled_img, content_list = self.image_processor.process_image(sample['image'])
            
            for level in ["conversation", "description", "complex_tasks"]:
                formatted_data = self._generate_level_data(level, labeled_img, content_list)
                all_formatted_data.extend(formatted_data)
        
        return all_formatted_data

    def _generate_level_data(self, level: str, labeled_img: Image, content_list: List[str]) -> List[Dict]:
        meta_prompt = get_enhanced_meta_prompt(level, content_list) 
        msgs = [{'role': 'user', 'content': [labeled_img, meta_prompt]}]
        
        response = self.vlm_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        user_prompt = response["question"]
        llm_response = response["response"]
        
        return [
            self._format_data(level, user_prompt, llm_response, labeled_img)
        ]

    @staticmethod
    def _format_data(level: str, prompt: str, response: str, image_data: Image) -> Dict:
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": f"You are a helpful assistant"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image_data}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ]
        }

class Dataset:
    def __init__(self):
        self.ds = load_dataset("agentsea/wave-ui-25k")

    def process_data(self):
        filtered_ds = self.ds.filter(lambda example: example["platform"] == "web")
        return filtered_ds

    def select_data(self, start_index, end_index=None):
        ds = self.process_data()
        if end_index:
            return ds.select(range(start_index, end_index))
        return ds['train']

def main():
    #Load the subdirectories of Omni Parser into weights
    blip2 = AutoModel.from_pretrained(
        "microsoft/OmniParser",
        subfolder="icon_caption_blip2",
        cache_dir="/kaggle/working/omni/weights/icon_caption_blip2"
    )

    icon_detect = AutoModel.from_pretrained(
        "microsoft/OmniParser",
        subfolder="icon_detect",
        cache_dir="/kaggle/working/omni/weights/icon_detect"
    )

    florence = AutoModel.from_pretrained(
        "microsoft/OmniParser",
        subfolder="icon_caption_florence",
        cache_dir="/kaggle/working/omni/weights/icon_caption_florence"
    ) 
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("--vlm_model_name", type=str, default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--yolo_model_path", type=str, default="OmniParser/weights/icon_detect/best.pt")
    parser.add_argument("--caption_model_name", type=str, default="blip2")
    parser.add_argument("--caption_model_path", type=str, default="OmniParser/weights/icon_caption_blip2")
    parser.add_argument("--output_file", type=str, default="data.json")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for dataset split")
    parser.add_argument("--end_index", type=int, help="Ending index for dataset split")
    
    args = parser.parse_args()
    config = PipelineConfig(**vars(args))
    
    # Load and slice dataset
    dataset_instance = Dataset()
    if config.end_index:
        dataset = dataset_instance.select_data(config.start_index, config.end_index)
    else:
        dataset = dataset_instance.select_data(config.start_index)

    generator = SyntheticDataGenerator(config)
    formatted_data = generator.generate_data(dataset)
    
    with open(config.output_file, "w") as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Data saved to {config.output_file}")

if __name__ == "__main__":
    main()


