from OmniParser.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import argparse
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, load_dataset

@dataclass
class PipelineConfig:
    vlm_model_name: str = "openbmb/MiniCPM-V-2_6"
    yolo_model_path: str = "OmniParser/weights/icon_detect/best.pt"
    caption_model_name: str = "blip2"
    caption_model_path: str = "OmniParser/weights/icon_caption_blip2"
    output_file: str = "output_synthetic_data.json"
    log_dir: str = "logs"
    batch_size: int = 1

class LoggerSetup:
    @staticmethod
    def setup(log_dir: str) -> logging.Logger:
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('SyntheticDataPipeline')
        logger.setLevel(logging.INFO)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f'pipeline_{timestamp}.log'
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

# [Previous meta-prompt function remains unchanged]
def get_enhanced_meta_prompt(level: str, parsed_content_list: List[str]) -> str:
    # ... [Keep your original implementation]
    pass

class ImageProcessor:
    def __init__(self, som_model, caption_model_processor, logger: logging.Logger):
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor
        self.logger = logger
        self.stats = {
            'total_images': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'processing_times': []
        }

    def process_image(self, image: Image) -> Tuple[Image.Image, List[str]]:
        start_time = time.time()
        self.stats['total_images'] += 1
        
        try:
            ocr_bbox_rslt, _ = self._perform_ocr(image)
            text, ocr_bbox = ocr_bbox_rslt[0], ocr_bbox_rslt[1]
            
            if text:
                self.stats['successful_ocr'] += 1
            else:
                self.stats['failed_ocr'] += 1
                self.logger.warning("OCR produced no text for image")
            
            labeled_img, content_list = self._get_labeled_image(image, ocr_bbox, text)
            
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            self.logger.debug(f"Image processed in {processing_time:.2f} seconds")
            return labeled_img, content_list
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise

    def get_statistics(self) -> Dict:
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        return {
            'total_images_processed': self.stats['total_images'],
            'successful_ocr_rate': self.stats['successful_ocr'] / self.stats['total_images'] if self.stats['total_images'] > 0 else 0,
            'average_processing_time': avg_time,
            'failed_ocr_count': self.stats['failed_ocr']
        }

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
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.info(f"Initializing SyntheticDataGenerator with config: {config}")
        
        self.stats = {
            'total_samples': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'start_time': time.time()
        }
        
        try:
            self.vlm_model = self._initialize_vlm_model()
            self.tokenizer = self._initialize_tokenizer()
            self.image_processor = ImageProcessor(
                self._initialize_som_model(),
                self._initialize_caption_processor(),
                logger
            )
            self.logger.info("Successfully initialized all models")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def generate_data(self, dataset: Dataset) -> List[Dict]:
        self.logger.info(f"Starting data generation for {len(dataset)} samples")
        all_formatted_data = []
        
        try:
            # Create progress bar
            progress_bar = tqdm(total=len(dataset), desc="Processing samples")
            
            for sample in dataset:
                self.stats['total_samples'] += 1
                
                try:
                    labeled_img, content_list = self.image_processor.process_image(sample['image'])
                    
                    for level in ["conversation", "description", "complex_tasks"]:
                        formatted_data = self._generate_level_data(level, labeled_img, content_list)
                        all_formatted_data.extend(formatted_data)
                        self.stats['successful_generations'] += 1
                        
                except Exception as e:
                    self.stats['failed_generations'] += 1
                    self.logger.error(f"Error processing sample {self.stats['total_samples']}: {str(e)}")
                
                # Update progress
                progress_bar.update(1)
                if self.stats['total_samples'] % 10 == 0:
                    self._log_progress()
            
            progress_bar.close()
            self._log_final_statistics()
            
            return all_formatted_data
            
        except Exception as e:
            self.logger.error(f"Fatal error in generate_data: {str(e)}")
            raise

    def _log_progress(self):
        elapsed_time = time.time() - self.stats['start_time']
        success_rate = (self.stats['successful_generations'] / 
                       (self.stats['total_samples'] * 3)) if self.stats['total_samples'] > 0 else 0
        
        self.logger.info(
            f"Progress: {self.stats['total_samples']} samples processed | "
            f"Success rate: {success_rate:.2%} | "
            f"Elapsed time: {elapsed_time:.2f}s"
        )

    def _log_final_statistics(self):
        image_stats = self.image_processor.get_statistics()
        elapsed_time = time.time() - self.stats['start_time']
        
        final_stats = {
            "total_samples_processed": self.stats['total_samples'],
            "successful_generations": self.stats['successful_generations'],
            "failed_generations": self.stats['failed_generations'],
            "success_rate": self.stats['successful_generations'] / (self.stats['total_samples'] * 3),
            "total_processing_time": elapsed_time,
            "average_time_per_sample": elapsed_time / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0,
            "image_processing_stats": image_stats
        }
        
        self.logger.info("Final Statistics:")
        self.logger.info(json.dumps(final_stats, indent=2))

    def _initialize_vlm_model(self) -> AutoModel:
        return AutoModel.from_pretrained(
            self.config.vlm_model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
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
        
        for sample in dataset:
            labeled_img, content_list = self.image_processor.process_image(sample['image'])
            
            for level in ["conversation", "description", "complex_tasks"]:
                formatted_data = self._generate_level_data(level, labeled_img, content_list)
                all_formatted_data.extend(formatted_data)
        
        return all_formatted_data

    def _generate_level_data(self, level: str, labeled_img: Image, content_list: List[str]) -> List[Dict]:
        meta_prompt = get_enhanced_meta_prompt(level, content_list)  # Using original meta-prompt function
        msgs = [{'role': 'user', 'content': [labeled_img, meta_prompt]}]
        
        response = self.vlm_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        
        return [
            self._format_data(level, prompt, response, labeled_img)
            for prompt, response in response.items()
        ]

    @staticmethod
    def _format_data(level: str, prompt: str, response: str, image_data: Image) -> Dict:
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": f"You are a helpful assistant that assist user in generating synthetic data based on user's prompt and image."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image_data}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ]
        }

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("--vlm_model_name", type=str, default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--yolo_model_path", type=str, default="OmniParser/weights/icon_detect/best.pt")
    parser.add_argument("--caption_model_name", type=str, default="blip2")
    parser.add_argument("--caption_model_path", type=str, default="OmniParser/weights/icon_caption_blip2")
    parser.add_argument("--output_file", type=str, default="output_synthetic_data.json")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    config = PipelineConfig(**vars(args))
    
    # Setup logger
    logger = LoggerSetup.setup(config.log_dir)
    logger.info("Starting synthetic data generation pipeline")
    
    try:
        dataset = load_dataset("biglab/webui-350k-elements", split="train")
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        
        generator = SyntheticDataGenerator(config, logger)
        formatted_data = generator.generate_data(dataset)
        
        with open(config.output_file, "w") as f:
            json.dump(formatted_data, f, indent=4)
        logger.info(f"Successfully saved synthetic data to {config.output_file}")
        
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()