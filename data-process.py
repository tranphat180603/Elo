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

from datasets import load_dataset
import json
class DataProcess():
    def __init__(self, image_ds_id = "tranphat1806/processed-WaveUI") -> None:
        self.image_ds = load_dataset(image_ds_id)

    def process(self,):
        with open("box_prop1.json", "r") as file:
            box_prop1 = json.load(file)
        with open("box_prop2.json", "r") as file:
            box_prop2 = json.load(file)
        
        #merge and remove duplicate
        raw_unif_list = box_prop1 + box_prop2
        raw_list = list(map(json.loads, set(map(json.dumps, raw_unif_list)))) #remove duplicate

        image_ds = self.image_ds['train']
        existed_img = image_ds['image']

        for sample in raw_list:
            if sample['image_name']