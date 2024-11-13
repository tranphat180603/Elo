import os 
import shutil

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

#load dataset WebUI 350k element
#process image with OmniParser
#meta-prompt into a VLM