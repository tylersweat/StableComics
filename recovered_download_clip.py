from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch


model_id = "openai/clip-vit-large-patch14"

# we initialize a tokenizer, image processor, and the model itself
# tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# save models to this directory
processor.save_pretrained("./downloaded_models/clip-vit-large-patch14-processor")
model.save_pretrained("./downloaded_models/clip-vit-large-patch14-model")


