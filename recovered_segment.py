'''
Script to segment a inputted image (via filepath) into COCO classes split by n binary masks,
where n is the number of detected classes. This script then proceeds to show the outputted
binary masks and prompt the user which he/she want to condition Stable Diffusion on
(for now let's just do one subject).  
'''

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

def run_inference(image, threshold=0.5):
    # load MaskFormer fine-tuned on COCO panoptic segmentation
    processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], return_binary_maps=True, threshold=threshold)[0]
    predicted_map = result["segmentation"]

    desc_str = ''
    desc_list = []
    for z in result['segments_info']:
        label_id = model.config.id2label[z['label_id']]
        id = z['id']
        s = z['score']
        desc_str += f"ID {id}: label id {label_id}, score: {s}\n"
        desc_list.append(f"ID: {id}, label id: {label_id}, score: {s}")

    return predicted_map, desc_list # stacked tensor of binary masks, string description for id selection


# read in args
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', help='Save your desired input image in ~StableComics/Attend-and-Excite-II/imgs_input')
args = parser.parse_args()


# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(args.image_path)

predicted_map, desc = run_inference(image)

f,a = plt.subplots(1,predicted_map.shape[0], figsize=(25, 25))
for i in range(predicted_map.shape[0]):
    a[i].set_title(desc[i])
    a[i].imshow(predicted_map[i,:,:])

plt.axis('off')
# tmp_loc = '~/StableComics/Attend-and-Excite-II/imgs_tmp/binary_masks.png'
tmp_loc = 'binary_masks.png'
plt.savefig(tmp_loc)

keep_id = int(input(f"Please reference {tmp_loc} and enter the id or index of the binary mask you would like to use to condition on (only choose one for now):"))
assert keep_id < predicted_map.shape[0]

image_tsnr = torch.tensor(np.array(image))

predicted_map_i = predicted_map[keep_id,:,:].int()

# repeat mask over 3 image channels 
predicted_map_i = np.repeat(predicted_map_i[:, :, np.newaxis], 3, axis=2) 

mask = image_tsnr * predicted_map_i

plt.clf()
plt.imshow(mask, interpolation='nearest')
# mask_loc = '~/StableComics/Attend-and-Excite-II/imgs_condition/mask.png'
mask_loc = 'mask.png'
plt.savefig(mask_loc) # is it ok the mask is 3 channel?
print(f"Saved mask here: {mask_loc}. Feel free to rename it if you want to stick around.")