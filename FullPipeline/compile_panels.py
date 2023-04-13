import json
from PIL import Image

outputs_dir = './outputs'
frame_x = 768
frame_y = 768
padding = 10
bg_color = (0, 0, 0)
n = 3

with open('human_choice.json', 'r') as f:
    comics = json.load(f)['comics']

for comic in comics:
    _id = comic['id']
    _style = comic['style']
    strip = Image.new('RGB', (n * frame_x + (n-1) * padding, frame_y), bg_color)
    strip.paste(Image.open(f'{outputs_dir}/{_id}/0/{comic["frame_0"]:02d}_{_style}.png'), (0, 0))
    strip.paste(Image.open(f'{outputs_dir}/{_id}/1/{comic["frame_1"]:02d}_{_style}.png'), (frame_x+padding, 0))
    strip.paste(Image.open(f'{outputs_dir}/{_id}/2/{comic["frame_2"]:02d}_{_style}.png'), (2*(frame_x+padding), 0))
    strip.save(f'{outputs_dir}/{_id}/{_id}_strip_{_style}.png')