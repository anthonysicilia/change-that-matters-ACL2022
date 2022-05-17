import sys
from PIL import Image

# courtesy of this overflow post
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
images = [Image.open(x) for x in ['error-gap-analysis.png', 'estim-error-analysis.png']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('all.png')