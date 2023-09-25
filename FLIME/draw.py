from PIL import Image
im = Image.open(r"F:\Lime-pytorch\data3\019\myplot.png")
im_resized = im.resize((224, 224))
im_resized.save(r"F:\Lime-pytorch\data3\019\2.jpg")
