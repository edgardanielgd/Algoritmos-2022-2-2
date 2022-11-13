from PIL import Image
import numpy as np

def openImage( path ):
  try:
    im = Image.open( path ).convert("RGB")
    width, height = im.size
    return width, height, list(im.getdata())
  except:
    return 0,0, None

def createImage( images ):

  PILImages = []
  
  for image in images:
    pixels, w, h = image
    data = np.array(pixels, dtype=np.uint8)
    im = Image.fromarray( data, "RGB")
    PILImages.append( im )

  return PILImages

        
  