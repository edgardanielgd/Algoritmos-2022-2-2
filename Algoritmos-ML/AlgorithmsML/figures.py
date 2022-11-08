from PIL import Image
import numpy as np

def openImage( path ):
  im = Image.open( path )
  width, height = im.size
  return width, height, list(im.getdata())

def createImage( images ):

  PILImages = []
  
  for image in images:
    pixels, w, h = image
    data = np.array(pixels, dtype=np.uint8)
    im = Image.fromarray( data, "RGB")
    PILImages.append( im )

  return PILImages

        
  