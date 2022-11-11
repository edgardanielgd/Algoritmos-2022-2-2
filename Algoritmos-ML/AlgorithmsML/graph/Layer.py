from AlgorithmsML.graph.Base import *
from AlgorithmsML.figures import createImage

# Dimensions:
# 0 : Width
# 1 : Height
# 2 : Number of features

class Layer:
  def __init__(self, dimensions):
    # Dimensions should be 3D arrays
    self.nodes = []

    self.nextLayer = None

    self.prevLayer = None
    
    self.dimensions = dimensions

    self.num_nodes = 0
    
  def addSuperLayer( self, upperLayer ):
    self.nextLayer = upperLayer
    upperLayer.onSublayerAdd( self )

  def passDataRecursive( self, label_index = None ):
    self.calculate()

    input_gradient = None
    loss = None

    if self.nextLayer is not None:
      input_gradient, loss = self.nextLayer.passDataRecursive( label_index )

    if label_index is not None:
      # Data is being trained
      if self.nextLayer is not None:
        return self.backpropagate( input_gradient, loss )
      else:
        return self.backpropagate( label_index , loss )

  def calculate(self):
    pass

  def backpropagate(self, output_gradient, loss):
    return output_gradient, loss
    
  def onSublayerAdd( self, subLayer ):
    self.prevLayer = subLayer
    
  def toImage( self, RGB_channel = True ):
    images = []

    features = self.dimensions[2]
    width = self.dimensions[1]
    height = self.dimensions[0]

    if RGB_channel:
      # Combine r,g and b channels
      no_rgb_features = features // 3
      for ifeature in range( no_rgb_features ):
        pixels_arr = []
  
        for y in range( height ):
          
          pixels_row = []
          for x in range( width ):

            # Red value
            node = self.nodes [
              ( ( y * width + x ) * 3 ) * no_rgb_features + ifeature
            ]

            red = node.value * 255

            # Green value
            node = self.nodes [
              ( ( y * width + x ) * 3 + 1) * no_rgb_features + ifeature
            ]

            green = node.value * 255

            # Blue value
            node = self.nodes [
              ( ( y * width + x ) * 3 + 2) * no_rgb_features + ifeature
            ]

            blue = node.value * 255

            pixels_row.append( (red, green, blue) )

          pixels_arr.append( pixels_row )
#         print( pixels_arr )
        images.append( ( pixels_arr, width, height ) )
    
    else:
      # Use only a single color, the same for all channels
      

      for ifeature in range( features ):
        pixels_arr = []
  
        for y in range( height ):

          pixels_row = []
          
          for x in range( width ):
            
            node = self.nodes [
              ( y * width + x ) * features + ifeature
            ]

            value = node.value * 255
            pixels_row.append( (value, value, value) )

          pixels_arr.append( pixels_row )
          
        images.append( ( pixels_arr, width, height ) )
      
    return createImage( images )

  def saveData( self , filename ):
    pass
  
  def loadData( self, filename ):
    pass