from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

INPUT_EMBEDED_PIXELS = 1
INPUT_MATRIX_PIXELS = 2
INPUT_MATRIX_ONE_CHANNEL = 3

class InputLayer( Layer ):
  def __init__ ( self, w, h, l = 3 ):
    super().__init__( [ 
      h, w, l
    ] )

    self.num_nodes = h * w * l

    for dummy_i in range( self.num_nodes ):
        self.nodes.append( Node( 0 ) )
        
  def getData( self, pixels, mode = INPUT_EMBEDED_PIXELS ):

    if mode == INPUT_EMBEDED_PIXELS :
      for i in range( self.dimensions[0] * self.dimensions[1] ):
        for isub in range( self.dimensions[2] ):
          self.nodes[
            self.dimensions[2] * i + isub
          ].value = pixels[ i ][ isub ]  / 255 - 0.5
        
    elif mode == INPUT_MATRIX_PIXELS :
      
      for irow in range( self.dimensions[1] ):
        for icol in range( self.dimensions[0] ):
          for isub in range( self.dimensions[2] ):
            self.nodes[
              (
                irow * self.dimensions[1] + icol
              ) * self.dimensions[2] + isub
            ].value = pixels[ irow ][ icol ][ isub ] / 255 - 0.5
    elif mode == INPUT_MATRIX_ONE_CHANNEL:

      for irow in range( self.dimensions[1] ):
        for icol in range( self.dimensions[0] ):
          for isub in range( self.dimensions[2] ):
            self.nodes[
              (
                irow * self.dimensions[1] + icol
              ) * self.dimensions[2] + isub
            ].value = pixels[ irow ] [
              icol + isub
            ]/ 255 - 0.5