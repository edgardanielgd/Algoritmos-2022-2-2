import AlgorithmsML.figures as fg

from AlgorithmsML.graph.ConvolutionLayer import *
from AlgorithmsML.graph.InputLayer import *
from AlgorithmsML.graph.Base import *

class ConvModel:

  def __init__(self, pfilter = [[
    [ 0, 0, 0],
    [ 0, 1, 0],
    [ 0, 0, 0]
  ]], height = 100, width = 100 ):
    
    self.inputLayer = InputLayer( height, width, 3 )
    
    filter = pfilter[0] if pfilter is not None else []
    filter_height = len( filter ) if filter is not None else 0
    filter_width = len( filter [0]) if filter is not None and len( filter ) > 0 else 0

    self.convLayer = ConvolutionalLayer2( 
      [ filter_height, filter_width ] if filter is not None and len(filter) > 0 else [],
      1, pfilter, FILTER_GLOBAL, LEARNING_RATE, SAME_SIZE_PADDING )
    self.inputLayer.addSuperLayer( self.convLayer )
    

  def testByData( self, image_data ):
    
    self.inputLayer.getData( 
      image_data,
      INPUT_EMBEDED_PIXELS,
      False
    )
    self.inputLayer.passDataRecursive( )

    return self.convLayer.toImage()

    