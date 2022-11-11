from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

class ReluLayer( Layer ):
  def __init__(self):
    super().__init__( 0 )
    
  def onSublayerAdd( self, subLayer ):
    super().onSublayerAdd( subLayer )
    self.dimensions = subLayer.dimensions
    self.num_nodes = subLayer.num_nodes
    
    for inode in range( self.num_nodes ):
      node = Node()
      node.addPrevNeighbors(
        [ Link( subLayer.nodes[ inode ] ) ]
      )
      self.nodes.append( node )
      
  def calculate( self ):
    for inode in range( self.num_nodes ):
      node = self.nodes[ inode ]
      neighbor = self.prevLayer.nodes[ inode ]
      node.value = max( neighbor.value, 0)

  def backpropagate( self, output_gradient, loss ):

    input_gradient = []

    for igradient in range( len( output_gradient ) ):
      
      if self.nodes[ igradient ].value > 0:
        input_gradient.append(
          output_gradient[ igradient ]
        )
      else:
        input_gradient.append( 0 )
        
    return input_gradient, loss