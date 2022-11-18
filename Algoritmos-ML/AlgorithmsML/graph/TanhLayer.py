from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

import math

class TanhLayer( Layer ):
  def __init__(self):
    super().__init__( 0 )
    
  def onSublayerAdd( self, subLayer ):
    super().onSublayerAdd( subLayer )
    self.dimensions = subLayer.dimensions
    self.num_nodes = subLayer.num_nodes
    self.prev_results = []
    
    for inode in range( self.num_nodes ):
      node = Node()
      node.addPrevNeighbors(
        [ Link( subLayer.nodes[ inode ] ) ]
      )
      self.nodes.append( node )
      
  def calculate( self ):
    self.prev_results = []

    for inode in range( self.num_nodes ):
      node = self.nodes[ inode ]
      neighbor = self.prevLayer.nodes[ inode ]
      node.value = math.tanh( neighbor.value )
      self.prev_results.append( node.value )

  def backpropagate( self, output_gradient, loss ):

    input_gradient = []

    for igradient in range( len( output_gradient ) ):
      
      input_gradient.append(
        ( 1 - self.prev_results[ igradient ] ** 2 ) * output_gradient[ igradient ]
      )
        
    return input_gradient, loss