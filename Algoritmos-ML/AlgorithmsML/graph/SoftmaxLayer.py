from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *
import math

class SoftmaxLayer ( Layer ):
  def __init__(self ):
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
      
  def calculate(self):
    
    exp_values = []
    exp_sum = 0

    for inode in range( self.num_nodes ):

      node = self.prevLayer.nodes[ inode ]

      exp_value = math.exp( node.value )
      
      exp_values.append( exp_value )

      exp_sum += exp_value
    
    for i in range( self.num_nodes ):

      prob_i = exp_values[i] / exp_sum
      self.nodes[i].value = prob_i

    # Useful for back propagation
    self.exp_values = exp_values
    self.exp_sum = exp_sum

  def backpropagate(self, label_index, _):
    # Label should be translated to an index
    prob_i = self.nodes[ label_index ].value

    gradient_label = -1 / prob_i
    input_gradient = []
    
    for i in range( self.num_nodes ):

      out_gradient = 0
      
      if i == label_index:

        out_gradient = self.exp_values[
          label_index
        ] * (
          self.exp_sum - self.exp_values[
            label_index
          ] ) / (self.exp_sum ** 2)

      else:
        out_gradient = -self.exp_values[
          label_index
        ] * self.exp_values [
          i 
        ] / (self.exp_sum ** 2)
        
      out_gradient *= gradient_label
        
      input_gradient.append(
        out_gradient
      )
    return input_gradient, prob_i