from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *
import math

class DenseLayer ( Layer ):
  def __init__(self, flat_count , lr = LEARNING_RATE):
    super().__init__( [
      flat_count
    ] )

    self.learning_rate = lr

    self.biases = []
    
  def onSublayerAdd( self, subLayer ):
    super().onSublayerAdd( subLayer )

    self.num_nodes = self.dimensions[0]

    for i in range( self.num_nodes ):
      node = Node()
      neighbors = [
        Link( 
          neigh_node,
          Weight(
            random() / subLayer.num_nodes 
          )
        ) for neigh_node in subLayer.nodes
      ]
    
      node.addPrevNeighbors( neighbors )
      
      self.nodes.append( node )
      self.biases.append( 0 )
      
  def calculate(self):
    
    for inode in range( self.num_nodes ):

      node = self.nodes[ inode ]
      
      dot_product = 0
      
      for link in node.prev_neighbors:
        
        dot_product += link.fromNode.value * link.weight.value

      dot_product += self.biases[ inode ]

      # Keeps values between -1 and 1
      node.value = dot_product / self.num_nodes

  def backpropagate(self, output_gradient, loss ):
    
    input_gradient = [
        0 for i in range( self.prevLayer.num_nodes )
    ]

    for i in range( self.num_nodes ):

      gradient = output_gradient [i]

      # Update biases
      self.biases [i] -= self.learning_rate * gradient

      node = self.nodes [ i ]
      for j in range( self.prevLayer.num_nodes ):
        
        link = node.prev_neighbors [j]

        # Update input gradient
        input_gradient [j] += gradient * link.weight.value

        # Update weight 
        
        link.weight.updateValue(
            link.weight.value - self.learning_rate * link.fromNode.value * gradient
        )

    return input_gradient, loss
        
  def saveData( self, filename ):
    with open( filename, "w+") as f:

      for inode in range( self.num_nodes ):
        node = self.nodes[ inode ]
        for prev_neighbor in node.prev_neighbors:
          f.write( 
            str(prev_neighbor.weight.value ) + "\n"
          )
        f.write( str(self.biases[ inode ] ) + "\n" )
        
  def loadData( self, filename ):
    with open( filename, "r") as f:
      lines = f.readlines()
      index = 0

      for inode in range( self.num_nodes ):
        node = self.nodes[ inode ]
        for prev_neighbor in node.prev_neighbors:
          value = float( lines[ index ] )
          index += 1
          prev_neighbor.weight.updateValue( value )
        
        value = float( lines[ index ] )
        index += 1
        self.biases[ inode ] = value
    