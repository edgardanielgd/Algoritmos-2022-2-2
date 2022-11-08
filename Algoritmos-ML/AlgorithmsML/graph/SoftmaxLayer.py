from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *
import math

class SoftmaxLayer ( Layer ):
  def __init__(self, flat_count ):
    super().__init__( [
      flat_count
    ] )

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
    
    exp_values = []
    exp_sum = 0

    for inode in range( len( self.nodes ) ):

      node = self.nodes[ inode ]
      
      dot_product = 0
      
      for link in node.prev_neighbors:
        
        dot_product += link.fromNode.value * link.weight.value

      dot_product += self.biases[ inode ]

      exp_value = math.exp( dot_product )
      
      exp_values.append( exp_value )

      exp_sum += exp_value
    
    for i in range( self.num_nodes ):

      prob_i = exp_values[i] / exp_sum
      self.nodes[i].value = prob_i

    # Useful for back propagation
    self.exp_values = exp_values
    self.exp_sum = exp_sum

  def backpropagate(self, label_index):
    # Label should be translated to an index
    prob_i = self.nodes[ label_index ].value

    gradient_label = -1 / prob_i
    gradients_data = []
    
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

      # Update biases
      self.biases [i] -= LEARNING_RATE * out_gradient
        
      gradients_data.append(
        out_gradient
      )

      # Gen input gradient (used by previous layer)

    input_gradient = []

    for iprev_node in range( self.prevLayer.num_nodes ):
      
      prev_node = self.prevLayer.nodes[ iprev_node ]
      input_gradient_item = 0
      
      for inode in range( self.num_nodes ):
        
        node = self.nodes[ inode ]
        node_gradient = gradients_data[ inode ]

        link_to_prev = node.prev_neighbors[ iprev_node ]

        # Update input gradient item data
        input_gradient_item += node_gradient * link_to_prev.weight.value
        
        # Update weight
        link_to_prev.weight.updateValue(
          link_to_prev.weight.value - 
            LEARNING_RATE * prev_node.value * node_gradient
        )

      input_gradient.append( input_gradient_item )

    return input_gradient
        
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
      
        
        
    
    

    