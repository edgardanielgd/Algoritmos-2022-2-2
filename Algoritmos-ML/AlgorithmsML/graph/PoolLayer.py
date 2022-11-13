from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

class PoolLayer ( Layer ):
  def __init__ (self, filterDims, maximize = True ):
    super().__init__( None )

    if len( filterDims ) == 2:
      self.filterDims = filterDims
    else:
      self.filterDims = [1,1]

    self.mode = maximize # True if maxPool, False if avgPool

    # Useful for backpropagation fast calculation
    self.propagation_log = []

  def onSublayerAdd( self, subLayer ):  
    super().onSublayerAdd( subLayer )
    sublayer_dim = subLayer.dimensions

    x_diff = sublayer_dim [0] // self.filterDims[0] 
    y_diff = sublayer_dim [1] // self.filterDims[1]

    self.dimensions = [ 
      x_diff,
      y_diff,
      sublayer_dim [2]
    ]

    self.num_nodes = x_diff * y_diff * sublayer_dim[2]

    sublayer_features = sublayer_dim [ 2 ]

    for y_offset in range( 0, sublayer_dim [1], self.filterDims[1] ):
      for x_offset in range( 0, sublayer_dim [0], self.filterDims[0] ):
        
        for isubfeat in range( sublayer_features ):
            
            node = Node( )

            for iter_y in range( y_offset, y_offset + self.filterDims[1] ):
              for iter_x in range( x_offset, x_offset + self.filterDims[0] ):

                link_to_prev_node = Link(

                  # Previous layer node
                  subLayer.nodes[
                    ( iter_y * sublayer_dim[0] + iter_x ) * sublayer_features + isubfeat
                  ]

                  # Weight isn't actually needed
                )
                
                node.addPrevNeighbors(
                  [
                    link_to_prev_node
                  ]
                )
            self.nodes.append( node )
  
  def calculate( self ):

    total_neighbors = self.filterDims[0] * self.filterDims[1]

    if self.mode: # Maximize Pool

      self.propagation_log = []

      for node in self.nodes:
        max = float("-inf")

        max_index = -1
        
        for ineighbor_link in range( len( node.prev_neighbors )):

          neighbor_link = node.prev_neighbors [ ineighbor_link ]
          
          cur_val = neighbor_link.fromNode.value

          if cur_val > max :
            max = cur_val
            max_index = ineighbor_link
            
        node.value = max

        # Caching the highest node value index
        self.propagation_log.append( max_index )
        
    else: # Avg Pool
      for node in self.nodes :
        avg = 0
        for neighbor_link in node.prev_neighbors:
          avg += neighbor_link.fromNode.value
  
        node.value = avg / total_neighbors

  def backpropagate( self, output_gradient, loss ):

    input_gradient = [
      0 for dummy_i in range( self.prevLayer.num_nodes )
    ]
    
    if self.mode :
      # Max Pooling
      for igradient in range( len( output_gradient )):
        gradient = output_gradient[ igradient ]
        node = self.nodes[ igradient ] # References the same node
        ineighbor = self.propagation_log[ igradient ]

        # igradient is the same than inode
        ifeature = igradient % self.dimensions[2]

        # Current layer node coords
        # Note it matches the upper-left value in sublayer
        # feature matrix (so this is the first neighbor)

        x = ((( 
          igradient - ifeature ) // self.dimensions[2]
          ) % self.dimensions[0] ) * self.filterDims[0] 
        y = ((( 
          igradient - ifeature ) // self.dimensions[2]
          ) // self.dimensions[0]  ) * self.filterDims[1]
        
        
        # Neighbor coords
        nx = x + ineighbor % self.filterDims[0]
        ny = y + ineighbor // self.filterDims[0]

        # Get neighbor index in sublayer's array
        neighbor_index = (
          self.prevLayer.dimensions[0] * ny + nx
        ) * self.dimensions[2] + ifeature
        
        input_gradient [neighbor_index] = gradient
          
    else:
      # Average Pooling

      filter_area = self.dimensions[0] * self.dimensions[1]
      
      for igradient in range( len( output_gradient )):
        gradient = output_gradient[ igradient ]
        node = self.nodes[ igradient ] # References the same node

        # igradient is the same than inode
        ifeature = igradient % self.dimensions[2]

        # Current layer node coords
        # Note it matches the upper-left value in sublayer
        # feature matrix (so this is the first neighbor)

        x = ((( 
          igradient - ifeature ) // self.dimensions[2]
          ) % self.dimensions[0] ) * self.filterDims[0] 
        y = ((( 
          igradient - ifeature ) // self.dimensions[2]
          ) // self.dimensions[0]  ) * self.filterDims[1]
        
        for ineighbor in range( len( node.prev_neighbors )):

          # Neighbor coords
          nx = x + ineighbor % self.filterDims[0]
          ny = y + ineighbor // self.filterDims[0]

          # Get neighbor index in sublayer's array

          neighbor_index = (
            self.prevLayer.dimensions[0] * ny + nx
          ) * self.dimensions[2] + ifeature
          
          input_gradient [neighbor_index] = gradient / filter_area
          
    return input_gradient, loss