from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

class ConvolutionalLayer ( Layer ):
  
  def __init__( self, filterDims, num_filters, filters = None ):
    super().__init__( None )

    self.num_filters = num_filters
    
    if len( filterDims ) == 2:
      self.filterDims = filterDims
    else:
      self.filterDims = [1,1]

    self.filters = []
    
    # Filters array with representation of weights, later they will be asociated to links
    # Filters will be flat arrays representing a matrix (so they are easier to access)

    filter_area = self.filterDims[0] * self.filterDims[1]
    
    for ifilter in range( num_filters ):

      filter = []
      
      for yfilter in range( self.filterDims[1] ):

        filter_row = []
        
        for xfilter in range( self.filterDims[0] ):

          new_weight = None

          if filters is not None:
            # If filters were passed, use theirs weight
            new_weight = Weight( filters [ ifilter ] [ xfilter ] [ yfilter ] )
          else:
            # Else gen a random weight
            new_weight = Weight( 
              random() / filter_area
            )
            
          filter_row.append( new_weight )

        filter.append( filter_row )
        
      self.filters.append( filter )

  def onSublayerAdd( self, subLayer ):

    super().onSublayerAdd( subLayer )
    sublayer_dim = subLayer.dimensions

    # Calculates final image dimensions
    x_diff = sublayer_dim [0] - self.filterDims[0] + 1
    y_diff = sublayer_dim [1] - self.filterDims[1] + 1

    # Updates resulting image dimensions
    self.dimensions = [ 
      x_diff,
      y_diff,
      self.num_filters * sublayer_dim [2]
    ]

    self.num_nodes = x_diff * y_diff * self.num_filters * sublayer_dim[2]

    sublayer_features = sublayer_dim [ 2 ]

    for y_offset in range( y_diff ):
      for x_offset in range( x_diff ):
        # Iterates through all possible filter overlapping
        
        for isubfeat in range( sublayer_features ):
          # For each feature (R, G or B channels) create a new feature in resulting image
          for ifilter in range( self.num_filters ):
            # Each filter should be related to its own channel or feature
            
            node = Node( )

            for iter_y_filter in range( self.filterDims[1] ):
              for iter_x_filter in range( self.filterDims[0] ):

                iter_y = iter_y_filter + y_offset
                iter_x = iter_x_filter + x_offset
                # Relate node to its filters-based value calculation neighbors
                
                weight = self.filters [ ifilter ] [ iter_x_filter ] [ iter_y_filter ]
                
                link_to_prev_node = Link( 
                  # Node from previous layer
                  subLayer.nodes[
                      ( iter_y * sublayer_dim[0] + iter_x ) * sublayer_features + isubfeat
                    ],

                  # Weight to be used, note there is a reference to this object in filters array
                  weight
                )
                
                node.addPrevNeighbors(
                  [
                    link_to_prev_node
                  ]
                )
            self.nodes.append( node )
  
  def calculate( self ):
    for node in self.nodes :
      dot_product = 0
      #print("NODE:\n")
      for neighbor_link in node.prev_neighbors:
        #print("FromNode:", neighbor_link.fromNode.value, neighbor_link.weight.value)
        dot_product += neighbor_link.fromNode.value * neighbor_link.weight.value

      node.value = dot_product
  
  def backpropagate( self, output_gradient ):

    input_gradient = [
      0 for dummy_id in range( self.prevLayer.num_nodes )
    ] # Init input gradient data
    
    # Copy filters
    filters_temp = []

    
    for filter in self.filters:
      
      filter_temp = []
      
      for row in filter:
        
        filter_row = []
        
        for weight in row:
          filter_row.append( weight.value )

        filter_temp.append( filter_row )

      filters_temp.append( filter_temp )

    # Update weights and input gradient data
    for inode in range( self.num_nodes ):
      node = self.nodes[ inode ]
      gradient = output_gradient[ inode ]

      if gradient != 0:
        # Optimizing a little bit complexity...

        # Filter associated to this node
        ifilter = inode % self.num_filters

        # Sublayer feature this node is associated to
        iprevFeature = inode % self.prevLayer.dimensions[2]
        
        # To which feature does this node belong to
        # Remember there are prevfeatures * filters features
        ifeature = inode % self.dimensions[2]

        # Note this matches with previous layer positonal coords
        # Current layer node coords
        x = (( 
          inode - ifeature ) // self.dimensions[2]
          ) % self.dimensions[0]
        y = (( 
          inode - ifeature ) // self.dimensions[2]
          ) // self.dimensions[0] 

        # Update filter values
        for yfilter in range( self.filterDims[1] ):
          for xfilter in range( self.filterDims[0] ):

            # Neighbor coords
            nx = x + xfilter
            ny = y + yfilter
    
            # Get neighbor index in sublayer's array
            neighbor_index = (
              self.prevLayer.dimensions[0] * ny + nx
            ) * self.prevLayer.dimensions[2] + iprevFeature

            # Update gradient entry
            input_gradient [neighbor_index] += gradient * filters_temp[
              ifilter
            ][
              xfilter
            ][
              yfilter
            ]
            
            associated_prev_value = node.prev_neighbors[
              yfilter * self.filterDims[0] + xfilter
            ].fromNode.value

            # Update filters
            # This is the main reason why we copied filters value at start
            current_filter_item_value = self.filters[
                ifilter 
              ][
                xfilter 
              ][
                yfilter 
              ].value
      
            self.filters[ ifilter ][ xfilter ][ yfilter ].updateValue(
              current_filter_item_value - 
                LEARNING_RATE * gradient * associated_prev_value
            )
    return input_gradient

  def saveData( self, filename ):
    with open( filename, "w+") as f:

      for filter in self.filters:
        for row in filter:
          for cell in row:
            f.write(
              str( cell.value ) + "\n"
            )
          
  def loadData( self, filename ):
    with open( filename, "r") as f:

      lines = f.readlines()
      index = 0
      for ifilter in range( self.num_filters ):
        for irow in range( self.filterDims[1] ):
          for icell in range( self.filterDims[0] ):
            value = float( lines[ index ] )
            index += 1
            self.filters[ ifilter ][ irow ][ icell ].updateValue(
              value
            )