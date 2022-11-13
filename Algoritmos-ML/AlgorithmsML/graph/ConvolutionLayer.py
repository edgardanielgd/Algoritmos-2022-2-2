from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

FILTER_GLOBAL = 1 # If each of given filter(s) should be applied to each sublayer's feature
FILTER_PER_FEATURE = 2 # If there is a different filter for each feature
FILTER_SUMMARIZE = 3 # Calculate different filters per feature and summarize them at the end

DEFAULT_PADDING = 1 # Filters will reduce main features dimensions
SAME_SIZE_PADDING = 2 # Filters will keep the same size of previous layer (padding)
CUSTOM_PADDING = 3 # User will give a custom padding dimensions

class ConvolutionalLayer2 ( Layer ):
  
  def __init__( self, 
      filterDims, num_filters, filters = None, 
      filter_mode = FILTER_GLOBAL, lr = LEARNING_RATE,
      padding_mode = DEFAULT_PADDING, padding = None ):
    
    super().__init__( None )

    self.learning_rate = lr

    self.num_filters = num_filters
    
    if len( filterDims ) == 2:
      self.filterDims = filterDims
    else:
      self.filterDims = [1,1]

    self.raw_filters = filters
    self.filters = []
    self.filter_mode = filter_mode
    self.padding_mode = padding_mode

    self.padding = [ 0, 0, 0, 0] # Left, right, top and bottom padding
    if self.padding_mode == SAME_SIZE_PADDING:
      self.padding = [
        self.filterDims[0] // 2, # Left padding
        self.filterDims[0] // 2 if self.filterDims[0] % 2 == 1 else self.filterDims[0] // 2 - 1, 
        # Right padding (ouput has same width than input)
        self.filterDims[1] // 2, # Top padding
        self.filterDims[1] // 2 if self.filterDims[1] % 2 == 1 else self.filterDims[1] // 2 - 1, 
        # Bottom padding (ouput has same width than input)
      ]
    elif self.padding_mode == CUSTOM_PADDING and self.padding is not None:
      self.padding = padding
    
    if filter_mode == FILTER_GLOBAL:
      # Filters can be created right now
      # Otherwise they should be created while adding sublayer
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

    # Calculates final image dimensions (Without counting padding)
    x_diff = sublayer_dim [0] - self.filterDims[0] + 1
    y_diff = sublayer_dim [1] - self.filterDims[1] + 1

    features = 0

    if self.filter_mode == FILTER_SUMMARIZE:
      features = self.num_filters
    else:
      features = self.num_filters * sublayer_dim [2]
    
    # Updates resulting image dimensions
    self.dimensions = [ 
      x_diff + self.padding[0] + self.padding[1],
      y_diff + self.padding[2] + self.padding[3],
      features
    ]

    self.num_nodes = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    sublayer_features = sublayer_dim [ 2 ]

    # Iterators useful for filter association with nodes
    filter_origin = lambda x: 0

    if self.filter_mode == FILTER_PER_FEATURE or self.filter_mode == FILTER_SUMMARIZE:
      # One different filter per feature and per filter type

      # First than all, filters should be created

      filter_area = 1
      if self.filter_mode == FILTER_PER_FEATURE:
        filter_area = self.filterDims[0] * self.filterDims[1]
      else:
        filter_area = self.filterDims[0] * self.filterDims[1] * sublayer_features
      
      for isubfeat in range( sublayer_features ):
        # i.e. filters = [ Sub1F1, Sub1F2, Sub2F1, Sub2F2, .. ]
        for ifilter in range( self.num_filters ):
          # Since there will be a different updatable filter per sublayer feature
    
          filter = []
          
          for yfilter in range( self.filterDims[1] ):
    
            filter_row = []
            
            for xfilter in range( self.filterDims[0] ):
    
              new_weight = None
    
              if self.raw_filters is not None:
                # If filters were passed, use theirs weight
    
                new_weight = Weight( self.raw_filters [
                  ifilter * sublayer_features + isubfeat
                ] [ xfilter ] [ yfilter ] )
                  
              else:
                # Else gen a random weight
                
                new_weight = Weight( 
                  random() / filter_area
                )
                
              filter_row.append( new_weight )
    
            filter.append( filter_row )
            
          self.filters.append( filter )

      # gets ( ifilter, isubfeat )
      filter_origin = lambda x : ( x[0] * sublayer_features) + x[1]
    else:
      # gets ( ifilter, isubfeat )
      filter_origin = lambda x : x[0]
    
    # Relate nodes with their previous layer neighbors

    def relate_with_neighbors():

      # Note all variables will be declared when needed
      for iter_y_filter in range( self.filterDims[1] ):
        for iter_x_filter in range( self.filterDims[0] ):

          iter_y = iter_y_filter + y_offset
          iter_x = iter_x_filter + x_offset

          if iter_x < 0 or iter_x >= sublayer_dim[0] or iter_y < 0 or iter_y >= sublayer_dim[1]:

            # Ignore this neighbor value (result of padding)
            continue


          # Relate node to its filters-based value calculation neighbors
          weight = self.filters [ 
            filter_origin(
              ( ifilter, isubfeat )
            ) # Gets filter info
          ] [ iter_x_filter ] [ iter_y_filter ]
          
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

    for y_offset in range( - self.padding[2], y_diff + self.padding[3] ):
      for x_offset in range( - self.padding[0], x_diff + self.padding[1] ):
        # Iterates through all possible filter overlapping
        
        if self.filter_mode != FILTER_SUMMARIZE:
          # Differences between number of nodes and correct relation between both ways of filters

          for isubfeat in range( sublayer_features ):
            # For each feature (R, G or B channels) create a new feature in resulting image
            for ifilter in range( self.num_filters ):
              # number of filters * number of sublayer features
                
                node = Node( )
                
                relate_with_neighbors()
                
                self.nodes.append( node )
        else:

          # Filter summarization needs to be calculated in a different way

          for ifilter in range( self.num_filters ):
            # Each filter will be related with a feature, but each node won't
            node = Node( )

            for isubfeat in range( sublayer_features ):
              # For each feature (R, G or B channels) create a new feature in resulting image
              
              relate_with_neighbors()

            self.nodes.append( node )
  
  def calculate( self ):
    for node in self.nodes :
      dot_product = 0
      #print("NODE:\n")
      for neighbor_link in node.prev_neighbors:
        #print("FromNode:", neighbor_link.fromNode.value, neighbor_link.weight.value)
        dot_product += neighbor_link.fromNode.value * neighbor_link.weight.value

      node.value = dot_product
  
  def backpropagate( self, output_gradient, loss ):

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

    def propagate_by_neighbor():

      # Propagate filter and input gradients

      neighbor_relative_pos = 0 # Neighbor index in node's neighbors array

      for yfilter in range( self.filterDims[1] ):
        for xfilter in range( self.filterDims[0] ):

          # Neighbor coords
          nx = x + xfilter
          ny = y + yfilter

          if nx < 0 or nx >= self.prevLayer.dimensions[0] or ny < 0 or ny >= self.prevLayer.dimensions[1]:
            # Skip positions where filter was not applied due to padding
            continue

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
            neighbor_relative_pos
          ].fromNode.value

          # Update neighbor offset in array
          neighbor_relative_pos += 1

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
              self.learning_rate * gradient * associated_prev_value
          )

    # Update weights and input gradient data
    for inode in range( self.num_nodes ):
      node = self.nodes[ inode ]
      gradient = output_gradient[ inode ]

      if gradient != 0:
        # Optimizing a little bit complexity...

        if self.filter_mode != FILTER_SUMMARIZE:
          ifilter = 0
          
          # Filter associated to this node
          if self.filter_mode == FILTER_GLOBAL:
            ifilter = inode % self.num_filters
          elif self.filter_mode == FILTER_PER_FEATURE:
            ifilter = inode % self.dimensions[2]

          # Sublayer feature this node is associated to
          iprevFeature = ( inode // self.prevLayer.dimensions[2] ) % self.prevLayer.dimensions[2]
          
          # To which feature does this node belong to
          # Remember there are prevfeatures * filters features
          ifeature = inode % self.dimensions[2]

          # Note this matches with previous layer positonal coords
          # Current layer node coords
          x = (( 
            inode - ifeature ) // self.dimensions[2]
            ) % self.dimensions[0] - self.padding[0]
          y = (( 
            inode - ifeature ) // self.dimensions[2]
            ) // self.dimensions[0] - self.padding[2]

          # Update filter values

          propagate_by_neighbor()
        else:
          # To which feature does this node belong to
          # Remember there are nfilters features
          ifeature = inode % self.dimensions[2] # (In this case, ifeature = associated filters cluster id)

          # Note this matches with previous layer positonal coords
          # Current layer node coords
          x = (( 
            inode - ifeature ) // self.dimensions[2]
            ) % self.dimensions[0] - self.padding[0]
          y = (( 
            inode - ifeature ) // self.dimensions[2]
            ) // self.dimensions[0] - self.padding[2]

          for ifilter in range( self.num_filters ):
            for iprevFeature in range( self.prevLayer.dimensions[2] ):
              
              # Update filter values

              propagate_by_neighbor()

    return input_gradient, loss

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