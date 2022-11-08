from random import random

LEARNING_RATE = 0.005

class Weight:
  # Useful optimizer for links dynamic weight change
  def __init__(self, value):
    self.updateValue( value )
    
  def updateValue( self, value):
    self.value = value

class Link:
  # Link from a sublayer to a superlayer
  # IMPORTANT: Note the destination node will reference the source
  # node, this results convenient for layer's nodes values calculations
  # based on previous layers
  
  def __init__(self, fromNode, value = None ):

    self.weight = None
    
    if value is None:
      self.weight = Weight( random() )
    else:
      self.weight = value
    
    self.fromNode = fromNode
  
class Node:
  def __init__(self, value = random() ):
    self.value = value
    # Actually nodes are aiming to this one (in-links)
    self.prev_neighbors = []
    
  def addPrevNeighbors( self, links ):
    # Adds links to back - neighbors
    for link in links:
      self.prev_neighbors.append( link )
      