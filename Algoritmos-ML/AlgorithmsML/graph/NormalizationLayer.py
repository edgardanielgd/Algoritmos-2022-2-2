from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

from random import random

import math

class TanhLayer( Layer ):
  def __init__(self):
    super().__init__( 0 )
    
  def onSublayerAdd( self, subLayer ):
    super().onSublayerAdd( subLayer )
    self.dimensions = subLayer.dimensions
    self.num_nodes = subLayer.num_nodes

    self.means_differences = []
    self.subnormalized_values = []
    self.normalized_values = []

    self.mean = 0
    self.sd = 0

    self.gamma = random()
    self.beta = random()
    
    for inode in range( self.num_nodes ):
      node = Node()
      node.addPrevNeighbors(
        [ Link( subLayer.nodes[ inode ] ) ]
      )
      self.nodes.append( node )
      
  def calculate( self ):

    self.means_differences = []
    self.subnormalized_values = []
    self.normalized_values = []

    self.mean = 0
    self.sd = 0

    for inode in range( self.num_nodes ):
      node = self.nodes[ inode ]
      self.mean += node.value
    
    self.mean /= self.num_nodes

    for inode in range( self.num_nodes ):
      node = self.nodes[ inode ]
      value = node.value

      mean_difference = value - self.mean
      self.means_differences.append( mean_difference )
      pow_mean_difference = mean_difference ** 2

      self.sd += pow_mean_difference
    
    self.sd = math.sqrt( self.d / self.num_nodes )

    for inode in range( self.num_nodes ):
      subnormalized = self.means_differences [ inode ] / self.sd
      self.subnormalized_values.append( subnormalized )

      normalized = subnormalized * self.gamma + self.beta
      self.nodes[ inode ].value = normalized
      self.normalized.append( normalized )

  def backpropagate( self, output_gradient, loss ):

    input_gradient = []

    for igradient in range( len( output_gradient ) ):
      
      input_gradient.append(
        ( 1 - self.prev_results[ igradient ] ** 2 ) * output_gradient[ igradient ]
      )
        
    return input_gradient, loss