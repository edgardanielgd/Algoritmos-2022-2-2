from AlgorithmsML.graph.Base import *
from AlgorithmsML.graph.Layer import *

from random import random

import math

class NormLayer( Layer ):
  def __init__(self, lr = LEARNING_RATE ):
    super().__init__( 0 )
    self.learning_rate = lr
    
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

      gradient = output_gradient[ igradient ]
      
      # Update beta value
      self.beta -= self.learning_rate * gradient

      # Update (send) input_gradient
      in_gradient = self.gamma / (self.num_nodes ** (3/2) * self.sd ** 2)

      numerator = ( self.num_nodes - 1) * self.sd ** 2 * self.num_nodes ** (1/2)

      thisDifference = self.means_differences[ igradient ]
      
      numerator -= 2 * thisDifference ** 2 * ( self.num_nodes - 1 )

      for idifference_m in self.means_differences:
        numerator += 2 * idifference_m * thisDifference
      
      in_gradient *= numerator

      input_gradient.append( in_gradient )

      # Update gamma value
      self.gamma -= self.learning_rate * self.normalized_values[ igradient ] * gradient
    
    return input_gradient, loss