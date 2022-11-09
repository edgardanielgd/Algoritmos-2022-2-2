import AlgorithmsML.figures as fg

from AlgorithmsML.graph.ConvolutionLayer2 import *
from AlgorithmsML.graph.InputLayer import *
from AlgorithmsML.graph.PoolLayer import *
from AlgorithmsML.graph.ReluLayer import *
from AlgorithmsML.graph.SoftmaxLayer import *

from keras.datasets import fashion_mnist # Set de numeros 28 x 28 pixeles

from random import shuffle
# from keras.dataset import fashion_mnist # Set de fashion

# (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

class MNISTFashionModel:

  def __init__(self):
    (self.images, self.labels ), (self.testImages, self.testLabels) = fashion_mnist.load_data()
    
    self.inputLayer = InputLayer( 28, 28, 1 )
    
    self.convLayer = ConvolutionalLayer2( [3,3], 8 )
    self.inputLayer.addSuperLayer( self.convLayer )
    
    self.maxPooling = PoolLayer( [2,2] ) # MaxPool
    self.convLayer.addSuperLayer( self.maxPooling )
    
    self.relu = ReluLayer() # ReLU Layer
    self.maxPooling.addSuperLayer( self.relu )
  
    self.softmax = SoftmaxLayer( 10 ) # Softmax Layer
    self.maxPooling.addSuperLayer( self.softmax )
  
  
  def train( self, ntrains, nepochs, savFiles ):

    trainImages = self.images[:ntrains]
    trainLabels = self.labels[:ntrains]
  
    indexes = [ index for index in range( ntrains ) ]
    
    for epoch_i in range( nepochs ):
  
      shuffle( indexes )  
      
      for i in indexes:
        self.inputLayer.getData( 
          trainImages[i],
          INPUT_MATRIX_ONE_CHANNEL
        )
        self.inputLayer.passDataRecursive( trainLabels[i] )
  
      print( "Ended Epoch ", epoch_i)
      
    self.convLayer.saveData(
      savFiles[0]
    )
    self.softmax.saveData(
      savFiles[1]
    )

  def load( self, loadFiles ):
    self.convLayer.loadData(
      loadFiles[0]
    )
    
    self.softmax.loadData(
      loadFiles[1]
    )

  def orderWinners( self ):
    winners = []

    for i in range( self.softmax.num_nodes ):
      winners.append(
        (i, self.softmax.nodes[ i ].value )
      )

    winners.sort(
      key = lambda x : x[1],
      reverse = True
    )
    
    return winners

  def testByIndex( self, index ):
    data = self.testImages[ index ]
    label = self.testLabels[ index]
    self.inputLayer.getData( 
      data,
      INPUT_MATRIX_ONE_CHANNEL
    )
    self.inputLayer.passDataRecursive( )
    return label, self.orderWinners()

  def testByData( self, data ):
    self.inputLayer.getData( 
      data,
      INPUT_MATRIX_ONE_CHANNEL
    )
    self.inputLayer.passDataRecursive( )
    return self.orderWinners()