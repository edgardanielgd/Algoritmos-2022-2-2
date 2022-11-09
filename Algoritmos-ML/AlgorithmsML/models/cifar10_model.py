import AlgorithmsML.figures as fg

from AlgorithmsML.graph.ConvolutionLayer2 import *
from AlgorithmsML.graph.InputLayer import *
from AlgorithmsML.graph.PoolLayer import *
from AlgorithmsML.graph.ReluLayer import *
from AlgorithmsML.graph.SoftmaxLayer import *

from keras.datasets import cifar10 # Set de numeros 28 x 28 pixeles

from random import shuffle
# from keras.dataset import fashion_mnist # Set de fashion

# (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

class CIFARModel:

  def __init__(self):
    (self.images, self.labels ), (self.testImages, self.testLabels) = cifar10.load_data()
    
    self.inputLayer = InputLayer( 32, 32, 1 )
    
    self.convLayer = ConvolutionalLayer2( [3,3], 32, None, FILTER_PER_FEATURE )
    self.inputLayer.addSuperLayer( self.convLayer )

    self.maxPooling = PoolLayer( [2,2] ) # MaxPool
    self.convLayer.addSuperLayer( self.maxPooling )

    self.relu = ReluLayer() # ReLU Layer
    self.maxPooling.addSuperLayer( self.relu )

    self.convLayer2 = ConvolutionalLayer2( [3,3], 32, None, FILTER_PER_FEATURE )
    self.relu.addSuperLayer( self.convLayer2 )

    self.maxPooling2 = PoolLayer( [2,2] ) # MaxPool
    self.convLayer2.addSuperLayer( self.maxPooling2 )

    self.relu2 = ReluLayer() # ReLU Layer
    self.maxPooling2.addSuperLayer( self.relu2 )

    self.softmax = SoftmaxLayer( 10 ) # Softmax Layer
    self.maxPooling2.addSuperLayer( self.softmax )
  
  
  def train( self, ntrains, nepochs, savFiles ):

    trainImages = self.images[:ntrains]
    
    trainLabels = self.labels[:ntrains]
  
    indexes = [ index for index in range( ntrains ) ]
    
    for epoch_i in range( nepochs ):
  
      shuffle( indexes )  
      
      for i in indexes:
        self.inputLayer.getData( 
          trainImages[i],
          INPUT_MATRIX_PIXELS
        )
        
        # For any reason, labels come in as arrays of size 1...
        self.inputLayer.passDataRecursive( trainLabels[i][0] )
  
      print( "Ended Epoch ", epoch_i)
      
    self.convLayer.saveData(
      savFiles[0]
    )
    self.convLayer2.saveData(
      savFiles[1]
    )
    self.softmax.saveData(
      savFiles[2]
    )

  def load( self, loadFiles ):
    self.convLayer.loadData(
      loadFiles[0]
    )
    self.convLayer.loadData(
      loadFiles[1]
    )
    self.softmax.loadData(
      loadFiles[2]
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
      INPUT_MATRIX_PIXELS
    )
    self.inputLayer.passDataRecursive( )
    return label, self.orderWinners()

  def testByData( self, data ):
    self.inputLayer.getData( 
      data,
      INPUT_MATRIX_ONE_CHANNEL_PER_FEATURE
    )
    self.inputLayer.passDataRecursive( )
    return self.orderWinners()