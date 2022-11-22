import AlgorithmsML.figures as fg

from AlgorithmsML.graph.ConvolutionLayer import *
from AlgorithmsML.graph.InputLayer import *
from AlgorithmsML.graph.PoolLayer import *
from AlgorithmsML.graph.ReluLayer import *
from AlgorithmsML.graph.SoftmaxLayer import *
from AlgorithmsML.graph.DenseLayer import *
from AlgorithmsML.graph.TanhLayer import *
from AlgorithmsML.graph.Base import *

from keras.datasets import fashion_mnist # Set de numeros 28 x 28 pixeles

# 0: T-shirt/Top
# 1: Trouser
# 2: Pullover
# 3: Dress
# 4: Coat
# 5: Sandals
# 6: Shirt
# 7: Sneaker
# 8: Bag
# 9: Ankle boots

from random import shuffle
# from keras.dataset import fashion_mnist # Set de fashion

# (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

class MNISTFashionModel:

  def __init__(self, lr = LEARNING_RATE ):
    (self.images, self.labels ), (self.testImages, self.testLabels) = fashion_mnist.load_data()
    
    self.inputLayer = InputLayer( 28, 28, 1 )
    
    self.convLayer = ConvolutionalLayer2( [3,3], 8, None, FILTER_GLOBAL, lr)
    self.inputLayer.addSuperLayer( self.convLayer )
    
    self.maxPooling = PoolLayer( [2,2] ) # MaxPool
    self.convLayer.addSuperLayer( self.maxPooling )

    self.dense = DenseLayer(10, lr) # Dense Layer
    self.maxPooling.addSuperLayer( self.dense )

    self.softmax = SoftmaxLayer( ) # Softmax Layer
    self.dense.addSuperLayer( self.softmax )
  
  
  def train( self, ntrains, nepochs, savFiles ):

    trainImages = self.images[:ntrains]
    trainLabels = self.labels[:ntrains]
  
    indexes = [ index for index in range( ntrains ) ]
    
    loss = []
    acc = []

    for epoch_i in range( nepochs ):
  
      shuffle( indexes )  
      
      epoch_loss = 0
      epoch_acc = 0

      for i in indexes:
        self.inputLayer.getData( 
          trainImages[i],
          INPUT_MATRIX_ONE_CHANNEL
        )

        epoch_loss += self.inputLayer.passDataRecursive( trainLabels[i] ) [1]
        epoch_acc += 1 if ( self.orderWinners()[0][0] == trainLabels[i] ) else 0

      acc.append( epoch_acc / ntrains )
      loss.append( epoch_loss / ntrains )

      print( "Ended Epoch ", epoch_i, epoch_acc / ntrains, epoch_loss / ntrains )
      
    self.convLayer.saveData(
      savFiles[0]
    )
    self.dense.saveData(
      savFiles[1]
    )

    return loss, acc

  def load( self, loadFiles ):
    self.convLayer.loadData(
      loadFiles[0]
    )
    
    self.dense.loadData(
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