import AlgorithmsML.figures as fg

from AlgorithmsML.graph.ConvolutionLayer import *
from AlgorithmsML.graph.InputLayer import *
from AlgorithmsML.graph.PoolLayer import *
from AlgorithmsML.graph.ReluLayer import *
from AlgorithmsML.graph.SoftmaxLayer import *
from AlgorithmsML.graph.DenseLayer import *
from AlgorithmsML.graph.TanhLayer import *
from AlgorithmsML.graph.Base import *

from keras.datasets import cifar10 # Set de numeros 28 x 28 pixeles

from random import shuffle
# from keras.dataset import fashion_mnist # Set de fashion

# (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

class CIFARModel:

  def __init__(self, lr = LEARNING_RATE ):
    (self.images, self.labels ), (self.testImages, self.testLabels) = cifar10.load_data()
    
    self.inputLayer = InputLayer( 32, 32, 3 ) # 32x32x3
    
    self.convLayer = ConvolutionalLayer2( [5,5], 32, None, FILTER_SUMMARIZE, lr, SAME_SIZE_PADDING )
    self.inputLayer.addSuperLayer( self.convLayer ) # 32x32x32

    self.maxPooling = PoolLayer( [2,2] ) # MaxPool
    self.convLayer.addSuperLayer( self.maxPooling ) # 16x16x24

    self.convLayer2 = ConvolutionalLayer2( [5,5], 1, None, FILTER_PER_FEATURE, lr, SAME_SIZE_PADDING )
    self.maxPooling.addSuperLayer( self.convLayer2 ) # 16x16x32

    self.maxPooling2 = PoolLayer( [2,2] ) # MaxPool
    self.convLayer2.addSuperLayer( self.maxPooling2 ) # 8x8x32

    self.convLayer3 = ConvolutionalLayer2( [5,5], 2, None, FILTER_PER_FEATURE, lr, SAME_SIZE_PADDING )
    self.maxPooling2.addSuperLayer( self.convLayer3 ) # 8x8x64

    self.maxPooling3 = PoolLayer( [2,2] ) # MaxPool
    self.convLayer3.addSuperLayer( self.maxPooling3 ) # 4x4x64

    self.dense1 = DenseLayer( 64, lr ) 
    self.maxPooling3.addSuperLayer( self.dense1 ) # 64

    self.tanh = TanhLayer() # ReLU Layer
    self.dense1.addSuperLayer( self.tanh ) # 64

    self.dense2 = DenseLayer( 10, lr )
    self.relu2.addSuperLayer( self.dense2 ) # 10

    self.softmax = SoftmaxLayer( ) # Softmax Layer
    self.dense2.addSuperLayer( self.softmax ) # 10
  
  
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
          INPUT_MATRIX_PIXELS
        )
        epoch_loss += self.inputLayer.passDataRecursive( trainLabels[i][0] )[1]
        epoch_acc += 1 if ( self.orderWinners()[0][0] == trainLabels[i][0] ) else 0

      acc.append( epoch_acc / ntrains )
      loss.append( epoch_loss / ntrains )

      print( "Ended Epoch ", epoch_i)
      
    self.convLayer.saveData(
      savFiles[0]
    )
    self.convLayer2.saveData(
      savFiles[1]
    )
    self.dense1.saveData(
      savFiles[2]
    )
    self.dense2.saveData(
      savFiles[3]
    )

    return loss, acc

  def load( self, loadFiles ):
    self.convLayer.loadData(
      loadFiles[0]
    )
    self.convLayer2.loadData(
      loadFiles[1]
    )
    self.dense1.loadData(
      loadFiles[2]
    )
    self.dense2.loadData(
      loadFiles[3]
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