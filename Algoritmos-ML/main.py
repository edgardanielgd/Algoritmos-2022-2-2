#from AlgorithmsML.models.mnist_fashion_model import *
#from AlgorithmsML.models.mnist_model import *
from AlgorithmsML.models.cifar10_model import *
#from AlgorithmsML.GUI.defaultGUI import *

"""
gui = GUInterface(
  28, 28
)
"""

"""
model = MNISTModel()

model.load(
  [
    "AlgorithmsML/convolutional2.txt",
    "AlgorithmsML/softmax2.txt"
  ]
)

print(
  model.testByIndex( 1057 )
)
"""

"""
model = MNISTFashionModel()
model.load(
  [
    "testconv1.txt",
    "testsoft1.txt"
  ]
)

print(
  model.testByIndex(
    1007
  )
)
"""

model = CIFARModel()
model.train(
  10, 3,
  [
    "testconv12.txt",
    "testconv122.txt",
    "testsoft2.txt"
  ]
)

print(
  model.testByIndex(
    1008
  )
)