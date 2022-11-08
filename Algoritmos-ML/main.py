from AlgorithmsML.models.mnist_fashion_model import *
from AlgorithmsML.models.mnist_model import *

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
#from models import mnist_model