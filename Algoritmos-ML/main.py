# from AlgorithmsML.models.mnist_fashion_model import *
# #rom AlgorithmsML.models.mnist_model import *
from AlgorithmsML.models.cifar10_model import *
# from AlgorithmsML.GUI.defaultGUI import *

"""
gui = GUInterface()
"""

"""
model = MNISTFashionModel()

print( model.train(
  1000,3,
  [
    "testConv3.txt",
    "testDense1.txt"
  ]
) )

for i in range( 1000, 2000, 10 ):
  print(
    model.testByIndex( i )
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
#"""
model = CIFARModel()
model.train(
  1000, 3,
  [
    "testconvcifar1.txt",
    "testconvcifar2.txt",
    "testdensecifar1.txt",
    "testdensecifar2.txt"
  ]
)

for i in range( 1000, 2000, 10 ):
  print(
    model.testByIndex( i )
  )
#"""