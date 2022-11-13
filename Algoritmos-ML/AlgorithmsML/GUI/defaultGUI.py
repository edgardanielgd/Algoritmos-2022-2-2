from tkinter import *
from tkinter import filedialog, ttk

from tkstylesheet import TkThemeLoader

from PIL import ImageTk, Image

from AlgorithmsML.models.mnist_fashion_model import *
from AlgorithmsML.models.mnist_model import *
from AlgorithmsML.models.cifar10_model import *
from AlgorithmsML.models.conv_model import *

import AlgorithmsML.figures as fg

def rgbtohex(r,g,b):
  return f'#{r:02x}{g:02x}{b:02x}'

MNIST_MODE = 1
MNIST_FASHION_MODE = 2
CIFAR10_MODE = 3
IMAGE_CONV_MODE = 4

class GUInterface:
    
  def __init__( self, width = 28, height = 28, path_dir = "./models_data/"):

    self.style = """
        Tk{
            background: "#565657"; /*background color of the root widget*/
        }

        Label{
            foreground: "#ebebeb";
            background: "#565657";
        }

        Button{
            foreground: "#ebebeb";
            background: "#565657";
        }

        """

    # Graphical config:
    self.mode = MNIST_MODE

    self.canv_width = 400
    self.canv_height = 400

    self.width = width
    self.height = height
    
    self.rect_width = self.canv_width // width
    self.rect_height = self.canv_height // height

    self.top = Tk()
    self.top.geometry(
      str( self.canv_width + 150 ) + "x" + str( self.canv_height + 50 )
    )

    self.canvas = Canvas(
      self.top, 
      height = self.canv_height, 
      width = self.canv_width,
      bg = rgbtohex(0,0,0)
    )
  
    self.canvas.place(
      x = 0, y = 0
    )
  
    self.canvas.bind(
      "<B1-Motion>",
      lambda event: self.canvasClick( event )
    )
  
    self.scale1 = Scale(
      self.top, from_=0, to=255,
      orient=HORIZONTAL
    )
  
    self.scale1.place(
      x = 0, y = self.canv_height  
    )

    self.scale2 = Scale(
      self.top, from_=0, to=255,
      orient=HORIZONTAL
    )
    
    self.scale3 = Scale(
      self.top, from_=0, to=255,
      orient=HORIZONTAL
    )
  
    self.action = Button(
      self.top,
      text = "Clasificar"
    )

    self.action.place(
      x = self.canv_width + 20 , y = 100
    )
    self.action.bind(
      "<Button-1>",
      lambda event : self.actionClick( event )
    )

    self.menubar = Menu( self.top )
    self.typeMenu = Menu( self.menubar, tearoff = 0)

    self.typeMenu.add_command(
      label = "Mnist", command = lambda : self.updateGUI( MNIST_MODE )
    )

    self.typeMenu.add_command(
      label = "Mnist fashion", command = lambda : self.updateGUI( MNIST_FASHION_MODE )
    )

    self.typeMenu.add_command(
      label = "Cifar 10", command = lambda : self.updateGUI( CIFAR10_MODE )
    )

    self.typeMenu.add_separator()

    self.typeMenu.add_command(
      label = "Image Convolution", command = lambda : self.updateGUI( IMAGE_CONV_MODE )
    )

    self.menubar.add_cascade( 
      label = "Tipo",
      menu = self.typeMenu
    )

    self.top.config( menu = self.menubar )

    # Draw squares
    self.rectangles = []
    self.values = []

    for i in range( self.height ):

      row = []
      values_row = []
      
      for j in range( self.width ):
        row.append(
          self.canvas.create_rectangle(
            i * self.rect_width,
            j * self.rect_height,
            i * self.rect_width + self.rect_width,
            j * self.rect_height + self.rect_height,
            fill = rgbtohex( 0, 0, 0)
          )
        )

        values_row.append(
          ( 0, 0, 0 )
        )

      self.values.append(
        values_row
      )

      self.rectangles.append(
        row
      )
    
    # Image control
    self.image = None

    self.filter_box = ttk.Combobox(
      state = "readonly",
      values = [
        "Identidad", "Bordes", "Desenfoque"
      ]
    )

    self.filter_box.current(0)
    
    print("GUI started, loading models....")
    # Models config
    self.mnist_model = MNISTModel()
    self.mnist_fashion_model = MNISTFashionModel()

    self.mnist_model.load(
      [
        path_dir + "MNISTConv.txt",
        path_dir + "MNISTDense.txt"
      ]
    )

    self.mnist_fashion_model.load(
      [
        path_dir + "MNISTFashionConv.txt",
        path_dir + "MNISTFashionDense.txt"
      ]
    )

    # Created every time image es selected
    self.conv_model = None

    theme = TkThemeLoader(self.top)
    theme.setStylesheet(self.style)  # pass as string

    self.top.mainloop()
  
  def canvasClick( self, event ):

    x = event.x // self.rect_width
    y = event.y // self.rect_height
    
    if self.mode == MNIST_MODE or self.mode == MNIST_FASHION_MODE:
      color_value = self.scale1.get()
      
      color = rgbtohex(
        color_value, color_value, color_value
      )

      self.canvas.itemconfig(
        self.rectangles[ x ][ y ],
        fill = color
      )
      self.values[ x ][ y ] = ( color_value, color_value, color_value )

    elif self.mode == CIFAR10_MODE:
      color_r = self.scale1.get()
      color_g = self.scale2.get()
      color_b = self.scale3.get()
      
      color = rgbtohex(
        color_r, color_g, color_b
      )

      self.canvas.itemconfig(
        self.rectangles[ x ][ y ],
        fill = color
      )
      self.values[ x ][ y ] = ( color_r, color_g, color_b )

  def actionClick( self, event ):
    if self.mode == MNIST_MODE:
      print(
        self.mnist_model.testByData(
          self.export()
        )
      )
    elif self.mode == MNIST_FASHION_MODE:
      print(
        self.mnist_fashion_model.testByData(
          self.export()
        )
      )
    elif self.mode == IMAGE_CONV_MODE:
      
      file = filedialog.askopenfilename(
        initialdir = "/", title = "Selecciona imagen",
        filetypes = [
          ("Image files", ".png .jpg")
        ]
      )

      w, h, image = fg.openImage( file )

      if image is None:
        print( "Invalid image" )
        return

      self.conv_model = ConvModel(
        self.getFilter(), w, h
      )

      result = self.conv_model.testByData(
        image
      )[0]

      # Saved due to Python garbage collector
      self.tk_image = ImageTk.PhotoImage(
        result
      )

      result.save("test.png")

      if self.image is None:
        self.image = self.canvas.create_image(
          100, 100, anchor = NW, image = self.tk_image
        )
      else:
        self.canvas.itemconfig(
          self.image, image = self.tk_image
        )

  def export( self, rgb = False ):

    if rgb :
      return self.values
      
    return list( map( 
      lambda row : list( map(
        lambda value: value[0],
        row
      )), self.values
    ))

  def updateGUI( self, mode ):

    if mode == MNIST_FASHION_MODE or mode == MNIST_MODE or mode == CIFAR10_MODE:
      
      self.filter_box.place_forget()

      self.scale1.place(
        x = 0, y = self.canv_height  
      )

      self.action.config( text = "Clasificar")

      if mode == CIFAR10_MODE:

        self.scale2.place(
            x = 200, y = self.canv_height 
          )

        self.scale3.place(
          x = 400, y = self.canv_height
        )

      else:
        self.scale2.place_forget()
        self.scale3.place_forget()

      self.resetCanvas( True )

    elif mode == IMAGE_CONV_MODE:
      
      self.filter_box.place(
        x = 100, y = self.canv_height
      )

      self.scale1.place_forget()
      self.scale2.place_forget()
      self.scale3.place_forget()

      self.action.config( text = "Abrir imagen")
      self.resetCanvas( False )  

    self.mode = mode

  def resetCanvas( self, showRectangles = False ):
    
    for irow in range( self.height ):
      for icol in range( self.width ):

        item_id = self.rectangles[ irow ][ icol ]

        self.canvas.itemconfig(
          item_id, state = "normal" if showRectangles else "hidden",
          fill = rgbtohex( 0, 0, 0 )
        )
        self.values[ irow ][ icol ] = ( 0, 0, 0)

    if self.image is not None:
      self.canvas.itemconfig(
        self.image, state = "hidden" if showRectangles else "normal"
      )

  def getFilter( self ):
    index = self.filter_box.current()
    print( index )

    if index == 0:
      return [[
        [ 0, 0, 0], [ 0, 1, 0], [ 0, 0, 0]
      ]]
    elif index == 1:
      return [[
        [ -1, -1, -1 ], [ -1, 4, -1 ], [ -1, -1, -1]
      ]]
    elif index == 2:
      return [[
        [ 1/9, 1/9, 1/9 ], [ 1/9, 1/9, 1/9 ], [ 1/9, 1/9, 1/9 ]
      ]]
