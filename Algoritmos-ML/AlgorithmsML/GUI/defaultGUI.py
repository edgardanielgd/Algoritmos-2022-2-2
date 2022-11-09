from tkinter import *

from AlgorithmsML.models.mnist_fashion_model import *
from AlgorithmsML.models.mnist_model import *
from AlgorithmsML.models.cifar10_model import *

def rgbtohex(r,g,b):
  return f'#{r:02x}{g:02x}{b:02x}'

class GUInterface:
    
  def __init__( self, width, height ):

    self.single_color = True

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
      bg="white"
    )
  
    self.canvas.pack()
    self.canvas.place(
      x = 0, y = 0
    )
  
    self.canvas.bind(
      "<Button-1>",
      lambda event: self.canvasClick( event )
    )
  
    self.scale1 = Scale(
      self.top, from_=0, to=255,
      orient=HORIZONTAL
    )
  
    self.scale1.pack()
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
  
    # self.scale3.pack()
    

    self.generate = Button(
      self.top,
      text = "Clasificar"
    )

    self.generate.pack()
    self.generate.place(
      x = self.canv_width + 20 , y = 100
    )
    self.generate.bind(
      "<Button-1>",
      lambda event : self.classifyClick( event )
    )

    self.menubar = Menu( self.top )
    self.typeMenu = Menu( self.menubar, tearoff = 0)

    self.typeMenu.add_command(
      label = "Mnist", command = lambda : self.mnistMenu()
    )

    self.typeMenu.add_command(
      label = "Mnist fashion", command = lambda : self.fashionMenu()
    )

    self.typeMenu.add_command(
      label = "Cifar 10", command = lambda : self.cifarMenu()
    )

    self.menubar.add_cascade( 
      label = "Tipo",
      menu = self.typeMenu
    )

    self.top.config( menu = self.menubar )

    # Draw squares
    self.rectangles = []
    self.values = []

    for i in range( self.width ):

      row = []
      values_row = []
      
      for j in range( self.height ):
        row.append(
          self.canvas.create_rectangle(
            i * self.rect_width,
            j * self.rect_height,
            i * self.rect_width + self.rect_width,
            j * self.rect_height + self.rect_height,
            fill = "white"
          )
        )

        values_row.append(
          ( 255, 255, 255 )
        )

      self.values.append(
        values_row
      )

      self.rectangles.append(
        row
      )
    
    self.top.mainloop()
  
  def canvasClick( self, event ):

    x = event.x // self.rect_width
    y = event.y // self.rect_height
    
    if self.single_color:
      color_value = self.scale1.get()
      
      color = rgbtohex(
        color_value, color_value, color_value
      )

      self.canvas.itemconfig(
        self.rectangles[ x ][ y ],
        fill = color
      )
      self.values[ x ][ y ] = ( color_value, color_value, color_value )
    else:
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

  def classifyClick( self, event ):
    print(" ola ")

  def export( self ):

    
    return self.values

  def mnistMenu( self ):
    self.scale2.pack_forget()
    self.scale3.pack_forget()

    self.single_color = True

  def fashionMenu( self ):
    self.scale2.pack_forget()
    self.scale3.pack_forget()

    self.single_color = True

  def cifarMenu( self ):
    self.scale2.pack()
    self.scale3.pack()

    self.scale2.place(
      x = 200, y = self.canv_height 
    )

    self.scale3.place(
      x = 400, y = self.canv_height
    )

    self.single_color = False
