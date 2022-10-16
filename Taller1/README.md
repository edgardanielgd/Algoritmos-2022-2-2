># Algoritmos 2022 - 2 ( Grupo 2 )
>
>## Taller 1
>Se implementó el método de Bisección de Bolzano para calcular, en base a un modelo o función y un intervalo predefinido
>
>### Complejidad de búsqueda:
>

```python
def roots_by_bisection( model ): # O( log n ),con n siendo el ancho del intervalo(base 2),una raiz de la 
				 # función en dicho intervalo dividido el minimo valor de f aceptado para retornar
  MIN_Y_VALUE = 1E-4  		# O(1)
  x0 = 0              		# O(1)
  x1 = 1000           		# O(1)

  x_middle = (x1 + x0) / 2      # O(1)

  fx0 = model( x0 )          	# O(1)
  fx1 = model( x1 )          	# O(1)
  fxm = model( x_middle )    	# O(1)

  while abs(fxm) > MIN_Y_VALUE: # O( log n )

    if fxm * fx0 < 0:           # O(1)

      x1 = x_middle             # O(1)
      fx1 = fxm                 # O(1)
    elif fxm * fx1 < 0:         # O(1)

      x0 = x_middle             # O(1)
      fx0 = fxm                 # O(1)
    else:
      print("No hay una raiz en éste intervalo !")  
      break
    
    x_middle = (x1 + x0) / 2    # O(1)
    fxm = model( x_middle )  	# O(1)
    
    
  return( x_middle, fxm )	# O(1)
```
>La complejidad del algoritmos es del orden O( log n ) (Logaritmo en base 2)
>Nótese que el método se asemeja a una búsqueda binaria, dado que el valor de n inicialmente es una variable continua, se puede tomar su valor entero como la división de n entre el mínimo valor de f(x) aceptado