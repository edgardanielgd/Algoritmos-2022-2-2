># Algoritmos 2022 - 2 ( Grupo 2 )
>
>## Laboratorio 2
>Se tomó la secuenciación del Genoma Viral Sars-Cov 2 y se implementó un algoritmo de fuerza bruta que recibirá como entrada dicha secuencia y una sub secuencia a buscar. Se retornará la cantidad de recurrencias de la sub secuencia "target" en la secuencia original "sequence", así como los índices de inicio y fin de dichas ocurrencias.
>
>### Complejidad de búsqueda:
>

```python
def  busqueda(subcadena,cadena): 			# O(n)

	lista=[]  								# O(1)
	posicion=0  							# O(1)
	contador=0								# O(1)
	posString=0								# O(1)
	if subcadena=="":  						# O(1)
		return  print("Subcadena vacia!")	# O(1)

	for i in cadena:  						# O(n)
		if i==subcadena[posString]:  		# O(1)
			posString+=1  					# O(1)
		else:								# O(1)
			posString=0  					# O(1)
			if subcadena[posString] == i:  	# O(1)
				posString = 1				# O(1)

		if posString==len(subcadena):  		# O(1)
			lista.append(
				[posicion-len(subcadena)+1,
				posicion]
			)								# O(1)
			contador+=1  					# O(1)
			posString=0  					# O(1)
			
		posicion+=1  						# O(1)
		
	return  print(contador, lista)			# O(1)
```
>La complejidad del algoritmos es del orden O(n)