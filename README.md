# Introducción a diferentes modelos de IA

En este repositorio comparamos 3 tipos de algoritmos: 
 1) [Redes neuronales](#ANN)
 2) [Aprendizaje por refuerzo](#RL)
 3) [Árboles de Búsqueda de Monte Carlo](#MCTS) 

Para visualizar resultados, creamos un modelo de cada tipo y los entrenamos para jugar al 3 en raya.

# Redes neuronales <a name=ANN />

# Aprendizaje por refuerzo <a name=RL />

# Árboles de busqueda de Monte Carlo <a name=MCTS />

Los arboles de búsqueda de montecarlo, MCTS (Monte Carlo Tree Search), son una técnica de 
búsqueda en el campo de la Inteligencia Artificial (IA), cuyo principal objetivo es maximizar 
la recompensa obtenida. Son usados ampliamente en juegos en los que se involucran 2 jugadores, 
uno contra el otro. 

El MCTS es un algoritmo de búsqueda heurístico y probabilístico que combina 
las implementaciones clásicas de búsqueda de árboles junto con los principios de 
aprendizaje automático del aprendizaje por refuerzo. 

En la búsqueda de árbol, siempre existe la posibilidad de que la mejor acción actual realmente no sea la 
acción más óptima. En tales casos, el algoritmo MCTS se vuelve útil ya que continúa evaluando otras alternativas 
periódicamente durante la fase de aprendizaje. Esto se conoce como el 'trade-off de exploración-explotación', tal 
como se explicó en el anterior capítulo.

 - Exploración. Explorar el espacio local de decisiones alternativas y averiguar si se podrían mejorar 
los resultados actuales. La exploración puede ser útil para garantizar que MCTS no pase por alto ningún 
camino potencialmente mejor. Pero rápidamente se vuelve ineficiente en situaciones con gran cantidad de 
pasos o repeticiones.
 - Explotación. Explota las acciones y estrategias que se encuentran como las mejores en el momento. 


## Estructura

El MCTS es una estructura arbórea en la que cada uno de los nodos del árbol va a representar un
estado $s$. La información importante la vamos a tener en las aristas, que van a representar
las distintas acciones que se pueden realizar desde un estado. Dos nodos (representando estados $s_0$
y $s_1$) estarán conectados por una arista si existe una acción $a$ tal que al realizar $a$ en $s_0$ nos lleva
a $s_1$. Cada arista presente en el árbol estará caracterizada por un par $(s, a)$ y, para cada una de
ellas, el MCTS va a guardar la información que necesita para decidir cuán prometedora es.

Los datos que se utilizan para calcular lo prometedora que es cada acción son 3:

 - $N(s, a)$: el número de veces que se ha ejecutado la acción $a$ desde el estado $s$.
 - $Q(s, a)$: el valor medio para la acción $a$ desde el estado $s$. Cada simulación ejecutada que
	pase por el estado $s$ y realice la acción $a$ va a terminar en una posición con un cierto valor $v$.
	Este campo contendrá la media de estos valores $v$, indicando cuán bueno ha sido en promedio
	dicho movimiento.
 - $W(s, a)$: la suma de los valores de las posiciones a las que se ha llegado en las simulaciones
	en las que se ha ejecutado la acción $a$ desde el estado $s$. Nos permitirá actualizar $Q(s, a)$ de
	forma más sencilla, $Q(s, a) = \dfrac{W(s,a)}{N(s,a)}$.



## Ejecución

La idea es construir un MCTS al comienzo de una partida con un único nodo representando
el estado inicial, la partida en la que no se ha efectuado ningún movimiento. A la hora de elegir
cada movimiento, vamos primero a poblarlo mediante simulaciones. Utilizaremos la información
recabada para decidir qué acción realizar y pasaremos al siguiente estado, donde repetiremos el
proceso.

Para un cierto estado, el movimiento a realizar se va a escoger en dos fases. La primera fase es
la de simulación, que es la encargada de poblar el MCTS. Se van a realizar múltiples simulaciones
y en cada una de ellas se va a seguir la línea de juego considerada más prometedora, hasta que se
llegue a un estado inexplorado. Cada simulación se divide a su vez en 4 etapas: 

 - Selección: se escogen las acciones a realizar en base a la información que se tiene hasta llegar
	a un nodo sin explorar.
 - Expansión: se expande el nuevo estado.
 - Evaluación: se evalúa el nuevo estado.
 - Actualización: se actualiza el MCTS incorporando la información obtenida en esta ejecución.

<image src="/images/mcts_fases.png" width="750" height="350">

Realizaremos simulaciones hasta haber hecho un número suficiente, o hasta que consumamos el
tiempo disponible. Una vez poblado, en la segunda fase, de elección, se escoge el movimiento más
prometedor en base a esta información.



