# Introducción a diferentes modelos de IA

En este repositorio comparamos 3 tipos de algoritmos: 

 1) [Redes neuronales](#id1)
 2) [Aprendizaje por refuerzo](#id2)
 3) [Árboles de Búsqueda de Monte Carlo](#id3) 

Para visualizar resultados, creamos un modelo de cada tipo y los entrenamos para jugar al 3 en raya.

# Redes neuronales <a name=id1> </a>

Una red neuronal es un modelo que emula el modo en que el cerebro humano procesa la información. 
La estructura de la red neuronal está formada por pequeños procesadores de información, llamados neuronas 
artificiales, basándose en el modelo de neuronas del cerebro humano desarrollado en sus inicios por Ramon y Cajal. 
De forma breve las similitudes entre la neurona humana y la articial son las siguientes.


|  Neurona Humana  |  Neurona Artificial  |
|------------------|----------------------|
| <image src="/images/neurona_humana.jpg" height="350"> | <image src="/images/neurona_artificial.png" height="350"> |
| <p>1. Canal de entrada de información: las dendritas.</p> <p>2. Órgano de cómputo: el soma.</p> <p>3. Canal de salida: el axón.</p> | <p>1. Canal de entrada de información: pesos de la neurona, denotados como $\vec{w} = (w_0, w_1,\dots,w_n) \in \mathbb{R}^{n+1}$. La información de entrada es un vector $\vec{x} = (x_1,\dots,x_n) \in \mathbb{R}^{n}$.</p> <p>2. Órgano de cómputo: producto vectorial entre $\vec{w}$ y $\vec{x}$. Este resultado es procesado por una función de activación $f:\mathbb{R} \rightarrow \mathbb{R}$.</p> <p>3. Canal de salida: el resultado del computo es guardado en la variable $y$.</p> |

La estructura de la red neuronal se divide en tres capas principales: la capa de entrada (input layer), una o 
más capas intermedias (hidden layers) y la capa de salida (output layer). En esta estrutura, las funciones de 
activación son una parte importante y se explicarán en la siguiente sección. Su elección depende principalmente de
dos factores: su derivavilidad y su coste computacional.

El entrenamiento de una red neuronal se centra en minimizar el error. Para ello, durante el proceso de aprendizaje 
se recopilan un gran número de pares de datos, $(\vec{x},y)$, donde $\vec{x}\in\mathbb{R}^n$ representa 
un dato de entrada e $y\in \mathbb{R}$ reprensenta su salida objetivo. Para cada par, se evalua $\vec{x}$ en la red 
neuronal obteniéndose la salida $z\in\mathbb{R}$, de ahí la red modifica los pesos con objetivo de minimizar el error 
entre la salida de la red, $z$ y la salida objetivo, $y$. 

# Aprendizaje por refuerzo <a name=id2> </a>

El objetivo del aprendizaje por refuerzo es extraer qué acciones deben ser elegidas en los diferentes 
estados para maximizar la recompensa. En cierta forma, buscamos que el agente aprenda lo que se llama 
una política, que formalmente podemos verla como una aplicación que dice en cada estado qué acción 
tomar. Dividiremos la política del agente en dos componentes: por una parte, cómo de buena cree el 
agente que es una acción sobre un estado determinado y, por otra, cómo usa el agente lo que sabe 
para elegir una de las acciones posibles.

<image src="/images/refuerzo.png" width="750" height="350">


# Árboles de busqueda de Monte Carlo <a name=id3> </a>

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



