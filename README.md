# Introducción a diferentes modelos de IA

<img src="images/fondo_2.png" width="40%" align="right" />

En este repositorio comparamos 4 tipos de algoritmos: 

 1) Algoritmo Minimax
 2) Redes neuronales
 3) Aprendizaje por refuerzo
 4) Árboles de Búsqueda de Monte Carlo 

He creado un [PDF]() en el que explico una introducción a 3 de las ramas mencionadas anteriormente, dejando fuera el Algoritmo Minimax. Esto es debido a que me quiero centrar en algoritmos de aprendizaje máquina, y el minimax como tal no tiene un modelo que evolucione; sin embargo lo he introducido en el repositorio porque me ha parecido una buena introducción para los demás.

### Ejecución

Para poder unificar todos estos modelos hemos creado una clase para cada uno con la función **move()** con el fin de que tengan todos la misma estructura. Así el usuario únicamente tienen que elegir contra qué modelo jugar y seguidamente se comenzará la partida. Para comenzar ejecutar el script:

```python
main.py
```

#### Librerías necesarias

 - numpy: para fórmulas matemáticas y números aleatorios.
 - tkinter: para los gráficos del juego.
 - pytorch: para la red neuronal.

# Índice

Aquí proporciono un pequeña introducción a cada uno de los algoritmos y después explico el código que he desarrollado para cada uno, por lo que si ya se tiene una noción básica del algoritmo se puede pasar directamente a esta segunda parte.

 1. [Algoritmo Minimax](#id1)
    - [Explicación del código](#id1.1)

 2. [Aprendizaje por refuerzo](#id3)
    - [Explicación del código](#id3.1)
    - [Resultados](#id3.2)

 3. [Árboles de Búsqueda de Monte Carlo](#id4)
    - [Explicación del código](#id4.1) 
    - [Resultados](#id4.2)

 4. [Redes neuronales](#id2)
    - [Explicación del código](#id2.1)
    - [Resultados](#id2.2)

## Minimax <a name=id1></a>

<img src="images/minimax.png" width="500" align="right"/>

El algoritmo Minimax es una técnica fundamental en la teoría de juegos y la inteligencia artificial. Se utiliza para la toma de decisiones en situaciones competitivas de suma cero, donde dos jugadores se enfrentan en un juego de estrategia.

En el contexto matemático, consideremos un juego con información perfecta y un árbol de juego completo. Sea G un juego entre dos jugadores, llamémoslos Jugador 1 (Max) y Jugador 2 (Min). Cada nodo del árbol de juego representa un estado del juego, y las aristas que salen de un nodo representan las posibles acciones o movimientos que se pueden tomar.

El objetivo del algoritmo Minimax es determinar la mejor jugada para Jugador 1 (Max) en cada estado del juego, asumiendo que Jugador 2 (Min) también juega de manera óptima. Esto se logra mediante una exploración exhaustiva del árbol de juego, utilizando una estrategia de retroceso (backtracking) para evaluar los posibles resultados de cada movimiento.

### Explicación del código <a name=id1.1></a>

*Observación: El código está desarrollado en la carpeta **Minimax/**. Finalmente, para poder jugar usaremos el objeto **Minimax_Model** del script **model.py**.* Para no tener que estar evaluando todo el rato el árbol de busqueda del algoritmo minimax, lo que hacemos en evaluar todos lo tableros posibles y guardar las acciones que hay que hacer en **rules.pkl**. Esto último es realizado en el script **create_model.py**.

El algoritmo Minimax se basa en la exploración exhaustiva de todos los movimientos posibles. Comienza evaluando el estado actual del juego representado por el objeto game. Si el juego ha terminado, la función devuelve una tupla (X, nº de victorias, nº de derrotas) donde X puede ser 0 para empate, 1 para victoria o -1 para derrota.

```python
    if game.finished():
        return (1, game.empty_spaces + 1, None)
    if game.full():
        return (0, 0, None)
```

Si el juego no ha terminado, la función itera sobre todas las posibles jugadas y realiza una llamada recursiva a sí misma para analizar los escenarios resultantes. Los resultados de las jugadas se registran en las listas x1 y x2, donde x1 almacena el resultado de cada jugada y x2 registra el número de victorias obtenidas para cada jugada.

Finalmente, la función selecciona la mejor jugada basándose en los resultados obtenidos. Si hay una jugada que resulta en victoria, se selecciona esa jugada. En caso de empate, se elige la jugada con más victorias acumuladas.

```python
    i = np.argmax(x1)
    if x1[i] < 1:
        index = [j for j in range(9) if x1[j] == x1[i]]
        j = np.argmax([x2[j] for j in index])
        i_max = index[j]
    else:
        i_max = i
```



## Aprendizaje por refuerzo <a name=id3></a>

<img src="images/rl.png" width="500" align="right" />

El Aprendizaje por Reforzamiento es un enfoque de la inteligencia artificial donde un agente aprende a tomar decisiones óptimas interactuando con su entorno y recibiendo retroalimentación en forma de recompensas. 

El algoritmo Q-learning es una técnica destacada en este campo. Busca aprender una función de valor óptimo llamada Q-función, que asigna valores a pares de estado-acción y representa la utilidad esperada a largo plazo. Mediante la exploración y explotación, el agente actualiza iterativamente los valores de la Q-función utilizando la regla de actualización de Q. Una vez que la Q-función ha convergido a su valor óptimo, el agente puede tomar decisiones óptimas eligiendo la acción con el mayor valor Q en cada estado.

### Explicación del código <a name=id3.1></a>

*Observación: El código está desarrollado en la carpeta **RL/** (Reinforcement Learning). Para el entrenamiento hemos usado el objeto **Q_trainner** del script **model.py**. Finalmente, guardaremos el modelo resultante (la tabla Q-table) en la carpeta models (concretamente con la librería pickle) y para poder jugar usaremos el objeto **RL_Model**.*

Primero creamos la Q-table vacía, es decir, para cada una de las posibles combinaciones del juego, creamos una fila de 9 elemento en los que iremos modificando para conseguir el mejor movimiento en cada jugada. 

La única información que tiene el juego es el siguiente movimiento, es decir, si en el siguiente movimiento:

 - Ganamos: ```q_next = 1```
 - Perdemos: ```q_next = -1```
 - Empatamos: ```q_next = 0```

Un vez tenemos dicha información, podemos empezar a simular partidas de prueba para completar la tabla lo mejor posible. Para este último paso utilizamos la función **updateQtable()** del objeto **Q_trainner**. Si no terminamos justo en el siguiente movimiento entonces tendremos que que actualizar el valor con el siguiente mejor Q posible, ```q_next = self.chose_best_action(next_game.board, q=True)```. Finalmente actualizamos la tabla de la siguiente forma: 

```python
q_new = (1 - self.alpha) * q0 + self.alpha * (reward + q_next)
```

Tras una serie de iteraciones obtenemos buenos resultados. Podemos ver la evolución del entrenamiento en el siguiente gráfico:

<img src="RL/models/q_100k_e1.png" width="400"/>


### Resultados <a name=id3.2></a>

El modelo final en la práctica, pese a no realizar los movimientos óptimos siempre, evita perder en todas la ocasiones y además si no se efectuan los movientos correctos, también es capaz de ganar al oponente. Podemos observar los resultados con unos ejemplos:


| Jugada | Comentarios |
|--------|-------------|
| <img src="images/RL/draw.gif" width="200" height="220"/> | **Empate:** la IA es el jugador "O". Podemos observar como siempre evita que el "X" gane. |
| <img src="images/RL/winner.gif" width="200" height="220"/> | **Gana la IA:** la IA es el jugador "X". Podemos observar como la IA le deja sin movimientos al oponente y consigue ganarle. |

En general, podemos ver como contra un oponente con movimientos aleatorios siempre gana, mientras que contra uno que sabe jugar siempre llegan al empate.

<img src="images/scores/rl.png" height="500" width="750"/>




## Árboles de Búsqueda de Monte Carlo <a name=id4></a>

<img src="images/mcts_fases.png" width="500" align="right" />

Los Árboles de Búsqueda de Monte Carlo (MCTS) son una técnica eficiente para la toma de decisiones en juegos y problemas de búsqueda. Utilizando simulación Monte Carlo y la construcción de un árbol de búsqueda, el MCTS busca encontrar la mejor acción en un estado dado, maximizando la recompensa a largo plazo. 

Durante la selección, se eligen nodos para la exploración y expansión, seguidos de simulaciones Monte Carlo para estimar la recompensa. Los valores de los nodos se actualizan usando la regla de copia de respaldo. Este proceso se repite hasta alcanzar un límite de tiempo o un criterio de terminación. 

### Explicación del código <a name=id4.1></a>

*Observación: El código está desarrollado en la carpeta **MCTS/** (Monte Carlo Tree Search). Para el entrenamiento hemos usado el objeto **MCTS_Model** del script **model.py**, aunque este se entrena directamente al crearse (el entrenamiento es muy corto, de unos pocos segundos). Finalmente, para poder jugar usaremos el objeto **MCTS_Model**.*

Con respecto al poco entrenamiento necesario para este modelo, he de mencionar que en cada movimiento volvemos a realizar una pequeña busqueda (de segundos, pues se realiza en vivo mientras se juega). El hecho es que este modelo realmente es una modificación mejorada del minimax. Fue creado para espacios de busqueda de grandes dimensiones en los que no es viable realizar una busqueda completa con minimax. 

El beneficio del MCTS, es que debido al historial que guarda en cada uno de los nodos, es capaz de recordar que jugadas son mejores y por ello realizar una busqueda más inteligente. Aunque respecto a esto último existe el dilema exploración-expansión que explico más detalladamente en el [PDF](). Por lo tanto para el un juego como el tres en raya, con un espacio pequeño, no es necesario un modelo tan complejo. Pero me parece un gran ejemplo para comprender su funcionamiento.

### Resultados <a name=id4.2></a>

Podemos observar los resultados con unos ejemplos:

| Jugada | Comentarios |
|--------|-------------|
| <img src="images/MCTS/draw.gif" width="200" height="220"/> | **Empate:** la IA es el jugador "X". Podemos observar como siempre evita que el oponente gane. |
| <img src="images/MCTS/winner.gif" width="200" height="220"/> | **Gana la IA:** la IA es el jugador "X". Podemos observar como la IA le deja sin movimientos al oponente y consigue ganarle. De hecho si se fijan detalladamente en la jugada, desde que el contrincante realiza su primer movimiento, el modelo le atrapa en una secuencia de movimiento en los que terminan con un 100% de victoria para la IA. |

En general, podemos observar como, al igual que en el aprendizaje por refuerzo, contra un oponente con movimientos aleatorios siempre gana, mientras que contra uno que sabe jugar siempre llegan al empate.

<img src="images/scores/mcts.png" height="500" width="750"/>



## Redes neuronales <a name=id2></a>

<img src="images/ann.png" width="500" align="right" />

Las redes neuronales tipo perceptrón son un enfoque fundamental en el aprendizaje automático y la inteligencia artificial. En el contexto del juego del tres en raya, se pueden utilizar para automatizar el juego y tomar decisiones estratégicas.

El perceptrón es la unidad básica de una red neuronal, que combina entradas ponderadas y aplica una función de activación para generar una salida. En este caso, las entradas representan el estado actual del tablero y las salidas indican la mejor jugada posible.

El entrenamiento implica ajustar los pesos y sesgos del perceptrón utilizando ejemplos de entrada y salida esperada. El objetivo es desarrollar una red neuronal entrenada que juegue al tres en raya de manera competente, mejorando la experiencia de juego y explorando conceptos en el aprendizaje automático.

### Explicación del código <a name=id2.1></a>

*Observación: El código está desarrollado en la carpeta **ANN/** (Artificial Neural Network). Para el entrenamiento hemos usado el objeto **ANN_trainner** del script **model.py**. Finalmente, guardaremos el modelo resultante (la red neuronal) en la carpeta models (concretamente con la librería de pytorch) y para poder jugar usaremos el objeto **ANN_Model**.*

Para su entrenamiento, esta rebirá como entrada un vector de longitud 9 (las casillas del juego) mapeadas de la siguiente manera:

 - 1 si está la ficha "x".
 - -1 si está la ficha "o".
 - 0 si está la casilla vacía.

y para la salida proporcionaremos un vector (de longitud 9) que indique la probabilidad de realizar cada una de las acciones, gracias a la función softmax.

### Resultados <a name=id2.2></a>

Después del entrenamiento, se observan los peores resultados en comparación con los otros tres modelos. No obstante, es posible modificar la estructura de la red neuronal y experimentar con diferentes conjuntos de datos de entrenamiento. Sin embargo, he decidido mantener estos resultados para ilustrar cómo la red neuronal aprende a partir de cada par de elementos de entrada y salida deseados. Es importante destacar que la red neuronal valora de manera similar tanto los movimientos intermedios como el último movimiento que lleva a una victoria o derrota en el juego, a diferencia de enfoques como el árbol de búsqueda y el aprendizaje por refuerzo, donde se refuerza de manera más significativa el último paso. Debido a que la red neuronal no ha completado su entrenamiento de manera óptima, no ha logrado aprender algunas funciones básicas clave, como bloquear al oponente cuando está a punto de ganar o colocar una ficha en la posición correcta para que la IA gane, lo que explica los resultados deficientes obtenidos.

<img src="images/scores/ann.png" height="500" width="750"/>


