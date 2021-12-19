# Proyecto - Chrome Dino con DQN

- Carlos Molina
- Mayra Salazar
  
## Sobre el proyecto

![alt tag](game.gif)

  
Se trata de enseñarle al chrome dino como se debe jugar por si mismo.
Originalmente, debería de ser un algoritmo que utiliza DQN, sin embargo optamos por el PPO2 ya que en el módulo stable-lines el DQN no soporta multiprocesos.  

Utilizamos `pyenv` para poder hacer un downgrade de Python 3.8.10 a Python 3.7.1, ya que el módulo `stable-lines` utiliza una versión vieja de Tensorflow (1.15).  

Para correr el proyecto:  

```Bash
    pipenv install
```
```Bash
    pipenv shell
```
```Bash
    ./dinoAI.py true
```


## Bibliografía y referencias:
- https://blog.paperspace.com/dino-run/
- https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
- https://stable-baselines.readthedocs.io/en/master/guide/examples.html?highlight=gif#id2
- https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#which-algorithm-should-i-use