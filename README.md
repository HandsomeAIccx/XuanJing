

## Component Introduction

1. Env is in responsible for parallelizing and wrapping the environment.
2. The task of interacting with the environment falls to the actor.
3. When an actor interacts with an environment, connection is in charge of managing the data format.
4. The data produced during the interaction between the actor and the environment is stored in the buffer.
5. enhancement is used to enhance the data in the buffer.
6. The algorithm in the algorithms module is used to process the data in the buffer.
7. Model parameters are updated by the learner using data and algorithms.
8. utils are a class of useful functions.

