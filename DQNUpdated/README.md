# Inidividual Project 3
# Deep Q-learning Network(DQN)
Please don't revise test.py, environment.py and agent.py

#### Due Date
* Thursday October 17, 2019 (23:59)

#### Total Points
* 100 (One Hundred)

## Leaderboard and Bonus Points
In this project, we will provide a leaderboard and give **10** bonus points to the **top 3** highest reward students! 
* Due date for leaderboard
   * Thursday **October31**, 2019 (23:59)
* Where to see the leaderboard 
  * We will create a discussion on Canvas and each of you can post your highest reward with a sreenshot. TA will summarize your posts and list the top 3 highest rewards every Thursday after the regular due date of the project 3 and post it below. <br>
  
  **Leaderboard for Breakout-DQN** 
  **Update Date: 10/31/2019 17:30**
  
  | Top | Date | Name | Score | Note |
  | :---: | :---:| :---: | :---: | :---: |
  | **1**   |10/31/2019| **Mohamed Mahdi Alouane** | **329.46**     | Double DQN with 1e-6 learning rate trained for 100K episodes
  | **2**   |10/22/2019| **Prathyush SP**          |  **142.77**    | Conv Network and Priority Buffer trained for 50k episodes
  |     |10/18/2019| Prathyush SP          |  81.07     | Simple DQN with Conv Based Architecture for 60k episodes
  | **3**   |10/28/2019| **Sapan Agrawal**         |   **91.34**    | Architecture described in the DQN paper for 120k episodes
  | 4   |11/1/2019| Hanshen Yu| 86.82 | Double DQN with suggested architecture, suggested params for 20k episodes
  | 5   |10/26/2019| Vamshi Krishna Uppununthala|  79.5   | Dueling DQN for 50k episodes
  | 6   |10/31/2019| Sai Vineeth K V | 66.5 | Dueling DQN with 19k episodes of training
  | 7   |10/24/2019|Shreesha Narasimha Murthy  |56.79     | Simple DQN with MSE for 40k episodes
  | 8   |10/20/2019|Sinan Morcel            |53.26        |  Plain DQN with TA's parameters
  
* How to elvaluate
  * You should submit your lastest trained model and python code before leaderboard due date. TA will run your code to make sure the result is consistent with your screenshot. 
* How to grade
  * Top 3 students on the leaderboard can get 10 bonus points for project 3.
  

## Installation
Type the following command to install OpenAI Gym Atari environment in your **virutal environment**.

`$ pip install opencv-python-headless gym==0.10.4 gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python test.py --test_dqn`

## Goal
In this project, you will be asked to implement DQN to play [Breakout](https://gym.openai.com/envs/Breakout-v0/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/). The goal of your training is to get averaging reward in 100 episodes over **40 points** in **Breakout**, with OpenAI's Atari wrapper & unclipped reward. For more details, please see the [slides](https://docs.google.com/presentation/d/1CbYqY5DfXQy4crBw489Tno_K94Lgo7QwhDDnEoLYMbI/edit?usp=sharing).

<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project3/metarials/project3.png" width="80%" >

## Deliverables
Please compress all the below files into a zipped file and submit the zip file (firstName_lastName_hw3.zip) to Canvas.

* **Trained Model**
  * Model file
  * If your model is too large for Canvas, upload it to a cloud space and write download.sh to download the model

* **PDF Report**
  * Set of Experiments Performed: 
    * Include a section describing the set of experiments that you performed
    * what structures you experimented with (i.e., number of layers, number of neurons in each layer)
    * what hyperparameters you varied (e.g., number of epochs of training, batch size and any other parameter values, weight initialization schema, activation function)
    * what kind of loss function you used and what kind of optimizer you used.
  * Special skills: Include the skills which can improve the generation quality. Here are some [tips](https://arxiv.org/pdf/1710.02298.pdf) may help. (Optional)
  * Visualization: Learning curve of DQN. 
    * X-axis: number of time steps
    * Y-axis: average reward in last 30 episodes.

* **Python Code**
  * All the code you implemented including sample codes.

## Grading
* **Trained Model (50 points)**
  * Getting averaging reward in 100 episodes over **40 points** in Breakout will get full credits. 
  * For every average reward below 40, you will be taken off 2 points. i.e., you will be taken off 2 points, if getting averaging reward in 100 episodes is 39 and taken off 4 points, if averaging reward is 38, so on so forth.

* **PDF Report (30 points)**
  * Set of parameters performed: 20 points
  * Visualization: 10 points
  
* **Python Code (20 points)**
  * You can get full credits if the scripts can run successfully, otherwise you may loss some points based on your error.

## Hints
* [Naive Pytorch Tutorial](https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project3/metarials/Pytorch_tutorial.ipynb)
* [Official Pytorch Tutorial](https://pytorch.org/tutorials/)
* [Official DQN Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [Official DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)
* [DQN Tutorial on Medium](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)

## Tips for Using GPU on Google Cloud
* [How to use Google Cloud Platform](https://docs.google.com/document/d/1JfIG_yBi-xEIdT6KP1-eUpgLDoY3t2QrAKULB9yf01Q/edit?usp=sharing)
* [How to use Pytorch on GPU](https://docs.google.com/document/d/1i8YawKjEwg7qpfo7Io4C_FvSYiZxZjWMLkqHfcZMmaI/edit?usp=sharing)
* Other choice for GPU
  * Use your own GPU
  * Apply [Ace account](https://arc.wpi.edu/computing/accounts/ace-accounts/) from WPI 
