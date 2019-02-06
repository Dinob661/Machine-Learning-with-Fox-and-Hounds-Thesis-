import pygame as pg
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import random
import os.path
import tensorflow as tf 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from math import sqrt


##Hyper-parameters##
LENGTH = 8                  #controls the board size. We are doing a normal chess sized board 8x8
EPISODES = 1000000          #How many games to train for
verbose = True              #set to true to prin the board
games_won_fox = 0           #used for stats
games_won_hound = 0         #used for stats
total_reward_fox = 0        #used for stats
total_reward_hound = 0      #used for stats
average_reward_fox = 0      #used for stats
average_reward_hound = 0    #used for stats


class Agent_Fox:
    def __init__(self):
        self.state_size = (LENGTH,LENGTH)           
        self.action_size = 4                        #outputs for fl, fr, bl, br
        self.memory = deque(maxlen=30000)           #How many states can the model remember
        self.gamma = 0.95                           #discout rate 
        self.epsilon = .9                          #exploration rate (anything less than 1 may result in rand move)
        self.epsilon_min = .8                      #the min threshold you want epsilon to decay to
        self.epsilon_decay = 0.995                  #the rate which epsilon decays from starting point
        self.learning_rate = 0.001                  #learning rate **not used** Adam defaults to 0.001 this is more for reference
        self.model = self.build_model()             #Build the model

    #Sets the role to fox
    def set_role(self, role):
        self.role = role

    #Loads model or creates new model
    def set_model(self, model):
        if model == None:
            self.model = self.build_model()
        else:
            self.model = model

    #Build the NN for the DQN
    def build_model(self):
        model = Sequential()
        model.add(Dense(64,input_dim=(LENGTH*LENGTH), activation='relu'))   #Input Layer
        model.add(Dense(64, activation='relu'))                             #Hidden Layer 1
        model.add(Dense(64, activation='relu'))                             #Hidden Layer 1
        model.add(Dense(64, activation='relu'))                             #Hidden Layer 1
        model.add(Dense(self.action_size, activation = 'softmax'))          #Output Layer
        model.compile(loss='mse', optimizer='Adam')
        return model 

    #Used for storing the necessary variables for experience replay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Provides an action to be performed
    def act(self,R, state, env):
        #get the list of allowed actions for the hound
        actions_allowed = env.allowed_actions_fox_agent()
        state = np.reshape(state, [1, LENGTH*LENGTH])
        a = np.random.uniform(0,1,size=1)

        if a[0] > self.epsilon: ##do a random move
            return actions_allowed[random.randint(0, len(actions_allowed)-1)], R
        act_values = self.model.predict(state)
        ######################
        #remove invalid actions
        for i in range(self.action_size):
            #print(act_values[0])
            if i not in actions_allowed:
                act_values[0][i]-=100
        ######################
        #check if move is valid, if it's not do a random move instead
        if np.argmax(act_values[0]) in actions_allowed:
            # print(actions_allowed)
            #print(np.argmax(act_values[0]))
            return np.argmax(act_values[0]), R
        else: #no valid moves so take random action
            # print('Taking Random Action -- Fox AI \n')
            # print(actions_allowed)
            # print(np.argmax(act_values[0]))
            R-=5 #punish for picking bad move
            return actions_allowed[random.randint(0, len(actions_allowed)-1)], R

    #Performs the action provided from act
    def take_action(self, action, env):
        action_to_take = action
        env.prev_fox_position = env.fox_position

        ##this moves the fox back (-1, -1) or what I call back left (bl)
        if action_to_take == 0:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (1,1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        ##this moves the fox back (-1, 1) or what I call back right (br)
        elif action_to_take == 1:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (1,-1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        ##this moves the fox back (1, -1) or what I call foward left (fl)
        elif action_to_take  == 2:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (-1,1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        ##this moves the fox back (1, 1) or what I call forward right (fr)
        elif action_to_take == 3:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (-1,-1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        return env.state

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         state = np.reshape(state, [1, LENGTH*LENGTH])
    #         next_state = np.reshape(next_state, [1, LENGTH*LENGTH])
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    #Performs Experience replay
    #This is a different implmentation currently than the fox model
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = np.reshape(state, [1, LENGTH*LENGTH])
            next_state = np.reshape(next_state, [1, LENGTH*LENGTH])
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss


    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         state = np.reshape(state, [1, LENGTH*LENGTH])
    #         next_state = np.reshape(next_state, [1, LENGTH*LENGTH])
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

class Agent_Hound:
    def __init__(self):
        self.state_size = (LENGTH,LENGTH)               #The board
        self.action_size = 8                            #2 outputs for fl and fr
        self.memory = deque(maxlen=30000)               #states the model can remember
        self.gamma = 0.95                               #discout rate 
        self.epsilon = .99                              #exploration rate (anything less than 1 may result in rand move)
        self.epsilon_min = .9                          #min threshold epsilon can decay to
        self.epsilon_decay = 0.95                       #controls the speed of epsilon decay
        self.learning_rate = 0.001                      #learning rate **not used** adam uses 0.001 by default this is for reference only
        self.model = self.build_model()                 #hounds model
        #self.target_model = self.build_model()         #unused, this is for experimenting with DDQN
        #self.update_target_model()                     #unused, this is for experimenting with DDQN                  
        #Hound action mapping
        self.hound_actions = {"h1fl": 0, "h1fr": 1,"h2fl": 2, "h2fr": 3,"h3fl": 4, "h3fr": 5,"h4fl": 6, "h4fr": 7}
   
    #Set the role of the agent
    def set_role(self, role):
        self.role = role

   #Loads or creates the hound model
    def set_model(self, model):
        if model == None:
            self.model = self.build_model()
        else:
            self.model = model
    
    #Set the hounds target model
    def set_target_model(self,model):
        if model == None:
            self.target_model = self.build_model()
        else:
            self.target_model = model

    #Update the target model
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    #Dense NN
    #Build the NN for the DQN
    # def build_model(self):
    #     model = Sequential()
    #     model.add(Dense(64,input_dim=(LENGTH*LENGTH), activation='relu'))   #Input Layer
    #     model.add(Dense(64, activation='relu'))                             #Hidden Layer 1
    #     model.add(Dense(64, activation='relu'))                             #Hidden Layer 1
    #     model.add(Dense(64, activation='relu'))                             #Hidden Layer 1
    #     model.add(Dense(self.action_size, activation = 'softmax'))          #Output Layer
    #     model.compile(loss='mse', optimizer='Adam')
    #     return model 

    #CNN 
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                 activation='relu',
                 input_shape=(8,8,1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer='Adam')
        return model

    #Saves the necessary variables for experience replay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Provides an action to be performed
    def act(self, R, state, env):
        #get the list of allowed actions for the hound
        actions_allowed = env.allowed_actions_hound_agent()
        #reshape the state for prediction
        #state = np.reshape(state, [1, LENGTH*LENGTH])----------------------------------This is for Dense NN Uncomment and comment out the line below if needed
        state = np.reshape(state, [1,8,8,1])
        #get random number for epsilon-greedy
        a = np.random.uniform(0,1, size=1)
        
        #Checks if hounds have reached a no-win state
        if len(actions_allowed) == 0:
            env.giveup += 1

        if a[0] > self.epsilon and len(actions_allowed) != 0: ##do a random move
            return actions_allowed[random.randint(0, len(actions_allowed)-1)], R
        act_values = self.model.predict(state)
        ######################
        #remove invalid moves
        for i in range(LENGTH):
            if i not in actions_allowed:
                act_values[0][i]-=100
        ######################
        #check if move is valid, if it's not do a random move instead
        if np.argmax(act_values[0]) in actions_allowed:
            #print(actions_allowed)
            #print(np.argmax(act_values[0]))
            #print(act_values)
            return np.argmax(act_values[0]), R
        else: #no valid moves so take random move but punish the hounds
            #print(actions_allowed)
            #print(np.argmax(act_values[0]))
            #print(act_values)
            #R -= 10 #punish hound for taking illegal move
            if len(actions_allowed) > 0:
                return actions_allowed[random.randint(0, len(actions_allowed)-1)], R
            else:
                return -1, R

    #performs action provided by act
    def take_action(self, action, env):
        action_to_take = action

        if action != -1:
            #1st hound move fl
            if action_to_take == 0: 
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 0 ##Start the hound move
                env.hound_positions[0] = np.subtract(env.hound_positions[0], (1,1)) ## update hound position
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 1 ##Place new hound position
                        
            #1st hound moves fr
            elif action_to_take == 1:
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 0 ##Start the hound move
                env.hound_positions[0] = np.subtract(env.hound_positions[0], (1,-1)) ## update hound position
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 1 ##Place new hound position

            #2nd hound moves fl
            elif action_to_take == 2: 
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 0 ##Start the hound move
                env.hound_positions[1] = np.subtract(env.hound_positions[1], (1,1)) ## update hound position
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 2 ##Place new hound position
                        
            #2nd hound moves fr
            elif action_to_take == 3:
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 0 ##Start the hound move
                env.hound_positions[1] = np.subtract(env.hound_positions[1], (1,-1)) ## update hound position
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 2 ##Place new hound position

            #3rd hound moves fl
            elif action_to_take == 4: 
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 0 ##Start the hound move
                env.hound_positions[2] = np.subtract(env.hound_positions[2], (1,1)) ## update hound position
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 3 ##Place new hound position
                        
            #3rd hound moves fr
            elif action_to_take == 5:
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 0 ##Start the hound move
                env.hound_positions[2] = np.subtract(env.hound_positions[2], (1,-1)) ## update hound position
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 3 ##Place new hound position

            #4th hound moves fl
            elif action_to_take == 6: 
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 0 ##Start the hound move
                env.hound_positions[3] = np.subtract(env.hound_positions[3], (1,1)) ## update hound position
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 4 ##Place new hound position
                        
            #4th hound moves fr
            elif action_to_take == 7:
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 0 ##Start the hound move
                env.hound_positions[3] = np.subtract(env.hound_positions[3], (1,-1)) ## update hound position
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 4 ##Place new hound position
        
        else: ##No Valid Move...
            pass

        return env.state

    #Performs Experience replay
    #This is a different implmentation currently than the fox model
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            #state = np.reshape(state, [1, LENGTH*LENGTH])---------------------------------This is for Dense NN, uncomment and comment out the 2 lines below if needed
            #next_state = np.reshape(next_state, [1, LENGTH*LENGTH])-----------------------This is for Dense NN, uncomment and comment out the 2 lines below if needed
            state = np.reshape(state, [1, 8,8,1]) #used for CNN 
            next_state = np.reshape(next_state, [1,8,8,1]) #used for CNN
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

#### This version of experience replay is for a DDQN, Not used currently
    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         state = np.reshape(state, [1, LENGTH*LENGTH])
    #         next_state = np.reshape(next_state, [1, LENGTH*LENGTH])            
    #         target = self.model.predict(state)
    #         if done:
    #             target[0][action] = reward
    #         else:
    #             # a = self.model.predict(next_state)[0]
    #             t = self.target_model.predict(next_state)[0]
    #             target[0][action] = reward + self.gamma * np.amax(t)
    #             # target[0][action] = reward + self.gamma * t[np.argmax(a)]
    #         self.model.fit(state, target, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
####


class Agent_Fox_Random: #This class is simply for a random fox, nothing really to see here.
    def __intit__(self):
        pass

    def set_role(self, role):
        self.role = role

    def act(self, env):
        actions_allowed = env.allowed_actions_fox_agent()
        action_to_take = actions_allowed[random.randint(0, len(actions_allowed)-1)]

        ##this moves the fox back (-1, -1) or what I call back left (bl)
        if action_to_take == 0:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (1,1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        ##this moves the fox back (-1, 1) or what I call back right (br)
        elif action_to_take == 1:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (1,-1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        ##this moves the fox back (1, -1) or what I call foward left (fl)
        elif action_to_take  == 2:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (-1,1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        ##this moves the fox back (1, 1) or what I call forward right (fr)
        elif action_to_take == 3:
            env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
            env.fox_position = np.subtract(env.fox_position, (-1,-1)) ## update fox position
            env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position

        return action_to_take


    def update(self, env):
        pass

    def update_state_history(self, s):
        pass

class Agent_Hound_Random: #This class is simply for a random hound, nothing to see here...
    def __init__(self):
        pass
    
    def set_role(self, role):
        self.role = role
    
    def act(self, env):
        actions_allowed = env.allowed_actions_hound_agent()

        #moves the hound forward left (-1,-1)
        if len(actions_allowed) == 0:
            pass

        else:  
            ##This protects against a no move scenario
            action_to_take = actions_allowed[random.randint(0, len(actions_allowed)-1)]

            #1st hound move fl
            if action_to_take == 0: 
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 0 ##Start the hound move
                env.hound_positions[0] = np.subtract(env.hound_positions[0], (1,1)) ## update hound position
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 1 ##Place new hound position
                        
            #1st hound moves fr
            elif action_to_take == 1:
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 0 ##Start the hound move
                env.hound_positions[0] = np.subtract(env.hound_positions[0], (1,-1)) ## update hound position
                env.state[env.hound_positions[0][0], env.hound_positions[0][1]] = 1 ##Place new hound position

            #2nd hound moves fl
            elif action_to_take == 2: 
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 0 ##Start the hound move
                env.hound_positions[1] = np.subtract(env.hound_positions[1], (1,1)) ## update hound position
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 2 ##Place new hound position
                        
            #2nd hound moves fr
            elif action_to_take == 3:
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 0 ##Start the hound move
                env.hound_positions[1] = np.subtract(env.hound_positions[1], (1,-1)) ## update hound position
                env.state[env.hound_positions[1][0], env.hound_positions[1][1]] = 2 ##Place new hound position

            #3rd hound moves fl
            elif action_to_take == 4: 
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 0 ##Start the hound move
                env.hound_positions[2] = np.subtract(env.hound_positions[2], (1,1)) ## update hound position
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 3 ##Place new hound position
                        
            #3rd hound moves fr
            elif action_to_take == 5:
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 0 ##Start the hound move
                env.hound_positions[2] = np.subtract(env.hound_positions[2], (1,-1)) ## update hound position
                env.state[env.hound_positions[2][0], env.hound_positions[2][1]] = 3 ##Place new hound position

            #4th hound moves fl
            elif action_to_take == 6: 
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 0 ##Start the hound move
                env.hound_positions[3] = np.subtract(env.hound_positions[3], (1,1)) ## update hound position
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 4 ##Place new hound position
                        
            #4th hound moves fr
            elif action_to_take == 7:
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 0 ##Start the hound move
                env.hound_positions[3] = np.subtract(env.hound_positions[3], (1,-1)) ## update hound position
                env.state[env.hound_positions[3][0], env.hound_positions[3][1]] = 4 ##Place new hound position
            
            return action_to_take
        
            
class Human: #Allows human to play against Agent
    def __init__(self):
        pass

    def set_role(self, role):
        self.role = role

    def act(self, env): #human can only be fox
        while True:
        # break if we make a legal move
            if self.role == 0:
                move = input("enter bl, br, fl, or fr to move")
    
                if move == "bl" and env.is_valid_fox_bl() == True:
                    env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
                    env.fox_position = np.subtract(env.fox_position, (1,1)) ## update fox position
                    env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position
                    break

                elif move == "br" and env.is_valid_fox_br() == True:
                    env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
                    env.fox_position = np.subtract(env.fox_position, (1,-1)) ## update fox position
                    env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position
                    break

                elif move == "fl" and env.is_valid_fox_fl() == True:
                    env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
                    env.fox_position = np.subtract(env.fox_position, (-1,1)) ## update fox position
                    env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position
                    break

                elif move == "fr" and env.is_valid_fox_fr() == True:
                    env.state[env.fox_position[0], env.fox_position[1]] = 0 ## Start the fox move
                    env.fox_position = np.subtract(env.fox_position, (-1,-1)) ## update fox position
                    env.state[env.fox_position[0], env.fox_position[1]] = -1 ## place new fox position
                    break
            
                else:
                    print("Invalid selection, please follow the instructions!")
            
            else:
                hound = int(input("Enter hound to move 1, 2, 3, or 4"))
                print(hound-1)
                move = input("enter fl or fr to move")

                if move == "fl" and env.is_valid_hound_move_bl(hound) == True:
                    env.state[env.hound_positions[hound-1][0], env.hound_positions[hound-1][1]] = 0 ##Start the hound move
                    env.hound_positions[hound-1] = np.subtract(env.hound_positions[hound-1], (1,1)) ## update hound position
                    env.state[env.hound_positions[hound-1][0], env.hound_positions[hound-1][1]] = hound ##Place new hound position
                    break

                elif move == "fr" and env.is_valid_hound_move_br(hound) == True:
                    env.state[env.hound_positions[hound-1][0], env.hound_positions[hound-1][1]] = 0 ##Start the hound move
                    env.hound_positions[hound-1] = np.subtract(env.hound_positions[hound-1], (1,-1)) ## update hound position
                    env.state[env.hound_positions[hound-1][0], env.hound_positions[hound-1][1]] = hound ##Place new hound position
                    break


                else:
                    print("Invalid selection, please follow the instructions!")


    def update(self, env):
        pass

    def update_state_history(self, s):
        pass

class Environment:
    def __init__(self):
        self.state = np.zeros((LENGTH, LENGTH))                         #this is the board
        self.fox = -1                                                   #represents the fox or player 1
        self.hounds = [1,2,3,4]                                         #represents the hounds 
        self.fox_position =  (0,0)                                      #keeps track of fox position on board
        self.hound_positions = [(0,0),(0,0),(0,0),(0,0)]                #keeps track of hounds positions on board
        self.winner = None                                              #identifies which palyer is the winner
        self.ended = False                                              #identifies winner at end of game
        self.turns = 0                                                  #logs which turn it is
        self.giveup = 0                                                 #Tracks if hounds are in a no-win state if = 3 then giveup
        self.prev_fox_position = (0,0)                                  #Logs previous fox position
        self.prev_distance = 0                                          #used for measuring hounds reward
        # Define action space 
        self.action_dict = {"bl": 0, "br": 1, "fl": 2, "fr": 3}
        self.action_coords = [(-1, -1), (-1, 1), (1, -1), (1, 1)] # translations
        self.hound_actions = {"h1fl": 0, "h1fr": 1,"h2fl": 2, "h2fr": 3,"h3fl": 4, "h3fr": 5,"h4fl": 6, "h4fr": 7}
        ########MOVE EXPLANATIONS###########
        # bl = back left, think of this as going from position (7,7) to (6,6)
        # br = back right, think of this as going from position (1,1) to (0,2)
        # fl = foward left, think of this as going from position (1,1) to (2,0)
        # fr = forward right, think of this as going from position (1,1)  to (2,2)

    #Checks if fox can move back left
    def is_valid_fox_bl(self):
        bl = np.subtract(self.fox_position,(1,1))
        if bl[0]!=-1 and bl[1] != -1 and  bl[0]!=LENGTH and bl[1]!=LENGTH and  self.state[bl[0], bl[1]] == 0:
            return True
        else:
            return False
    
    #Checks if fox can move back right+
    def is_valid_fox_br(self):
        br = np.subtract(self.fox_position,(1,-1))
        if br[0]!=-1 and br[1] != -1 and  br[0]!=LENGTH and br[1]!=LENGTH and  self.state[br[0], br[1]] == 0:
            return True
        else:
            return False

    #Checks if fox can move forward left
    def is_valid_fox_fl(self):
        fl = np.subtract(self.fox_position,(-1,1))
        if fl[0]!=-1 and fl[1] != -1 and  fl[0]!=LENGTH and fl[1]!=LENGTH and  self.state[fl[0], fl[1]] == 0:
            return True
        else:
            return False

    #Checks if fox can move forward right
    def is_valid_fox_fr(self):
        fr = np.add(self.fox_position,(1,1))
        if fr[0]!=-1 and fr[1] != -1 and  fr[0]!=LENGTH and fr[1]!=LENGTH and  self.state[fr[0], fr[1]] == 0:
            return True
        else:
            return False

    #Check if fox has any valid move (used for rewards and end game)
    def has_valid_fox_move(self):
        #calculate back left, back right, forward left, forward right
        bl = np.subtract(self.fox_position,(1,1))
        br = np.subtract(self.fox_position,(1,-1))
        fl = np.subtract(self.fox_position,(-1,1))
        fr = np.add(self.fox_position,(1,1))

        ##check fox can move backwards to the left(-1, -1)
        if bl[0]!=-1 and bl[1] != -1 and bl[0]!=LENGTH and bl[1]!=LENGTH and  self.state[bl[0], bl[1]] == 0:
            return True

        ##check fox can mvoe backwards to the right (-1, +1)
        elif br[0]!=-1 and br[1] != -1 and br[0]!=LENGTH and br[1]!=LENGTH and self.state[br[0], br[1]] == 0:
            return True

        ##check if fox can move forward to the left (+1, -1)
        elif fl[0]!=-1 and fl[1] != -1 and fl[0]!=LENGTH and fl[1]!=LENGTH and  self.state[fl[0], fl[1]] == 0:
            return True
            
        ##check if fox can move forward to the right (+1, +1)
        elif fr[0]!=-1 and fr[1] != -1 and fr[0]!=LENGTH and fr[1]!=LENGTH and  self.state[fr[0], fr[1]] == 0:
            return True

        ##No valid move to be made, game is over
        else:
            return False

    #Check if hound selected can move back left
    def is_valid_hound_move_bl(self, i):
        i -= 1 #convert the index down for proper indexing of hound 0,1,2,3 etc.
        bl = np.subtract(self.hound_positions[i],(1,1))
        ##check ifhound can move forward to left (-1, -1)
        if bl[0]!=-1 and bl[1] != -1 and bl[0]!=LENGTH and bl[1]!=LENGTH and  self.state[bl[0], bl[1]] == 0:
            return True
        else:
            return False

    #Check if hound selected can move back right
    def is_valid_hound_move_br(self, i):
        i -= 1 #convert the index down for proper indexing of hound 0,1,2,3 etc.
        br = np.subtract(self.hound_positions[i],(1,-1))
        ##No valid move for hound at this position
        if br[0]!=-1 and br[1] != -1 and br[0]!=LENGTH and br[1]!=LENGTH and  self.state[br[0], br[1]] == 0:
            return True

    #This function returns all the valid moves for the hounds
    def allowed_actions_hound_agent(self):
        actions_allowed = [] ##create a 3-tuple first pos is hound, last 2 are valid coordinates to move
        
        ##every poissbile hound move 
        h1fl = np.subtract(self.hound_positions[0],(1,1)) 
        h1fr = np.subtract(self.hound_positions[0],(1,-1))
        h2fl = np.subtract(self.hound_positions[1],(1,1)) 
        h2fr = np.subtract(self.hound_positions[1],(1,-1))
        h3fl = np.subtract(self.hound_positions[2],(1,1)) 
        h3fr = np.subtract(self.hound_positions[2],(1,-1))
        h4fl = np.subtract(self.hound_positions[3],(1,1)) 
        h4fr = np.subtract(self.hound_positions[3],(1,-1))
         
        #Checks to see if each mvoe is valid ---all of them----
        if h1fl[0] >= 0 and h1fl[0] < LENGTH and h1fl[1] >= 0 and h1fl[1] < LENGTH and self.state[h1fl[0], h1fl[1]] == 0:
            actions_allowed.append(self.hound_actions["h1fl"])
        
        if h1fr[0] >= 0 and h1fr[0] < LENGTH and h1fr[1] >= 0 and h1fr[1] < LENGTH and self.state[h1fr[0], h1fr[1]] == 0:
            actions_allowed.append(self.hound_actions["h1fr"])

        if h2fl[0] >= 0 and h2fl[0] < LENGTH and h2fl[1] >= 0 and h2fl[1] < LENGTH and self.state[h2fl[0], h2fl[1]] == 0:
            actions_allowed.append(self.hound_actions["h2fl"])
        
        if h2fr[0] >= 0 and h2fr[0] < LENGTH and h2fr[1] >= 0 and h2fr[1] < LENGTH and self.state[h2fr[0], h2fr[1]] == 0:
            actions_allowed.append(self.hound_actions["h2fr"])

        if h3fl[0] >= 0 and h3fl[0] < LENGTH and h3fl[1] >= 0 and h3fl[1] < LENGTH and self.state[h3fl[0], h3fl[1]] == 0:
            actions_allowed.append(self.hound_actions["h3fl"])
        
        if h3fr[0] >= 0 and h3fr[0] < LENGTH and h3fr[1] >= 0 and h3fr[1] < LENGTH and self.state[h3fr[0], h3fr[1]] == 0:
            actions_allowed.append(self.hound_actions["h3fr"])

        if h4fl[0] >= 0 and h4fl[0] < LENGTH and h4fl[1] >= 0 and h4fl[1] < LENGTH and self.state[h4fl[0], h4fl[1]] == 0:
            actions_allowed.append(self.hound_actions["h4fl"])
        
        if h4fr[0] >= 0 and h4fr[0] < LENGTH and h4fr[1] >= 0 and h4fr[1] < LENGTH and self.state[h4fr[0], h4fr[1]] == 0:
            actions_allowed.append(self.hound_actions["h4fr"])
        
        return actions_allowed

    #Returns all the allowed actions for the hound agent
    def allowed_actions_fox_agent(self):
        actions_allowed = []
        if self.is_valid_fox_bl():
            actions_allowed.append(self.action_dict["bl"])
        if self.is_valid_fox_br():
            actions_allowed.append(self.action_dict["br"])
        if self.is_valid_fox_fl():
            actions_allowed.append(self.action_dict["fl"])
        if self.is_valid_fox_fr():
            actions_allowed.append(self.action_dict["fr"])
        return actions_allowed

    #get the distance of all hounds to the fox
    def getHoundsDistance(self):
        d = 0
        x2 = self.fox_position[0]
        y2 = self.fox_position[1]

        for i in range(len(self.hound_positions)):
            x1 = self.hound_positions[i][0]
            y1 = self.hound_positions[i][1]

            temp = ((x2-x1)**2)+((y2-y1)**2)
            d += sqrt(temp)
        
        return d

    #Returns the hound agents reward
    def get_hound_reward(self, R):
        #Get hounds total distance
        #distance = self.getHoundsDistance()

        #hounds have won
        if env.has_valid_fox_move() == False:
            R += 100    
        #hounds have lost
        if env.fox_position[0] == LENGTH -1:
            R -= 100

        

        return R

    #Returns the fox agents reward
    def get_fox_reward(self, R):
        
        #fox wins
        if env.fox_position[0] == LENGTH - 1:
            R += 10 

        #hounds won
        if env.has_valid_fox_move() == False:
            R -= 10  
        
        #reward fox for moving forward
        if env.fox_position[0] > env.prev_fox_position[0]:
            R+=.25
    

        #reward fox for moving forward
        if env.fox_position[0] < env.prev_fox_position[0]:
            R-=.50

        return R

    #Used to reset the game after the game is over
    def reset_state(self):
        self.state = np.zeros((LENGTH, LENGTH))
        self.state[0,0] = -1
        self.state[LENGTH-1,LENGTH-1] = 1
        self.state[LENGTH-1,LENGTH-3] = 2
        self.state[LENGTH-1,LENGTH-5] = 3
        self.state[LENGTH-1,LENGTH-7] = 4
        #set the positions for the hounds back to default state
        self.fox_position = (0,0)
        self.hound_positions[0] = (LENGTH-1, LENGTH-1)
        self.hound_positions[1] = (LENGTH-1, LENGTH-3)
        self.hound_positions[2] = (LENGTH-1, LENGTH-5)
        self.hound_positions[3] = (LENGTH-1, LENGTH-7)
        self.winner = None
        self.ended = False
        self.turns = 0
        return self.state

    #This function is not used in the main program
    #It is used simply when training the fox on an empty board
    def reset_fox_only_state(self):
        self.state = np.zeros((LENGTH, LENGTH))
        self.state[0,0] = -1 ##sets fox
        #set the positions for the hounds back to default state
        self.fox_position = (0,0)
        self.winner = None
        self.ended = False
        self.turns = 0
        return self.state

    #draws the board in GUI Format
    def render(self):
        pg.init()                                               #Initialize board
        pg.display.set_caption('ML With Fox and Hounds')
        colors = [(238,238,210), (118,150,86)]                  # Set up colors [light green, Olive Green]
        n = LENGTH                                              # The size of rows and columns of the board
        surface_sz = 480                                        # Proposed physical surface size.
        sq_sz = surface_sz // n                                 # sq_sz is length of a square.
        surface_sz = n * sq_sz                                  # Adjust to exactly fit n squares.

        # Create the surface of (width, height), and its window.
        surface = pg.display.set_mode((surface_sz, surface_sz))

        fox = pg.image.load("Images/blackk.png")
        hound = pg.image.load("Images/whiteq.png")

        # Use an extra offset to center the piece in its square.
        piece_offset = (sq_sz-fox.get_width()) // 2

        # Draw a fresh background (a blank chess board)
        for row in range(n):                                        # Draw each row of the board.
            c_indx = row % 2                                        # Alternate starting color
            for col in range(n):                                    # Run through cols drawing squares
                the_square = (col*sq_sz, row*sq_sz, sq_sz, sq_sz)
                surface.fill(colors[c_indx], the_square)
                # Now flip the color index for the next square
                c_indx = (c_indx + 1) % 2

        # Draw the board
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.state[i][j] == -1:
                    surface.blit(fox, (j*sq_sz+piece_offset,i*sq_sz+piece_offset))
                if self.state[i][j] == 1:
                    surface.blit(hound, (j*sq_sz+piece_offset,i*sq_sz+piece_offset))
                if self.state[i][j] == 2:
                    surface.blit(hound, (j*sq_sz+piece_offset,i*sq_sz+piece_offset))
                if self.state[i][j] == 3:
                    surface.blit(hound, (j*sq_sz+piece_offset,i*sq_sz+piece_offset))
                if self.state[i][j] == 4:
                    surface.blit(hound, (j*sq_sz+piece_offset,i*sq_sz+piece_offset))
        
        # render the board to the screen
        pg.display.flip()

    #Draws the board in terminal
    def draw_board(self):
        counter = 0
        for i in range(LENGTH):
            counter+=1
            for j in range(LENGTH):
                if self.state[i,j] == -1:
                    print(' F ', end='')
                    counter+=1
                elif self.state[i,j] == 0 and counter % 2 == 0:
                    print(' * ', end='')
                    counter+=1
                elif self.state[i,j] == 0 and counter % 2 != 0:
                    print(' . ', end='')
                    counter+=1
                elif self.state[i,j] == 1:
                    print(' 1 ', end='')
                    counter+=1
                elif self.state[i,j] == 2:
                    print(' 2 ', end='')
                    counter+=1
                elif self.state[i,j] == 3:
                    print(' 3 ', end='')
                    counter+=1
                elif self.state[i,j] == 4:
                    print(' 4 ', end='')
                    counter+=1
            print('')
  
    #Checks if game is over returns True or False
    def game_over(self):
        #check if hounds won
        if self.has_valid_fox_move() == False and self.fox_position[0]!= LENGTH-1:
            self.ended = True
            self.winner = "hounds"
            print("The Hounds have Won!")
            self.draw_board()
            self.giveup = 0
            return True

        #Check if fox won
        elif self.fox_position[0] == LENGTH-1:
            self.ended = True
            self.winner = "fox"
            print("The Fox has Won!")
            self.draw_board()
            self.giveup = 0
            return True

        elif self.giveup == 3:
            self.ended = True
            self.winner = "fox"
            print("The Fox has Won!")
            self.draw_board()
            self.giveup = 0
            return True

        #there is no winner yet!
        else:
            return False


if __name__ == '__main__':
    env = Environment()
    env.reset_state()
    a = int(input("enter: \n 1 - TRAIN AI Hound vs Random Fox \n 2 - TRAIN AI Fox vs Random Hound  \n 3 - TRAIN AI Fox vs AI Hounds \n 4 - Human vs AI Hounds \n 5 - Human vs AI Fox \n"))
    batch_size = 32
    total_reward = 0
    average_reward = 0

    #Train AI Hound vs Random Fox
    if a == 1:
        p1 = Agent_Fox_Random()
        p2 = Agent_Hound()
    
    #Train AI Fox vs Random Hound
    elif a == 2:
        p1 = Agent_Fox()
        p2 = Agent_Hound_Random()

    #AI Fox vs AI Hound
    elif a == 3:
        p1 = Agent_Fox()
        p2 = Agent_Hound()

    #Human vs AI Hound
    elif a == 4:
        p1 = Human()
        p2 = Agent_Hound()
    
    #Human vs AI Fox
    elif a == 5:
        p1 = Agent_Fox()
        p2 = Human()

    #set the players roles
    #Fox will always be P1 but choice is between human, random, or AI agent
    #Hound will always be P2 but choice is between human random or AI agent
    p1.set_role(0)
    p2.set_role(1)

   
    ##Training block for AI hound vs random fox
    if a == 1:
        #load the hound model prior to the training
        if os.path.isfile('Hound_Agent.model'):
            print("Loading model...")
            p2.set_model(tf.keras.models.load_model('Hound_Agent.model'))
        else:
            print("No Model File found.... starting a new model called: Hound_Agent.model")
            p2.set_model(None)

        #load the hound target model prior to the training -- This is used for the DDQN which is not being used currently
        # if os.path.isfile('Hound_Target.model'):
        #     print("Loading model...")
        #     p2.set_target_model(tf.keras.models.load_model('Hound_target.model'))
        # else:
        #     print("No Model File found.... starting a new target model called: Hound_Target.model")
            p2.set_target_model(None)

        for e in range(EPISODES):
            #reset state at the start of the game
            state = env.reset_state()
            done = False
            current_player = None
            reward_hound = 0
            turn_count = 0
            loss =  0
            curr_state = None
            next_state = None

            ##save the model every 10 episodes
            if e % 10 == 0 and e!=0:
                print("saving models....")
                p2.model.save("Hound_Agent.model")
                #p2.target_model.save("Hound_Target.model")
            
            while not done:
                env.render()#Draw board in GUI Form
                turn_count += 1
                if verbose == True:
                    print("Turn: {}, Fox wins: {}, Hound wins: {}, current game: {},Reward: {}, Avg. Reward:{}, Hound Win %: {}%".format(turn_count, games_won_fox, games_won_hound, e+1, reward_hound,average_reward_hound ,(games_won_hound/(e+1))*100))
                    print('=======================')
                    env.draw_board() #Draw the board on CMD line
                    print('=======================')
                    

                #set the current player
                if current_player == p1:
                    current_player = p2
                else:
                    current_player = p1

                #decide action for fox or hound
                if current_player == p1: #Fox's move
                    curr_state = cp.deepcopy(env.state)
                    action = p1.act(env) ##this changes the current state of the board

                    #State reward and done
                    done = env.game_over()

                    if done:
                        ##These are Used for the DDQN##
                        #update target model
                        #p2.update_target_model()
                        ##^^^^^^^^^^^^^^^^^^^^^^^^^^##

                        #update rewards
                        total_reward_hound += reward_hound
                        #calculate average rewards
                        average_reward_hound = total_reward_hound/(e+1)
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        print("episode: {}/{}, e: {:.2}, fox wins: {}, Reward Hound: {} hound wins: {}".format(e+1, EPISODES, p2.epsilon, games_won_fox, reward_hound,games_won_hound))
                        break
                    if len(p2.memory) > batch_size:
                        p2.replay(batch_size)

                else: # Hound AI Decision Block
                    curr_state = cp.deepcopy(env.state)
                    action, reward_hound = p2.act(reward_hound, curr_state, env)
                    ##Take Action
                    next_state = p2.take_action(action, env)
                    reward_hound = env.get_hound_reward(reward_hound)
                    done = env.game_over()
                    p2.remember(curr_state, action, reward_hound, next_state, done)

                    if done:
                        #update target model
                        #p2.update_target_model()
                        #update rewards
                        total_reward_hound += reward_hound
                        #calculate average rewards
                        average_reward_hound = total_reward_hound/(e+1)
                        #update wins
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        print("episode: {}/{}, e: {:.2}, fox wins: {}, Reward Hound: {} hound wins: {}".format(e+1, EPISODES, p2.epsilon, games_won_fox, reward_hound,games_won_hound))
                        break
                    if len(p2.memory) > batch_size:
                        p2.replay(batch_size)

        # train the agent with the experience of the episode
        p2.replay(32)
 
    #training block for AI fox vs random hound
    elif a == 2:
        ##load the model prior to the training -- Fox
        if os.path.isfile('Fox_Agent.model'):
            print("Loading Fox model...")
            p1.set_model(tf.keras.models.load_model('Fox_Agent.model'))
        else:
            print("No Model File found.... starting a new model called: Fox_Agent.model")
            p1.set_model(None)

        for e in range(EPISODES):
            #reset state at the start of the game
            state = env.reset_state()
            done = False
            current_player = None
            reward_fox = 0
            turn_count = 0

            ##save the model
            if e % 10 == 0 and e!=0:
                print("saving models....")
                p1.model.save("Fox_Agent.model")

                
            while not done:
                env.render()    #Draw board in GUI Form
                turn_count += 1
                if verbose == True:
                    print("Turn: {}, F_wins: {}, F_Reward: {:.2f}, Avg. F_Reward:{:.2f}, H_wins: {}, Current game: {}".format(turn_count, games_won_fox, reward_fox, average_reward_fox, games_won_hound, e+1))
                    print('=======================')
                    env.draw_board()
                    print('=======================')

                #set the current player
                if current_player == p1:
                    current_player = p2
                else:
                    current_player = p1

                #decide action for fox or hound 
                if current_player == p1:
                    curr_state = env.state
                    action, reward_fox = p1.act(reward_fox, curr_state, env)

                    #State reward and done
                    next_state = p1.take_action(action, env)
                    reward_fox = env.get_fox_reward(reward_fox)
                    done = env.game_over()
                    p1.remember(curr_state, action, reward_fox, next_state, done)

                    if done:
                        #update rewards
                        total_reward_fox += reward_fox
                        #calculate average rewards
                        average_reward_fox = total_reward_fox/(e+1)
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        print("episode: {}/{}, e: {:.2}, Reward Fox: {}, fox wins: {}, hound wins: {}".format(e, EPISODES, p1.epsilon, reward_fox, games_won_fox, games_won_hound))
                        break
                    if len(p1.memory) > batch_size:
                        p1.replay(batch_size)

                else: # Random Hound decision block
                    curr_state = env.state
                    action = p2.act(env)
                    done = env.game_over()

                    if done:
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        print("episode: {}/{}, e: {:.2}, Reward Fox: {}, fox wins: {}, hound wins: {}".format(e, EPISODES, p1.epsilon, reward_fox, games_won_fox, games_won_hound))
                        break

        # train the agent with the experience of the episode
        p1.replay(32)

    ##Train AI Fox vs AI Hound
    elif a == 3:
        ##load the model prior to the training -- Hound
        if os.path.isfile('Hound_Agent.model'):
            print("Loading Hound model...")
            p2.set_model(tf.keras.models.load_model('Hound_Agent.model'))
        else:
            print("No Model File found.... starting a new model called: Hound_Agent.model")
            p2.set_model(None)

        ##load the model prior to the training -- Fox
        if os.path.isfile('Fox_Agent.model'):
            print("Loading Fox model...")
            p1.set_model(tf.keras.models.load_model('Fox_Agent.model'))
        else:
            print("No Model File found.... starting a new model called: Fox_Agent.model")
            p1.set_model(None)

        for e in range(EPISODES):
            #reset state at the start of the game
            state = env.reset_state()
            done = False
            current_player = None
            reward_hound = 0
            reward_fox = 0
            turn_count = 0

            ##save the model
            if e % 10 == 0 and e!=0:
                print("saving models....")
                p1.model.save("Fox_Agent.model")
                p2.model.save("Hound_Agent.model")

                
            while not done:
                env.prev_distance = env.getHoundsDistance() #set distance right away
                env.render()    #Draw board in GUI Form
                turn_count += 1
                if verbose == True:
                    print("Turn: {}, F_wins: {}, F_Reward: {:.2f}, Avg. F_Reward:{:.2f}, H_wins: {}, H_Reward: {:.2f},Avg H_Reward:{:.2f}, current game: {}".format(turn_count, games_won_fox, reward_fox, average_reward_fox, games_won_hound, reward_hound, average_reward_hound, e+1))
                    print('=======================')
                    env.draw_board()
                    print('=======================')

                #set the current player
                if current_player == p1:
                    current_player = p2
                else:
                    current_player = p1

                #decide action for fox or hound 
                if current_player == p1:
                    curr_state = cp.deepcopy(env.state)
                    action, reward_fox = p1.act(reward_fox, curr_state, env)

                    #State reward and done
                    next_state = p1.take_action(action, env)
                    reward_fox = env.get_fox_reward(reward_fox)
                    done = env.game_over()
                    p1.remember(curr_state, action, reward_fox, next_state, done)

                    if done:
                        #update rewards
                        total_reward_hound += reward_hound
                        total_reward_fox += reward_fox
                        #calculate average rewards
                        average_reward_fox = total_reward_fox/(e+1)
                        average_reward_hound = total_reward_hound/(e+1)
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        print("episode: {}/{}, e: {:.2}, Reward Fox: {}, fox wins: {}, Reward Hound: {} hound wins: {}".format(e, EPISODES, p1.epsilon, reward_fox, games_won_fox, reward_hound,games_won_hound))
                        break
                    if len(p1.memory) > batch_size:
                        p1.replay(batch_size)

                else: # Hound AI Decision Block
                    curr_state = cp.deepcopy(env.state)
                    action, reward_hound = p2.act(reward_hound, curr_state, env)
                    ##Take Action
                    next_state = p2.take_action(action, env)
                    reward_hound = env.get_hound_reward(reward_hound)
                    done = env.game_over()
                    p2.remember(curr_state, action, reward_hound, next_state, done)

                    #get previous distance for next loop for hound reward
                    env.prev_distance = env.getHoundsDistance()

                    if done:
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        print("episode: {}/{}, e: {:.2}, Reward Fox: {}, fox wins: {}, Reward Hound: {} hound wins: {}".format(e, EPISODES, p1.epsilon, reward_fox, games_won_fox, reward_hound,games_won_hound))
                        break
                    if len(p2.memory) > batch_size:
                        p2.replay(batch_size)

        # train the agent with the experience of the episode
        p1.replay(32)
        p2.replay(32)

    ##Human vs AI Hound block
    elif a == 4:
        endGame = 0     #Controls if the user ends the game
        e = 0

        #load the hound model prior to the training
        if os.path.isfile('Hound_Agent.model'):
            print("Loading model...")
            p2.set_model(tf.keras.models.load_model('Hound_Agent.model'))
        else:
            print("No Model File found.... starting a new model called: Hound_Agent.model")
            p2.set_model(None)

        while endGame != 1:
            #reset state at the start of the game
            state = env.reset_state()
            done = False
            current_player = None
            reward_hound = 0
            turn_count = 0
            loss =  0
            curr_state = None
            next_state = None 
            
            while not done:
                env.render()#Draw board in GUI Form
                turn_count += 1
                if verbose == True:
                    print("Turn: {}, User wins: {}, AI Hound wins: {}, current game: {}".format(turn_count, games_won_fox, games_won_hound, e+1))
                    print('=======================')
                    env.draw_board() #Draw the board on CMD line
                    print('=======================')
                    

                #set the current player
                if current_player == p1:
                    current_player = p2
                else:
                    current_player = p1

                #decide action for fox or hound
                if current_player == p1: #Fox's move
                    action = p1.act(env) ##this changes the current state of the board

                    #check if game over
                    done = env.game_over()

                    if done:
                        #render final board state
                        env.render()#Draw board in GUI Form
                        #update game counter
                        e+=1
                        #update wins
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        #Display results
                        print("Turn: {}, User wins: {}, AI Hound wins: {}, current game: {}".format(turn_count, games_won_fox, games_won_hound, e+1))
                        
                        #Prompt user to play again
                        endGame = int(input("enter: \n 0 - To play again \n 1 - To END The game  \n"))
                        break


                else: # Hound AI Decision Block
                    curr_state = cp.deepcopy(env.state)
                    action, reward_hound = p2.act(reward_hound, curr_state, env)
                    ##Take Action
                    next_state = p2.take_action(action, env)
                    reward_hound = env.get_hound_reward(reward_hound)
                    done = env.game_over()

                    if done:
                        #render final board state
                        env.render()#Draw board in GUI Form
                        #update game counter
                        e+=1
                        #update wins
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1
                        #display Results
                        print("Turn: {}, User wins: {}, AI Hound wins: {}, current game: {}".format(turn_count, games_won_fox, games_won_hound, e+1))
                        
                        #Prompt user to play again
                        endGame = int(input("enter: \n 0 - To play again \n 1 - To END The game  \n"))
                        break

    #Human vs AI Fox
    elif a == 5:
        endGame = 0     #Controls if the user ends the game
        e = 0
        ##load the model prior to the training -- Fox
        if os.path.isfile('Fox_Agent.model'):
            print("Loading Fox model...")
            p1.set_model(tf.keras.models.load_model('Fox_Agent.model'))
        else:
            print("No Model File found.... starting a new model called: Fox_Agent.model")
            p1.set_model(None)

        while endGame != 1:
            #reset state at the start of the game
            state = env.reset_state()
            done = False
            current_player = None
            reward_fox = 0
            turn_count = 0

            ##save the model
            if e % 10 == 0 and e!=0:
                print("saving models....")
                p1.model.save("Fox_Agent.model")

                
            while not done:
                env.render()    #Draw board in GUI Form
                turn_count += 1
                if verbose == True:
                    print("Turn: {}, F_wins: {}, F_Reward: {:.2f}, Avg. F_Reward:{:.2f}, H_wins: {}, Current game: {}".format(turn_count, games_won_fox, reward_fox, average_reward_fox, games_won_hound, e+1))
                    print('=======================')
                    env.draw_board()
                    print('=======================')

                #set the current player
                if current_player == p1:
                    current_player = p2
                else:
                    current_player = p1

                #decide action for fox or hound 
                if current_player == p1:
                    curr_state = env.state
                    action, reward_fox = p1.act(reward_fox, curr_state, env)

                    #State reward and done
                    next_state = p1.take_action(action, env)
                    reward_fox = env.get_fox_reward(reward_fox)
                    done = env.game_over()
                    p1.remember(curr_state, action, reward_fox, next_state, done)

                    if done:
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        #Display results
                        print("Turn: {}, AI Hound wins: {}, User wins: {}, current game: {}".format(turn_count, games_won_fox, games_won_hound, e+1))
                        
                        #Prompt user to play again
                        endGame = int(input("enter: \n 0 - To play again \n 1 - To END The game  \n"))
                        break

                else: # Random Hound decision block
                    curr_state = env.state
                    action = p2.act(env)
                    done = env.game_over()

                    if done:
                        #render final board state
                        env.render()#Draw board in GUI Form
                        #update game counter
                        e+=1
                        #update wins
                        if env.winner == "fox":
                            games_won_fox += 1
                        else:
                            games_won_hound += 1

                        #Display results
                        print("Turn: {}, Hound Wins: {}, User Wins: {}, current game: {}".format(turn_count, games_won_fox, games_won_hound, e+1))
                        
                        #Prompt user to play again
                        endGame = int(input("enter: \n 0 - To play again \n 1 - To END The game  \n"))
                        break

   

 

