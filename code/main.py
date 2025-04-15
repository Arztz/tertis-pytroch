import argparse
import datetime
import itertools
import random
from ddqn import DQN
from experience_replay import ReplayMemory
import psutil
from setting import *
from sys import exit
from game import Game
from score import Score
from preview import Preview
from random import choice
from os.path import join
import os
import yaml
import torch
import torch.nn as nn
import matplotlib
from matplotlib import pyplot as plt
import graph
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
g = graph.ShowGraph()
torch.set_num_threads(22)
device = "cuda" if torch.cuda.is_available() else "cpu"
class Main:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.pretrained_model = hyperparameters.get('pretrained_model', None)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')
        #general
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Tertis')

        #shapes
        self.next_shapes = [choice(list(TETROMINOS.keys())) for shape in range(3)]

        #component
        self.game = Game(self.get_next_shape,self.update_score)
        self.score = Score()
        self.preview = Preview()

        #audio
        # self.music = pygame.mixer.Sound(join('sound','music.wav'))
        # self.music.set_volume(0.05)
        # self.music.play(-1)
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states,actions,new_states,rewards,terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)   
        new_states = torch.stack(new_states).to(device) 
        rewards = torch.stack(rewards).to(device)   
        terminations = torch.as_tensor(terminations).float().to(device)
        # truncates = torch.as_tensor(terminations).float().to(device)
        with torch.no_grad():

            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_score(self,lines,score,level):
        self.score.lines = lines
        self.score.score = score
        self.score.level = level

    def get_next_shape(self):
        next_shape = self.next_shapes.pop(0)
        self.next_shapes.append(choice(list(TETROMINOS.keys())))
        return next_shape
    
    def log_message(self,str):
        print(str)
        with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(str+ '\n') 

    def close(self):
        pygame.quit()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            #display
            self.display_surface.fill(GRAY)
            # print("1")
            self.game.run()
            self.score.run()
            self.preview.run(self.next_shapes)
            #updating the game
            pygame.display.update()
            self.clock.tick()


    def train(self,is_training=True):
        start_time = datetime.datetime.now().strftime(DATE_FORMAT)
        last_graph_update_time = start_time
        
        start_time = datetime.datetime.now()
        last_graph_update_time = start_time

        if is_training:
            self.log_message(f"DDQN Start time: {start_time}")

        env = self.game
        reward_per_episode = []
        epsilon_history = []

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        print(f'state: {num_states}  action: {num_actions}')
        policy_dqn = DQN(num_states,num_actions,self.fc1_nodes,self.enable_dueling_dqn ).to(device)
        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):

                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
                print(f"Loaded pretrained model from {self.pretrained_model}")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states,num_actions,self.fc1_nodes,self.enable_dueling_dqn  ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            best_reward = -9999999
        else:
            # Load the model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        #  Training loop
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.as_tensor(state,dtype=torch.float,device=device)

            terminated = False
            episode_reward = 0.0
            max_timesteps = 30000
            steps = 0
            self.score = Score()
            self.preview = Preview()
            done = False
            while (not done and episode_reward < self.stop_on_reward):
                steps += 1

                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.as_tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        
                # Processing:
                new_state, reward, terminated = env.run(action.item())
                # print(action)
                # print(f"Terminated: {terminated}")
                
                reward += 0.01
                # truncated = steps >= max_timesteps
                episode_reward += reward

                new_state = torch.as_tensor(new_state,dtype=torch.float,device=device)
                reward = torch.as_tensor(reward,dtype=torch.float,device=device)
                if is_training:
                    memory.append((state.detach(),action.detach() if action.requires_grad else action,new_state.detach(),reward.detach(),terminated))#,truncated))
                    step_count += 1
                state = new_state
                
                self.score.run()
                self.preview.run(self.next_shapes)
                pygame.display.update()
                self.clock.tick()
                done = terminated# or truncated 
                if done:
                    state = env.reset()
                    
            reward_per_episode.append(episode_reward)    
            # print(f"Episode: {episode}")
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_reward} at episode {episode}"
                    print(f"Memory used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    g.save_graph(reward_per_episode, epsilon_history,file=self.GRAPH_FILE)
                    last_graph_update_time = current_time
                if len(memory) > self.mini_batch_size:
                                            #sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
                    epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
        
if __name__ == "__main__":
    # main.run()
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args = parser.parse_args()

    dql = Main(hyperparameters_set=args.hyperparameters)
    if args.train:
        dql.train(is_training=True)
    else:
        dql.train(is_training=False)