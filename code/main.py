import argparse
import datetime
import itertools
import random
from hybrid_cnn import HybridCNN
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
from graph_utils import ShowGraph
from setting import *
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
# g = graph.ShowGraph()
g = ShowGraph(log_dir=f'runs/{datetime.datetime.now()}')
torch.set_num_threads(22)
device = "cuda" if torch.cuda.is_available() else "cpu"
class Main:
    def __init__(self, hyperparameters_set,render=False):
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

        self.render = render
        #general
        if render:
            pygame.init()
            self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
            pygame.display.set_caption('Tertis')
        self.clock = pygame.time.Clock()

        #shapes
        self.next_shapes = [choice(list(TETROMINOS.keys())) for shape in range(3)]

        #component
        if render:
            self.game = Game(self.get_next_shape,self.update_score,render)
            self.score = Score()
            self.preview = Preview()
        else:
            self.game = Game(self.get_next_shape,None,False)


        #audio
        # self.music = pygame.mixer.Sound(join('sound','music.wav'))
        # self.music.set_volume(0.05)
        # self.music.play(-1)
    def optimize(self, mini_batch, policy_dqn, target_dqn,episode):
        pic,states,actions,new_pic,new_states,rewards,terminations = zip(*mini_batch)
        pic = torch.cat(pic, dim=0).to(device)
        # print("pic.shape =", pic.shape)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)   
        new_pic = torch.cat(new_pic, dim=0).to(device)
        # print("new_pic.shape =", pic.shape)
        new_states = torch.stack(new_states).to(device) 
        rewards = torch.stack(rewards).to(device)   
        terminations = torch.as_tensor(terminations).float().to(device)
        # truncates = torch.as_tensor(terminations).float().to(device)
        with torch.no_grad():

            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(pic,new_states).argmax(dim=1)
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_pic,new_states,episode).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_pic,new_states).max(dim=1)[0]


        current_q = policy_dqn(pic,states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

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


    def train(self,is_training=True,render=False):
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
        policy_dqn = HybridCNN(num_actions,33,self.enable_dueling_dqn ).to(device)
        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):

                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
                print(f"Loaded pretrained model from {self.pretrained_model}")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = HybridCNN(num_actions,33,self.enable_dueling_dqn).to(device)
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
            pic,state, _ = env.reset()
            pic = torch.as_tensor(pic,dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
            # print("pic", pic.shape)      # → (1, 1, 20, 10)
            # print("state", state.shape)  # → (1, 33)
            state = torch.as_tensor(state,dtype=torch.float,device=device).unsqueeze(0)

            terminated = False
            episode_reward = 0.0
            max_timesteps = 30000
            steps = 0
            # self.score = Score()
            # self.preview = Preview()
            done = False
            while (not done and episode_reward < self.stop_on_reward):
                steps += 1

                # print(f"Step {steps}")
                
                    
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.as_tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(pic,state.unsqueeze(dim=0)).squeeze().argmax()
                        
                # Processing:
                new_pic,new_state, reward, terminated ,score= env.run(action.item())
                # print(action)
                # print(f"Terminated: {terminated}")
                # reward += 0.01
                # truncated = steps >= max_timesteps
                episode_reward += reward
                new_pic = torch.as_tensor(new_pic,dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
                new_state = torch.as_tensor(new_state,dtype=torch.float,device=device).unsqueeze(0)
                # print("new_pic", new_pic.shape)      # → (1, 1, 20, 10)
                # print("new_state", new_state.shape)  # → (1, 33)
                reward = torch.as_tensor(reward,dtype=torch.float,device=device)
                if is_training:
                    memory.append((pic.detach(),state.detach(),action.detach() if action.requires_grad else action,new_pic.detach(),new_state.detach(),reward.detach(),terminated))#,truncated))
                    step_count += 1
                state = new_state
                if render:
                    self.score.run()
                    self.preview.run(self.next_shapes)
                    pygame.display.update()
                self.clock.tick()
                done = terminated# or truncated 
                if done:
                    state = env.reset()
            
            reward_per_episode.append(episode_reward)    
            # print(f"Episode {episode} Reward {episode_reward} Score: {score}")
            if episode % 50 ==0:
                pic_data = new_pic.squeeze().cpu().numpy().astype(int)

                # print จากล่างขึ้นบน
                for row in pic_data:
                    print(''.join(str(cell) for cell in row))

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

                if len(memory) > self.mini_batch_size:
                                            #sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn,episode=episode)
                    epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
                    epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    # g.save_graph(reward_per_episode, epsilon_history,file=self.GRAPH_FILE)
                    g.log(reward_per_episode, epsilon)
                    last_graph_update_time = current_time 
        g.close()
if __name__ == "__main__":
    # main.run()
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args = parser.parse_args()

    dql = Main(hyperparameters_set=args.hyperparameters,render=RENDER)
    try:
        if args.train:
            dql.train(is_training=True,render=RENDER)
        else:
            dql.train(is_training=False,render=RENDER)
    finally:
        
        g.close()