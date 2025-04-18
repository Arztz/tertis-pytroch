import random
from setting import *
from times import Timer
# from sys import exit
from os.path import join
import numpy as np
from gymnasium import spaces
shape_names = list(TETROMINOS.keys())
step=0
class Game:
    def __init__(self,get_next_shape,update_score,render):
        
        #general
        self.render = render

        if render:
            self.surface = pygame.Surface((GAME_WIDTH,GAME_HEIGHT))
            self.display_surface = pygame.display.get_surface()
            self.rect = self.surface.get_rect(topleft = (PADDING,PADDING))
            self.line_surface = self.surface.copy()
            self.line_surface.fill((0,255,0))
            self.line_surface.set_colorkey((0,255,0))
            self.line_surface.set_alpha(120)
        else:
            self.surface = None
            self.display_surface = None
            self.rect = None
            self.line_surface = None

        self.sprites = pygame.sprite.Group()
        
        self.get_next_shape = get_next_shape
        self.update_score = update_score
        #lines


        
        #timer
        self.down_speed = UPDATE_START_SPEED
        self.down_pressed = False
        self.timers = {
            'vertical move': Timer(self.down_speed,True,self.move_down),
            'horizontal move':  Timer(MOVE_WAIT_TIME),
            'rotate':  Timer(ROTATE_WAIT_TIME)
        }
        self.timers['vertical move'].activate()

        #sound
        # self.music = pygame.mixer.Sound(join('sound','landing.wav'))
        # self.music.set_volume(0.05)

        #AI
        # field_low = np.zeros(ROWS * COLUMNS, dtype=np.float32)
        # field_high = np.ones(ROWS * COLUMNS, dtype=np.float32)
        #5
        extra_low = np.array([1, 0, 0, 0.0, 0.0], dtype=np.float32)
        extra_high = np.array([99, 99999, 999, 600, 600], dtype=np.float32)

        #7x4=28
        shape_low = np.zeros(7, dtype=np.float32)
        shape_high = np.ones(7, dtype=np.float32)

        preview1_low = np.zeros(7, dtype=np.float32)
        preview1_high = np.ones(7, dtype=np.float32)

        preview2_low = np.zeros(7, dtype=np.float32)
        preview2_high = np.ones(7, dtype=np.float32)

        preview3_low = np.zeros(7, dtype=np.float32)
        preview3_high = np.ones(7, dtype=np.float32)

        low = np.concatenate([shape_low,preview1_low,preview2_low,preview3_low, extra_low])
        high = np.concatenate([shape_high,preview1_high,preview2_high,preview3_high, extra_high])
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )       
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.field_data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        self.data = self.field_data
        self.sprites.empty()

        self.current_level = 1
        self.current_score = 0
        self.current_lines = 0
        self.reward = 0

        self.down_speed = UPDATE_START_SPEED
        self.down_speed_faster = self.down_speed * 0.1
        self.timers['vertical move'].duration = self.down_speed
        self.timers['vertical move'].activate()

        self.tetromino = Tetromino(
            random.choice(list(TETROMINOS.keys())),
            self.sprites,
            self.create_new_tetromino,
            self.field_data
        )
        self.terminated = False
        pic, obs = self._get_observation()
        return pic, obs, {}
    def shape_to_bin(self,shape):
        shape_index = shape_names.index(shape)
        shape_one_hot = np.zeros(7, dtype=np.float32)
        
        shape_one_hot[shape_index] = 1.0
        return shape_one_hot
    
    def _get_observation(self):
        
        self.data = self.get_data()
        pic = np.array(self.data, dtype=np.float32)

        current_shape = self.shape_to_bin(self.tetromino.shape)
        preview1 = self.shape_to_bin(self.tetromino.shape)
        preview2 = self.shape_to_bin(self.tetromino.shape)
        preview3 = self.shape_to_bin(self.tetromino.shape)
        extra_info = np.array([
                            self.current_level,
                            self.current_score,
                            self.current_lines,
                            self.down_speed,
                            self.down_speed_faster
                        ], dtype=np.float32)  
        
        observation = np.concatenate([current_shape,preview1,preview2,preview3, extra_info])
        # print(observation)
        return pic,observation

    def calculate_score(self,num_lines,render=False):
        self.current_lines += num_lines
        self.current_score += SCORE_DATA[num_lines] * self.current_level
        self.reward = 10 * num_lines
        if self.current_lines / 10 > self.current_level:
            self.current_level += 1
            self.down_speed *= 0.75
            self.down_speed_faster = self.down_speed * 0.3
            self.timers['vertical move'].duration = self.down_speed
        if render:
            self.update_score(self.current_lines,self.current_score,self.current_level)


    def check_game_over(self):
        for block in self.tetromino.blocks:
            if block.pos.y < 0 :
                return True
        return False

    def create_new_tetromino(self):
        # self.music.play()
        self.terminated = self.check_game_over()
        self.check_finished_rows()
        self.tetromino = Tetromino(
            self.get_next_shape(),
            self.sprites,
            self.create_new_tetromino,
            self.field_data)
        
    def timer_update(self):
        for timer in self.timers.values():
            timer.update()

    def move_down(self):
        # print('Timer')
        self.tetromino.move_down()

    def draw_grid(self):
        for col in range(1,COLUMNS):
            x =col*CELL_SIZE
            if(self.render):
                pygame.draw.line(self.line_surface,LINE_COLOR,(x,0),(x,self.surface.get_height()),1)
        for row in range(1,ROWS):
            y = row*CELL_SIZE
            if(self.render):
                pygame.draw.line(self.line_surface,LINE_COLOR,(0,y),(self.surface.get_width(),y),1)
        
        if(self.render):
            self.surface.blit(self.line_surface,(0,0))

    def input(self,action):
        # action_map = {
        #     0: pygame.K_LEFT,
        #     1: pygame.K_RIGHT,
        #     2: pygame.K_UP,
        #     3: pygame.K_DOWN
        # } 
        if(self.render):
            keys = pygame.key.get_pressed()
            if not self.timers['horizontal move'].active:
                if keys[pygame.K_LEFT]:
                    self.tetromino.move_horizontal(-1)
                    self.timers['horizontal move'].activate()
                elif keys[pygame.K_RIGHT]:
                    self.tetromino.move_horizontal(1)
                    self.timers['horizontal move'].activate()


            if not self.timers['rotate'].active:
                if keys[pygame.K_UP]:
                    self.tetromino.rotate()
                    self.timers['rotate'].activate()

            if not self.down_pressed and (keys[pygame.K_DOWN]):
                self.down_pressed = True
                self.timers['vertical move'].duration = self.down_speed_faster
                # if self.current_level > 5:
                #     self.reward = 1

            if  self.down_pressed and not  (keys[pygame.K_DOWN]):
                self.down_pressed = False
                self.timers['vertical move'].duration = self.down_speed
        else:  
        #--------------------------------
        # print(keys)
            # if not self.timers['horizontal move'].active:
            if action == 0:
                self.tetromino.move_horizontal(-1)
                # self.timers['horizontal move'].activate()
            elif action == 1:
                self.tetromino.move_horizontal(1)
                # self.timers['horizontal move'].activate()


            if not self.timers['rotate'].active:
                if action == 2:
                    self.tetromino.rotate()
                    # self.timers['rotate'].activate()

            # if not self.down_pressed and action == 3:
            #     self.down_pressed = True
                # self.timers['vertical move'].duration = self.down_speed_faster
                # if self.current_level > 5:
                #     self.reward = 1

            if  self.down_pressed and not action == 3:
                self.down_pressed = False
                self.timers['vertical move'].duration = self.down_speed

    def check_finished_rows(self):
        delete_rows = []
        for i,row in enumerate(self.field_data):
            if all(row):
                delete_rows.append(i)
            # else:
            #     self.reward = -1

        if delete_rows:
            for delete_row in delete_rows:
                # print(self.field_data[delete_row])
                for block in self.field_data[delete_row]:
                    if isinstance(block, Block):
                        block.kill()
                for row in self.field_data:
                    for block in row:
                        if block and block.pos.y < delete_row:
                            block.pos.y += 1
            self.field_data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
            for block in self.sprites:
                self.field_data[int(block.pos.y)][int(block.pos.x)] = block

            self.calculate_score(len(delete_rows))
    def get_data(self):
        data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        for y in range(ROWS):
            for x in range(COLUMNS):
                if isinstance(self.field_data[y][x], Block):
                    data[y][x] = 1
        return data
    # def check_dangerrous(self):
    #         danger_columns = []
    #         data_np = np.array(self.data)
    #         for col in range(data_np.shape[1]):  # สำหรับแต่ละคอลัมน์
    #             column_data = data_np[:, col]  # ดึงคอลัมน์ออกมา

    #             ones_indices = np.where(column_data == 1)[0]  # หาตำแหน่งที่เป็น 1
    #             if len(ones_indices) >= 2:
    #                 # ถ้ามี 1 มากกว่า 1 จุด และไม่ติดกันทั้งหมด (ห่างกันเกิน 1)
    #                 if not np.all(np.diff(ones_indices) == 1):
    #                     danger_columns.append(col)
    #                     self.reward = -5
    #                     return True
    #         if danger_columns.count == 0:
    #             self.reward = 0.1
    #             return False
    def is_low_and_flat(self):
        data_np = np.array(self.data)
        filled_rows = [np.sum(row) for row in data_np]

        lower_half = filled_rows[-5:]  # แถวล่าง 50%
        avg_fill = np.mean(lower_half)

        # ถ้าเฉลี่ยการเติมแถวล่างมากกว่า 70% แปลว่าเตี้ยและแบน
        if avg_fill >= data_np.shape[1] * 0.7:
            
            return True
        else:
            self.reward = 1
            # self.reward = 1
            return False 

    def run(self,action=None):

        #update
        if self.render:
            self.timer_update()
            self.sprites.update()
        # print(action)
        self.input(action)
        self.tetromino.move_down() 
        Flat = self.is_low_and_flat()
        if self.render:

            self.sprites.update()
            #drawing
            self.surface.fill(GRAY)
            self.sprites.draw(self.surface)

            self.draw_grid()
            self.display_surface.blit(self.surface,(PADDING,PADDING))
            pygame.draw.rect(self.display_surface,LINE_COLOR,self.rect,2,2)

        pic,obs = self._get_observation()
        # for row in self.data:
        #     print(row)
        # input("Press Enter to continue...")
        if self.terminated:
            if self.current_score < 40:  # เล่นแป๊บเดียวก็ตาย
                self.reward = -80
            elif self.current_lines == 0:
                self.reward = -100  # ไม่ได้เคลียร์แถวเลย

        return_reward = self.reward
        self.reward = 0
        # print(return_reward)
        return pic, obs, return_reward, self.terminated,self.current_score

class Tetromino:
    def __init__(self,shape,group,create_new_tetromino,field_data):
        self.shape = shape
        self.block_pos =  TETROMINOS[shape]['shape']
        self.color = TETROMINOS[shape]['color']
        self.create_new_tetromino = create_new_tetromino
        self.field_data = field_data

        #create block
        self.blocks = [Block(group,pos,self.color) for pos in self.block_pos]
        
    def move_down(self):
        if not self.next_move_vertical_collide(self.blocks,1):
            for block  in self.blocks:
                block.pos.y += 1
        else:
            for block in self.blocks: 
                self.field_data[int(block.pos.y)][int(block.pos.x)] = block
            self.create_new_tetromino() 

    def rotate(self):
        if (self.shape != 'O' and self.shape != 'I')and len(self.blocks) > 0:
            pivot_pos = self.blocks[0].pos
            new_block_positions = [block.rotate(pivot_pos) for block in self.blocks]

            for pos in new_block_positions:
                x, y = int(pos.x), int(pos.y)
                if x < 0 or x >= COLUMNS or y < 0 or y >= ROWS:
                    return 
                if self.field_data[y][x]:
                    return
            for i,block in enumerate(self.blocks):
                block.pos = new_block_positions[i]

    def move_horizontal(self,amount):
        if not self.next_move_horizontal_collide(self.blocks,amount):
            for block  in self.blocks:
                block.pos.x += amount

    def next_move_horizontal_collide(self,blocks,amount):
        collision_list = [block.horizontal_collide(int(block.pos.x + amount),self.field_data) for block in self.blocks]
        return True if any(collision_list) else False
        
    def next_move_vertical_collide(self,blocks,amount):
        collision_list = [block.vertical_collide(int(block.pos.y + amount),self.field_data) for block in self.blocks]
        return True if any(collision_list) else False
    
class Block(pygame.sprite.Sprite):
    def __init__(self,group,pos,color, render=False):
        #general
        super().__init__((group if render else []))
        self.render = render
        self.pos = pygame.Vector2(pos) + BLOCK_OFFSET
        if self.render:
            self.image = pygame.Surface((CELL_SIZE,CELL_SIZE))
            self.image.fill(color)
            self.rect = self.image.get_rect(topleft = self.pos * CELL_SIZE)
        else:
            self.image = None
            self.rect = None
        #position
        
        

    def rotate(self,pivot_pos):    
        return pivot_pos + (self.pos - pivot_pos).rotate(90)
    
    def update(self):
        self.rect.topleft = self.pos * CELL_SIZE

    def horizontal_collide(self,x, field_data):
        if not 0 <= x < COLUMNS:
            return True
        
        if field_data[int(self.pos.y)][x]:
            return True
        
    def vertical_collide(self,y,field_data):
        if y >= ROWS:
            return True
        #y >= 0 and 
        if y >= 0 and  field_data[y][int(self.pos.x)]:
            return True
