from utils import *
from globals import screen_size, screen_scaling, path
import pygame
import math
from pygame.math import Vector2
import numpy as np

vec2 = Vector2


"""
Definition of game environment which controls and rewards car
"""

class Env():
    def __init__(self, fps):
        # load track
        self.scaling = screen_scaling  # CURRENT FIX for usage on different screens
        grass_img = scale_image(pygame.image.load(path + "assets/map/grass_background.png"), self.scaling * (1 + 1/3))
        track_img = scale_image(pygame.image.load(path + "assets/map/track.png"), self.scaling)

        self.width = track_img.get_width()
        self.height = track_img.get_height()

        self.map_images = [(grass_img, (0, 0)), (track_img, (0, 0))]

        # for computer vision
        self.comp_vision_ON = False
        self.beam_ON = False

        # initialise car
        tile_size = 72
        start_pos = (4 * tile_size * self.scaling, 1.75 * tile_size * self.scaling)
        self.car = Car(start_pos, fps)

        # save initial state
        self.state = self.car.state

        # save hyperparameters for DQN agent
        self.action_size = self.car.action_size
        self.state_size = self.car.state_size

    """
    Main functions
    """
    def reset(self):
        self.state = self.car.reset()
        return self.state

    def step(self, action_idx):
        next_state, reward, done = self.car.step(action_idx)
        return next_state, reward, done

    """
    Drawing
    """
    def render(self, screen):
        # Drawing
        if self.comp_vision_ON:
            self.draw_computer_vision(screen)
        else:
            self.draw(screen)
            # show car
            self.car.show_state(screen)
            self.car.show_action(screen)
            if self.beam_ON:
                red = (255, 0, 0)
                green = (0, 255, 0)
                orange = (255, 165, 0)
                if self.beam_ON:
                    self.car.draw_beams(screen, orange)
                self.car.draw_next_gate_after_direction(screen, red)
                self.car.draw_next_gate_direction(screen, green)
        pygame.display.update()

    def draw(self, screen):
        for img, pos in self.map_images:
            screen.blit(img, pos)

    def draw_computer_vision(self, screen):
        screen.fill((0, 0, 0, 0))
        white = (255, 255, 255)
        green = (0, 255, 0)
        orange = (255, 165, 0)
        self.car.draw_beams(screen, white)
        self.car.draw_next_gate_direction(screen, green)
        self.car.draw_past_gate_direction(screen, orange)
        self.car.draw_rect(screen)

"""
General car object
"""
# Main source for basic car mechanics (NOT Q-LEARNING ASPECT): https://github.com/Code-Bullet/Car-QLearning/blob/master/Game.py
class Car(pygame.sprite.Sprite):
    def __init__(self, position, fps):
        self.scaling = screen_scaling
        self.start_pos = position
        self.x, self.y = position
        self.vel = 0
        self.direction = vec2(1, 0)
        self.acc = 0
        pygame.sprite.Sprite.__init__(self)
        self.img = scale_image(pygame.image.load(path + "assets/car/car_red_5.png").convert_alpha(), .4 * self.scaling)
        self.update_mask()
        self.width = self.img.get_width() #40
        self.height = self.img.get_height() #20
        self.turningRate = 4.5 / self.width
        self.friction = .97 # 0.98
        self.maxSpeed = self.width / 4.0
        self.maxReverseSpeed = -1 * self.maxSpeed / 2.0
        self.accelerationSpeed = self.width / 160.0
        self.driftMomentum = 0
        self.driftFriction = .85

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        # reward system related
        self.num_gates_crossed = 0
        ## necessary to check if finish line has been left
        self.finish_line_crossed = False
        self.prev_finish_line_crossed = False
        self.finish_line_reward = False
        self.fps = fps
        self.last_gate_pos = Vector2(self.x, self.y)

        ## manual definition of reward gates
        self.num_gates = 34
        self.border = border(self.scaling)
        self.reward_gates = reward_gates(self.scaling, self.num_gates_crossed)
        self.no_finish_line = finish_line(self.scaling, True)
        self.finish_line = finish_line(self.scaling, False)

        # FOR OBTAINING THE ANGLE TO THE NEXT GATE (CURRENT FIX)
        self.next_gate_pos = [[264.3048311403092, 81.9], [408.1479667629719, 84.25341474646493],
        [548.4186780893119, 119.17097614014048], [677.2882318269155, 125.13862019318192],
        [805.2110160041187, 77.86661690410043], [968.5731322019908, 64.28504376439385],
        [1108.3899034558385, 148.51277651868077], [1157.454265410554, 303.01586724599343],
        [1088.8739155162898, 464.9010336220801], [890.6697921620014, 540.081449758843],
        [692.9509198201459, 630.2935727203267], [654.6023814881245, 821.0900214201558],
        [761.2101876871108, 945.5255378224434], [912.8143727140599, 866.1166857816819],
        [1023.2207650737586, 788.1788046681667], [1143.5731592406312, 808.7552646229701],
        [1140.0195371250309, 944.2328908312688], [1047.4605603915018, 1092.8853808104227],
        [866.7318586924993, 1171.348882936682], [651.1114746592181, 1187.7882291676412],
        [464.69428038051, 1123.1002074978926], [353.70329181544577, 984.8185448400843],
        [371.66893592981893, 793.6431006951887], [465.7265569956395, 637.3652158120723],
        [558.5064819208914, 489.6149848768322], [587.5583463451536, 358.41549065392854],
        [448.7085633521466, 406.4036922959319], [338.1818958449041, 501.8036353651021],
        [159.2784626716964, 573.9681275825119], [198.35428627486527, 405.97197863644533],
        [316.2174830373177, 307.1946354878292], [366.2111959696949, 211.06338076688317],
        [186.19884727169915, 239.41711602826757], [65.46895314323993, 165.97200339253746],
        [155.098017870885, 77.14386382455129]]

        ## defining angles of beams
        self.beam_angles = [deg for deg in range(0, 59, 10)]
        self.beam_angles = self.beam_angles + [deg for deg in range(60, 181, 30)]
        self.beam_angles = self.beam_angles + [(-1 * deg) % 360 for deg in range(10, 59, 10)]
        self.beam_angles = self.beam_angles + [(-1 * deg) % 360 for deg in range(60, 179, 30)]

        # NECESSARY TO DEFINE NORMALISED STATE VALUES
        self.max_dist = max(screen_size)/2  # for distance to border
        self.max_drift_mom = (self.maxSpeed * self.turningRate * self.width / (9.0 * 8.0)) * 5 # based on formula in "exec_controls" and then trial and error

        # FOR STATE-ACTION BASED TRAINING
        self.action = 4
        self.state = self.get_state()

        self.score = 0
        self.lifespan = 0
        self.reward = 0
        self.terminal = False

        ## necessary hyperparameter
        # besides distances to border based on beams, extra 4 include pos and neg
        # parts of velocity, and pos and neg parts of drift momentum
        self.state_size = len(self.state)
        self.action_size = 9
        self.action_space = np.arange(self.action_size)

        # visualising states by different car colours
        self.collision_car_img = scale_image(pygame.image.load(path + "assets/car/car_black_5.png").convert_alpha(), .4 * self.scaling)
        self.reward_car_img = scale_image(pygame.image.load(path + "assets/car/car_green_5.png").convert_alpha(), .4 * self.scaling)
        self.finish_car_img = scale_image(pygame.image.load(path + "assets/car/car_yellow_5.png").convert_alpha(), .4 * self.scaling)
        self.bool_collision = False
        self.bool_reward = False
        self.bool_finish = False

    """
    Main methods of agent control
    """
    def reset(self):
        self.x, self.y = self.start_pos
        self.vel = 0
        self.direction = vec2(1, 0)
        self.acc = 0
        self.driftMomentum = 0

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        self.num_gates_crossed = 0
        self.last_gate_pos = Vector2(self.x, self.y)
        self.update_reward_gate_mask()
        self.score = 0
        self.lifespan = 0
        self.terminal = False

        self.state = self.get_state()
        return self.state

    def step(self, action_idx):
        self.action = action_idx
        self.change_controls()
        state, reward, done = self.make_step()
        return state, reward, done

    """
    Methods that are executed by main methods
    """
    def change_controls(self):
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        if self.action == 0:
            self.turningLeft = True
            self.accelerating = True
        elif self.action == 1:
            self.accelerating = True
        elif self.action == 2:
            self.turningRight = True
            self.accelerating = True
        elif self.action == 3:
            self.turningLeft = True
        elif self.action == 4:
            pass
        elif self.action == 5:
            self.turningRight = True
        elif self.action == 6:
            self.turningLeft = True
            self.reversing = True
        elif self.action == 7:
            self.reversing = True
        elif self.action == 8:
            self.reversing = True
            self.turningRight = True

    def make_step(self):
        self.exec_controls()
        self.move()
        state, reward, done = self.get_state_rewards()
        return state, reward, done

    """
    Methods that prepare for methods that execute main methods
    """
    def exec_controls(self):
        scaling = 6 + abs(self.vel)/self.maxSpeed
        multiplier = abs(self.vel) / scaling
        if self.vel < 0:
            multiplier *= -1

        driftAmount = self.vel * self.turningRate * self.width / (9.0 * 8.0)
        if self.vel < 5:
            driftAmount = 0

        if self.turningLeft:
            self.direction = self.direction.rotate(-radiansToAngle(self.turningRate) * multiplier)
            self.driftMomentum += driftAmount
        elif self.turningRight:
            self.direction = self.direction.rotate(radiansToAngle(self.turningRate) * multiplier)
            self.driftMomentum -= driftAmount
        self.acc = 0
        if self.accelerating:
            if self.vel < 0:
                self.acc = 3 * self.accelerationSpeed
            else:
                self.acc = self.accelerationSpeed
        elif self.reversing:
            if self.vel > 0:
                self.acc = -1 * self.accelerationSpeed
            else:
                self.acc = -1 * self.accelerationSpeed

    def move(self):
        global vec2
        self.vel += self.acc
        self.vel *= self.friction
        self.constrain_vel()

        driftVector = vec2(self.direction)
        driftVector = driftVector.rotate(90)

        addVector = vec2(0, 0)
        addVector.x += self.vel * self.direction.x
        addVector.x += self.driftMomentum * driftVector.x
        addVector.y += self.vel * self.direction.y
        addVector.y += self.driftMomentum * driftVector.y
        self.driftMomentum *= self.driftFriction

        if addVector.length() != 0:
            addVector.normalize()

        addVector.x * abs(self.vel)
        addVector.y * abs(self.vel)

        self.x += addVector.x
        self.y += addVector.y

    def constrain_vel(self):
        if self.maxSpeed < self.vel:
            self.vel = self.maxSpeed
        elif self.vel < self.maxReverseSpeed:
            self.vel = self.maxReverseSpeed

    def get_state_rewards(self):
        state = self.get_state()
        reward, done = self.reward_state()
        return state, reward, done

    """
    Checking and output of states
    """
    def get_state(self):
        self.update_mask()
        self.check_state()

        normalised_dist = self.get_norm_distances()  # (inverted) normalised distances to border
        state = np.array([1 - np.min([1, dist]) for dist in normalised_dist])
        # TRIAL: Normalised angle and distance to next gate
        norm_angle_to_next_gate, norm_dist_to_next_gate = self.get_norm_angle_dist_to_next_gate()
        # # state = np.append(state, [norm_dist_to_next_gate])  # TRIAL: Only distance
        # # state = np.append(state, [norm_dist_to_next_gate])
        #
        # # TRIAL2: Normalised angle and distance to next gate after
        norm_angle_to_next_gate_after, norm_dist_to_next_gate_after = self.get_norm_angle_dist_to_next_after_next_gate()
        # # state = np.append(state, [norm_dist_to_next_gate])

        # IDEA: Instead of giving it the angle, use sin, cos (as pointing vector)
        x_part = math.cos(math.radians(norm_angle_to_next_gate * 180))
        y_part = math.sin(math.radians(norm_angle_to_next_gate * 180))
        state = np.append(state, [x_part, y_part])
        # IDEA2: To smooth behaviour (and forward planning) also give it the pointing vector to the gate after that
        x_part_ = math.cos(math.radians(norm_angle_to_next_gate_after * 180))
        y_part_ = math.sin(math.radians(norm_angle_to_next_gate_after * 180))
        state = np.append(state, [x_part_, y_part_])

        # instead of having positive and negative values we split each into a positive and negative component
        if self.vel >= 0:
            pos_vel = self.vel / self.maxSpeed
            neg_vel = 0
        else:
            pos_vel = 0
            neg_vel = abs(self.vel / self.maxReverseSpeed)
        state = np.append(state, [pos_vel, neg_vel])

        if self.driftMomentum >= 0:
            pos_drift_mom = self.driftMomentum/self.max_drift_mom
            neg_drift_mom = 0
        else:
            pos_drift_mom = 0
            neg_drift_mom = abs(self.driftMomentum/self.max_drift_mom)
        state = np.append(state, [pos_drift_mom, neg_drift_mom])

        return state

    def check_state(self):
        self.bool_collision = False
        self.bool_reward = False
        self.bool_finish = False
        # necessary to not award finish line reward while staying on the line
        self.prev_finish_line_crossed = self.finish_line_crossed
        if pygame.sprite.collide_mask(self, self.border):
            self.bool_collision = True
            # self.collision_point = pygame.sprite.collide_mask(self, self.border)
            # offset = (int(self.x), int(self.y))
            # self.collision_mask = self.border.mask.overlap_mask(self.mask, offset)
            #print(self.collision_point)
        elif pygame.sprite.collide_mask(self, self.reward_gates):
            self.bool_reward = True
            # self.last_gate_pos = Vector2(self.x, self.y)
            # CURRENT FIX TO GET POSITION OF NEXT GATE
            # buffer = np.sqrt((self.width/2)**2 + (self.height/2)**2)
            # collision_point_x = self.x + buffer * math.cos(math.radians(get_angle(self.direction)))
            # collision_point_y = self.y + buffer * math.sin(math.radians(get_angle(self.direction)))
            # self.next_gate_pos.append([collision_point_x, collision_point_y])
        elif pygame.sprite.collide_mask(self, self.finish_line):
            self.bool_finish = True
            self.finish_line_crossed = True
            if not(self.prev_finish_line_crossed):
                self.finish_line_reward = True
                # CURRENT FIX TO GET POSITION OF NEXT GATE
                # buffer = np.sqrt((self.width/2)**2 + (self.height/2)**2)
                # collision_point_x = self.x + buffer * math.cos(math.radians(get_angle(self.direction)))
                # collision_point_y = self.y + buffer * math.sin(math.radians(get_angle(self.direction)))
                # self.next_gate_pos.append([collision_point_x, collision_point_y])
            else:
                self.finish_line_reward = False
        else:
            self.finish_line_crossed = False
            pass

    def reward_state(self):
        self.reward = -1/self.fps  # every second costs one point
        if self.bool_collision:
            self.reward -= 50
            self.terminal = True
        elif self.bool_reward:
            self.reward += 1 # * vel_scaling  # reward gate passing with +2 at full speed and +1 close to stand-still
            # OLD VERSION
            # self.reward += 1
            self.num_gates_crossed += 1
            self.update_reward_gate_mask()
        elif self.bool_finish:
            if self.finish_line_reward:
                if (self.direction[0] >= 0) & (self.vel >= 0):  # only reward if crossed finish from left
                    self.reward += 50
                    self.num_gates_crossed = 0
                    self.update_reward_gate_mask()
                else:
                    self.reward -= 50
                    self.terminal = True
            else:
                pass
        self.lifespan += 1
        self.score += self.reward
        return self.reward, self.terminal

    """
    Calculations for states
    """
    def get_norm_distances(self):
        scaling = self.max_dist
        front_angle = get_angle(self.direction)
        all_angles = [(front_angle - deg) % 360 for deg in self.beam_angles]
        all_dist = []
        for angle in all_angles:
            dist = self.get_distance(angle)
            # Reduce by the distance to the car rectangle
            pos, dist_to_car_rect = self.get_pos_on_car_rect(angle)
            dist = (dist - dist_to_car_rect) / scaling
            all_dist.append(dist)

        return all_dist

    def get_distance(self, angle):
        pos, dist = self.get_pos_on_car_rect(angle)
        origin = Vector2(pos[0], pos[1])

        target_x = origin[0] + math.cos(math.radians(angle)) * screen_size[0]
        target_y = origin[1] + math.sin(math.radians(angle)) * screen_size[1]
        target = Vector2(target_x, target_y)

        direction_raw = target - origin
        direction = direction_raw.normalize()

        collision_point = origin
        for _ in range(int(direction_raw.length())):
            collision_point += direction
            if (self.border.mask.get_at(collision_point) == 1):
                return np.linalg.norm(np.array(collision_point) - np.array([self.x, self.y]))

    def get_norm_angle_dist_to_next_gate(self):
        position_of_gate = self.get_pos_of_next_gate()
        vec_to_next_gate = position_of_gate - Vector2(self.x, self.y)
        # CURRENT VERSION: relative angle
        angle_to_next_gate = (get_angle(self.direction) - get_angle(vec_to_next_gate)) % 360
        if angle_to_next_gate > 180:
            angle_to_next_gate = -1 * (360 - angle_to_next_gate)
        dist_to_next_gate = np.linalg.norm(np.array(position_of_gate) - np.array([self.x, self.y]))

        # subtract distance from center to rectangle of car
        global_angle = get_angle(vec_to_next_gate)
        pos, dist_to_car_rect = self.get_pos_on_car_rect(global_angle)
        dist_to_next_gate = (dist_to_next_gate - dist_to_car_rect) / self.max_dist

        angle_to_next_gate /= 180  # 180 # 360 used if global angle

        return angle_to_next_gate, dist_to_next_gate

    def get_norm_angle_dist_to_next_after_next_gate(self):
        position_of_gate = self.get_pos_of_next_after_next_gate()
        vec_to_next_gate_after = position_of_gate - Vector2(self.x, self.y)
        # CURRENT VERSION: relative angle
        angle_to_next_gate_after = (get_angle(self.direction) - get_angle(vec_to_next_gate_after)) % 360
        if angle_to_next_gate_after > 180:
            angle_to_next_gate_after = -1 * (360 - angle_to_next_gate_after)
        dist_to_next_gate_after = np.linalg.norm(np.array(position_of_gate) - np.array([self.x, self.y]))

        # subtract distance from center to rectangle of car
        global_angle = get_angle(vec_to_next_gate_after)
        pos, dist_to_car_rect = self.get_pos_on_car_rect(global_angle)
        dist_to_next_gate_after = (dist_to_next_gate_after - dist_to_car_rect) / self.max_dist

        angle_to_next_gate_after /= 180  # 180 # 360 used if global angle

        return angle_to_next_gate_after, dist_to_next_gate_after

    def get_pos_of_next_gate(self):  # TRIAL: Additional input of direction to next gate
    # to speed up learning (and give the agent a tangible target)
        next_gate_idx = self.num_gates_crossed
        next_gate_pos = self.next_gate_pos[next_gate_idx]

        return next_gate_pos

    def get_pos_of_next_after_next_gate(self):  # TRIAL: Additional input of direction to next gate
    # to speed up learning (and give the agent a tangible target)
        next_after_gate_idx = self.num_gates_crossed + 1
        if next_after_gate_idx == len(self.next_gate_pos):
            next_after_gate_idx = 0
        next_after_gate_pos = self.next_gate_pos[next_after_gate_idx]

        return next_after_gate_pos

    # IMPORTANT: If we were giving the distances from the car position to the border
    # along the beams, the border is touched even if the distance is non-zero (as the
    # position is the center of the car rectangle); hence we need to obtain the
    # starting position of the beam along the car object
    def get_pos_on_car_rect(self, angle):
        # following is simple trigonometry
        # TRIAL VERSION: (account for negative direction of car orientation)
        # angle_to_direction = self.get_angle_rel_to_direction(angle)
        # PREVIOUS VERSION
        angle_to_direction = (angle - get_angle(self.direction)) % 360

        # ATTENTION: Width and height are flipped because they are measured from the (rotated) image
        width = self.height
        height = self.width

        angle_to_front_left_corner = math.degrees(math.atan2(width, height))
        angle_to_rear_left_corner = 180 - angle_to_front_left_corner

        angle_to_front_right_corner = 360 - angle_to_front_left_corner
        angle_to_rear_right_corner = 180 + angle_to_front_left_corner

        if (angle_to_direction > angle_to_front_left_corner) and (angle_to_direction < angle_to_rear_left_corner):
            dist = (width/2) / math.sin(math.radians(angle_to_direction))
        elif (angle_to_direction > angle_to_rear_right_corner) and (angle_to_direction < angle_to_front_right_corner):
            dist = (width/2) / math.sin(math.radians(angle_to_direction - 180))
        elif (angle_to_direction <= angle_to_front_left_corner) or (angle_to_direction >= angle_to_front_right_corner):
            dist = (height/2) / math.cos(math.radians(angle_to_direction))
        else:
            dist = (height/2) / math.cos(math.radians(angle_to_direction - 180))

        pos_x = self.x + dist * math.cos(math.radians(angle))  # ATTENTION: Accounts for rotation of car already
        pos_y = self.y + dist * math.sin(math.radians(angle))
        pos = (pos_x, pos_y)

        return pos, dist

    """
    Sprite movement and drawing
    """
    def update_mask(self):
        rotated = pygame.transform.rotate(self.img, -get_angle(self.direction))
        self.mask = pygame.mask.from_surface(rotated)
        self.rect = rotated.get_rect(center=(self.x, self.y))

    def update_reward_gate_mask(self):
        self.reward_gates = reward_gates(self.scaling, self.num_gates_crossed)

    def update_finish_line_mask(self):
        self.finish_line = finish_line(self.scaling, self.finish_line_crossed)

    def draw_rect(self, screen):
        rect = self.img.get_rect(center=(self.x, self.y))
        angle = -get_angle(self.direction)

        pivot = Vector2(self.x, self.y)

        p0 = (Vector2(rect.topleft) - pivot).rotate(-angle) + pivot
        p1 = (Vector2(rect.topright) - pivot).rotate(-angle) + pivot
        p2 = (Vector2(rect.bottomright) - pivot).rotate(-angle) + pivot
        p3 = (Vector2(rect.bottomleft) - pivot).rotate(-angle) + pivot
        pygame.draw.lines(screen, (255, 255, 0), True, [p0, p1, p2, p3], 3)

    def show(self, screen):
        rotated = pygame.transform.rotate(self.img, -get_angle(self.direction))
        rect = rotated.get_rect()
        screen.blit(rotated, (self.x - rect.width / 2, self.y - rect.height / 2))

    def show_score(self, screen):
        white = (255, 255, 255)
        message = "Score: " + str(int(np.floor(self.score)))
        font = pygame.font.Font(None, 40)
        screen_message = font.render(message, 1, white)
        screen.blit(screen_message, (10 * self.scaling, 10  * self.scaling))

    def show_action(self, screen):
        action_img_name = path + "assets/misc/arrow_keys/arrow_keys_" + str(self.action) + ".png"
        action_img = scale_image(pygame.image.load(action_img_name).convert_alpha(), .3 * self.scaling)
        screen.blit(action_img, (10 * self.scaling, screen_size[1] - 350 * self.scaling))

    def show_state(self, surface):
        state_img = self.img
        if self.bool_collision:
            state_img = self.collision_car_img
        elif self.bool_reward:
            state_img = self.reward_car_img
        elif self.bool_finish:
            state_img = self.finish_car_img

        rotated = pygame.transform.rotate(state_img, -get_angle(self.direction))
        rect = rotated.get_rect()
        surface.blit(rotated, (self.x - rect.width / 2, self.y - rect.height / 2))
        self.show_score(surface)

    def draw_beams(self, screen, color):
        # TRIAL VERSION: (account for negative direction of car orientation)
        # front_angle = -get_angle(self.direction)
        # all_angles = [(front_angle + deg) % 360 for deg in self.beam_angles]
        # PREVIOUS VERSION:
        front_angle = get_angle(self.direction)
        # TRIAL VERSION: Instead of adding the degree to the car angle we substract it
        all_angles = [(front_angle - deg) % 360 for deg in self.beam_angles]

        # TEST VERSION: Draw beams only using the input the car receives
        all_dist = self.get_norm_distances()
        all_dist = [dist * self.max_dist for dist in all_dist]
        for i in range(len(all_angles)):
            angle = all_angles[i]
            pos, _ = self.get_pos_on_car_rect(angle)
            dist = all_dist[i]
            target_x = pos[0] + math.cos(math.radians(angle)) * dist
            target_y = pos[1] + math.sin(math.radians(angle)) * dist
            # print(target_x, target_y, self.x, self.y)
            # DEBUGGING VERSION: Different colors for each side, front and back
            orange = (255, 165, 0)
            yellow = (255, 255, 0)
            black = (0, 0, 0)
            color_angle = self.beam_angles[i]
            if (color_angle == 0) or (color_angle == 180):
                color = black
            elif (color_angle > 0) and (color_angle < 180):
                color = yellow
            else:
                color = orange
            pygame.draw.line(screen, color, (pos[0], pos[1]), (target_x, target_y), 2)
            pygame.draw.circle(screen, color, (target_x, target_y), 4)

    # FOR DEBUGGING: Draw position of next gate
    def draw_next_gate_direction(self, screen, color):
        # NEW VERSION: Given the NN the sine, cos of the global angle as pointing vector
        position_of_gate = self.get_pos_of_next_gate()
        global_angle = get_angle(position_of_gate - Vector2(self.x, self.y))
        x_part = math.cos(math.radians(global_angle))
        y_part = math.sin(math.radians(global_angle))
        pos, _ = self.get_pos_on_car_rect(global_angle)

        target_x = pos[0] + x_part * 20
        target_y = pos[1] + y_part * 20
        pygame.draw.line(screen, color, (pos[0], pos[1]), (target_x, target_y), 2)

    def draw_next_gate_after_direction(self, screen, color):
        position_of_gate_after = self.get_pos_of_next_after_next_gate()
        global_angle = get_angle(position_of_gate_after - Vector2(self.x, self.y))
        x_part = math.cos(math.radians(global_angle))
        y_part = math.sin(math.radians(global_angle))

        pos, _ = self.get_pos_on_car_rect(global_angle)
        target_x = pos[0] + x_part * 30
        target_y = pos[1] + y_part * 30
        pygame.draw.line(screen, color, (pos[0], pos[1]), (target_x, target_y), 2)


"""
Definition of sprite classes
"""
class border(pygame.sprite.Sprite):
    def __init__(self, scaling):
        pygame.sprite.Sprite.__init__(self)
        self.image = scale_image(pygame.image.load(path + "assets/map/track_border.png").convert_alpha(), scaling)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

class reward_gates(pygame.sprite.Sprite):
    def __init__(self, scaling, num_crossed):
        pygame.sprite.Sprite.__init__(self)
        img_name = path + "assets/map/reward_gates/reward_gates-" + str(num_crossed) + ".png"
        self.image = scale_image(pygame.image.load(img_name).convert_alpha(), scaling)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

class finish_line(pygame.sprite.Sprite):
    def __init__(self, scaling, bool_crossed):
        pygame.sprite.Sprite.__init__(self)
        img_name = path + "assets/map/finish_line/finish_line_" + str(bool_crossed) + ".png"
        self.image = scale_image(pygame.image.load(img_name).convert_alpha(), scaling)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
