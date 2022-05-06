import pygame
import math

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

def blit_rotate_center(wind, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    wind.blit(rotated_image, new_rect.topleft)

def get_angle(vec):
    if vec.length() == 0:
        return 0
    return math.degrees(math.atan2(vec.y, vec.x))

def radiansToAngle(rads):
    return rads * 180 / math.pi

def angleToRadians(angle):
    return angle / (180 / math.pi)

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
