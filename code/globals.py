import pygame
from utils import scale_image
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/"
screen_scaling = .65
track_img = scale_image(pygame.image.load(path + "assets/map/track.png"), screen_scaling)
screen_size = (track_img.get_width(), track_img.get_height())
