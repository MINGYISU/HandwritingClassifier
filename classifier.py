import numpy as np
import pygame
import sys
from tensorflow import keras
import time

# Check for command-line arguments
if len(sys.argv) != 2:
    sys.exit("Usage: python classifier.py model_name")
model = keras.models.load_model(sys.argv[1])

from pygame.locals import (
    QUIT, 
    KEYDOWN, 
    K_ESCAPE
)

# define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize pygame
pygame.init()

# Define screen width and height
WIDTH = 1000
HEIGHT = 750
size = WIDTH, HEIGHT

# Create the screen object
screen = pygame.display.set_mode(size)
background_image = pygame.image.load("src_image/background.png")
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
pygame.display.set_caption("Handwriting Classifier")

# Size of the drawing board (has to be (28, 28))
ROW, COL = 28, 28

# Margin offset and the side length of the cell
CELL = 20
OFFSET_W = (WIDTH - ROW * CELL) / 4
OFFSET_H = (HEIGHT - COL * CELL) / 2


# Create the drawing board
drawboard = [[0] * COL for _ in range(ROW)]


running = True

# Main Loop
while running:
    # Check for user interaction
    for event in pygame.event.get():
        # If the user hits the close window button
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # Terminate the program
            sys.exit(0)
        
    # Fill the screen with white
    # screen.fill(BLACK)
    screen.blit(background_image, (0, 0))

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    mouse = None
    if click == 1:
        mouse = pygame.mouse.get_pos()

    # Draw the cells
    for i in range(ROW):
        for j in range(COL):
            rect = pygame.Rect(
                OFFSET_W + i * CELL, 
                OFFSET_H + j * CELL, 
                CELL, CELL
            )

            # Darken the grey cell
            if drawboard[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)
            # Draw a blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            
            # Draw boarders of the cells
            pygame.draw.rect(screen, BLACK, rect, 1)

            

            

    pygame.display.flip()

