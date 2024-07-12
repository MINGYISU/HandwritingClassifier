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
RED = (255, 71, 77)
GREEN = (144, 228, 144)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Initialize pygame
pygame.init()

# Define screen width and height
WIDTH = 1000
HEIGHT = WIDTH * 0.618
size = WIDTH, HEIGHT

# Fonts
FontPath = "fonts/PartyLET-plain.ttf"
FONT = pygame.font.Font(FontPath, 30)
bigFONT = pygame.font.Font(FontPath, 80)

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
classification = None

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

            # Darken the grey cell if this cell has been written on
            if drawboard[i][j]:
                channel = 255 - (drawboard[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)
            # Draw a blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            
            # Draw boarders of the cells
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If the user clicks on this cell, darken it and its neighbours
            if mouse and rect.collidepoint(mouse):
                # The cell itself
                drawboard[i][j] = 250 / 255
                # Neighbours
                if i + 1 < ROW:
                    drawboard[i + 1][j] = 220 / 255
                if j + 1 < COL:
                    drawboard[i][j + 1] = 220 / 255
                if i + 1 < ROW and j + 1 < COL:
                    drawboard[i + 1][j + 1] = 190 / 255
    
    BUTTON_OFFSET_W =  OFFSET_W + ROW * CELL + 30
    BUTTON_W = 180
    BUTTON_H = BUTTON_W * 0.618
    # Draw reset button
    ResetButton = pygame.Rect(
        BUTTON_OFFSET_W, OFFSET_H, 
        BUTTON_W, BUTTON_H
    )
    ResetText = FONT.render("RESET", True, WHITE)
    ResetTextRect = ResetText.get_rect()
    ResetTextRect.center = ResetButton.center
    pygame.draw.rect(screen, RED, ResetButton)
    screen.blit(ResetText, ResetTextRect)

    # Draw classifier button
    ClassButton = pygame.Rect(
        BUTTON_OFFSET_W, OFFSET_H + 170, 
        BUTTON_W, BUTTON_H
    )
    ClassText = FONT.render("CLASSIFY", True, WHITE)
    ClassTextRect = ClassText.get_rect()
    ClassTextRect.center = ClassButton.center
    pygame.draw.rect(screen, GREEN, ClassButton)
    screen.blit(ClassText, ClassTextRect)

    # Display area
    Display = pygame.Rect(
        BUTTON_OFFSET_W, OFFSET_H + 300, 
        BUTTON_W, BUTTON_H + 150
    )
    pygame.draw.rect(screen, WHITE, Display)

    # Reset the drawing if the reset button is clicked
    if mouse and ResetButton.collidepoint(mouse):
        drawboard = [[0] * COL for _ in range(ROW)]
        # Reset classification
        classification = None

    # Generate a classification if the class button is clicked
    if mouse and ClassButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(drawboard).reshape(1, 28, 28, 1)]
        ).argmax()

    # Show classification if generated
    if classification is not None:
        classificationText = bigFONT.render(str(classification), True, BLACK)
        classificationRect = classificationText.get_rect()
        classificationRect.center = Display.center
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()

