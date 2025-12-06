import carla
import pygame

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Main loop
try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position on screen
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Raycast from spectator camera
                spectator = world.get_spectator()
                transform = spectator.get_transform()
                
                # Convert mouse screen coordinates to world coordinates (rough approach)
                # NOTE: For exact conversion you need the CARLA camera projection, 
                # which is more complex.
                print(f"Mouse clicked at screen: {mouse_x}, {mouse_y}")
                print(f"Spectator location: {transform.location}, rotation: {transform.rotation}")

finally:
    pygame.quit()
