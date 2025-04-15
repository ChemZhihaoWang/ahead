import random
import os
from PIL import Image, ImageDraw

def generate_continuous_grid_image(image_size, grid_size, line_width, save_path):
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    
    cell_width = image_size[0] // grid_size[0]
    cell_height = image_size[1] // grid_size[1]
    
    # Draw horizontal and vertical lines
    for i in range(grid_size[0] + 1):
        x = i * cell_width
        draw.line([(x, 0), (x, image_size[1])], fill='black', width=line_width)
    
    for j in range(grid_size[1] + 1):
        y = j * cell_height
        draw.line([(0, y), (image_size[0], y)], fill='black', width=line_width)
    
    # Generate the continuous path
    path = [(1, grid_size[1] - 2)]
    current_x, current_y = 1, grid_size[1] - 2  # Start from the first cell within the boundary
    directions = ['right', 'up', 'left', 'down']
    last_direction = None

    while current_x < grid_size[0] - 2 or current_y > 1:
        possible_moves = []
        if current_x < grid_size[0] - 2 and last_direction != 'left':  # Can move right
            possible_moves.append(('right', (current_x + 1, current_y)))
        if current_y > 1 and last_direction != 'down':  # Can move up
            possible_moves.append(('up', (current_x, current_y - 1)))
        if current_x > 1 and last_direction != 'right':  # Can move left
            possible_moves.append(('left', (current_x - 1, current_y)))
        if current_y < grid_size[1] - 2 and last_direction != 'up':  # Can move down
            possible_moves.append(('down', (current_x, current_y + 1)))

        if not possible_moves:
            break  # No more moves possible, end loop

        direction, next_move = random.choice(possible_moves)
        path.append(next_move)
        current_x, current_y = next_move

        # Ensure at least one step in the new direction
        for _ in range(1):
            if direction == 'right' and current_x < grid_size[0] - 2:
                current_x += 1
            elif direction == 'up' and current_y > 1:
                current_y -= 1
            elif direction == 'left' and current_x > 1:
                current_x -= 1
            elif direction == 'down' and current_y < grid_size[1] - 2:
                current_y += 1
            path.append((current_x, current_y))

        last_direction = direction

        if current_x == grid_size[0] - 2 and current_y == 1:
            break  # Reached top-right corner

    # Ensure we reach the top-right corner
    if (grid_size[0] - 2, 1) not in path:
        path.append((grid_size[0] - 2, 1))

    for (x, y) in path:
        draw.rectangle(
            [x * cell_width, y * cell_height, 
             (x + 1) * cell_width, (y + 1) * cell_height],
            fill='black'
        )
    
    image.save(save_path)
    print(f"Image saved at {save_path}")

def generate_batch_images(image_size, grid_size, line_width, folder_path, num_images):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for i in range(num_images):
        save_path = os.path.join(folder_path, f'{i+1}.png')
        generate_continuous_grid_image(image_size, grid_size, line_width, save_path)
        print(f"Image {i+1} saved at {save_path}")

# Parameters for individual image
image_size = (553, 553)  # Adjust to the desired image size
grid_size = (23, 23)  # Adjust to the desired grid size
line_width = 1  # Adjust to the desired line width

# Parameters for batch generation
folder_path = 'randomfig_100w'  # Folder to save images
num_images = 1000000  # Number of images to generate

generate_batch_images(image_size, grid_size, line_width, folder_path, num_images)
