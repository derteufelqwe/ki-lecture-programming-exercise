from typing import List, Any
from queue import Queue
from PIL import Image
import numpy as np


class TileColor:
    # Define the colors for each tile type
    WALL = (0, 0, 0, 255)
    OUTSIDE = (255, 255, 255, 0)
    FLOOR = (200, 113, 55, 255)
    TEA_KITCHEN = (0, 0, 255, 255)
    PROFESSOR_OFFICE = (0, 255, 0, 255)
    LABOR = (255, 255, 0, 255)


class TileType:
    WALL = 'wall'
    OUTSIDE = 'outside'
    FLOOR = 'floor'
    TEA_KITCHEN = 'tea_kitchen'
    PROFESSOR_OFFICE = 'professor_offic'
    LABOR = 'labor'

    @staticmethod
    def of(pixel: tuple):
        """
        Pixel to tile type
        """

        return {
            TileColor.WALL: TileType.WALL,
            TileColor.OUTSIDE: TileType.OUTSIDE,
            TileColor.FLOOR: TileType.FLOOR,
            TileColor.TEA_KITCHEN: TileType.TEA_KITCHEN,
            TileColor.PROFESSOR_OFFICE: TileType.PROFESSOR_OFFICE,
            TileColor.LABOR: TileType.PROFESSOR_OFFICE,
        }[pixel]


def parse_image() -> np.matrix:
    """
    Parses the image and creates a matrix of tile types.
    """
    cell_cnt = 21
    img = Image.open('lageplan.png')
    cell_size = 420 / cell_cnt
    plan_matrix = list()

    for i in range(cell_cnt):
        row = list()
        for j in range(cell_cnt):
            # Get one pixel for each tile. x, y have an offset of 10 to get the color of the tile
            # and not of the border
            pixel = img.getpixel((int(i * cell_size + 10), int(j * cell_size + 10)))
            row.append(TileType.of(pixel))
        plan_matrix.append(row)

    img.close()
    return np.matrix(plan_matrix)


# --- Aufgabe a ---
plan_matrix = parse_image()
mx, my = plan_matrix.shape

# --- Aufgabe b ---
start = (3, 17)
end = (1, 3)
adj_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]    # For finding adjacent fields
queue = Queue()
queue.put(start)
# Make the path is reconstructable
previous_matrix = np.array([[(None, None) for _ in range(21)] for _ in range(21)])
visited = set()   # Store already visited nodes, otherwise performance is terrible

while not queue.empty():
    node_x, node_y = queue.get()
    visited.add((node_x, node_y))

    # Enqueue neighbours
    for adj_vec in adj_vectors:
        next_x, next_y = node_x + adj_vec[0], node_y + adj_vec[1]

        # Don't re-visit
        if (next_x, next_y) in visited:
            continue

        # Check if node is in range
        if next_x < 0 or next_x > mx or next_y < 0 or next_y > my:
            continue

        # Don't go through walls
        node_type = plan_matrix[next_x, next_y]
        if node_type == TileType.WALL:
            continue

        queue.put((next_x, next_y))
        previous_matrix[next_x][next_y] = np.array((node_x, node_y))

    if (node_x, node_y) == end:   # Stop when target is reached
        break

# Reconstruct the path using the previous_matrix
path = list()
point = end
while point != start:
    path.insert(0, point)
    point = tuple(previous_matrix[point])
path.insert(0, start)

print(f'Shortest path ({len(path)} steps): {" -> ".join(map(str, path))}')

print('Done')
