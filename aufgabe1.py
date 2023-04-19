from typing import List, Any
from queue import Queue
from PIL import Image


class Matrix:
    """
    Utility class to make working with matrices easier
    """

    def __init__(self, data: List[List[Any]]):
        self._data = data

    def __getitem__(self, idx):
        a, b = idx
        return self._data[a][b]

    def __setitem__(self, key, value):
        a, b = key
        self._data[a][b] = value

    def size(self):
        return len(self._data), len(self._data[0])

    def __repr__(self):
        return f'Matrix()'


# Define the colors for each tile type
wall = (0, 0, 0, 255)
outside = (255, 255, 255, 0)
floor = (200, 113, 55, 255)
tea_kitchen = (0, 0, 255, 255)
professor_office = (0, 255, 0, 255)
labor = (255, 255, 0, 255)


def get_square_type(pixel: tuple):
    """
    Returns the tile type for each tile as string
    """

    return {
        wall: 'wall',
        outside: 'outside',
        floor: 'floor',
        tea_kitchen: 'tea_kitchen',
        professor_office: 'professor_office',
        labor: 'labor'
    }[pixel]


def parse_image():
    """
    Parses the image and creates a matrix of tile types.
    """

    img = Image.open('lageplan.png')
    cell_size = 420 / 21
    plan_matrix = list()

    for i in range(20):
        row = list()
        for j in range(20):
            # Get one pixel for each tile. x, y have an offset of 10 to get the color of the tile
            # and not of the border
            pixel = img.getpixel((int(i * cell_size + 10), int(j * cell_size + 10)))
            row.append(get_square_type(pixel))
        plan_matrix.append(row)

    img.close()
    return Matrix(plan_matrix)


# --- Aufgabe a ---
matrix = parse_image()
mx, my = matrix.size()

# --- Aufgabe b ---
start = (3, 17)
end = (1, 3)
adj_vec = [(-1, 0), (0, 1), (1, 0), (0, -1)]    # For finding adjacent fields
# Elements: (prev_point, dist)
# An entry in this matrix also marks the tile as visited
prev_matrix = Matrix([[(None, None) for _ in range(my)] for _ in range(mx)])
prev_matrix[start] = (start, 0)
queue = Queue()   # Keep track of tiles to process
queue.put((*start, 0))

while not queue.empty():
    x, y, dist = queue.get()

    # Enqueue neighbours
    for ox, oy in adj_vec:
        xn, yn = x + ox, y + oy
        # Check if coords are valid
        if xn >= mx or yn >= my or xn < 0 or yn < 0:
            continue

        # Don't revisit
        if (prev_point_dist := prev_matrix[xn, yn]) != (None, None):
            # Check if a shorter path to a visited tile was found
            if prev_point_dist[1] > dist + 1:
                prev_matrix[xn, yn] = ((x, y), dist + 1)
            continue

        # Don't go through walls
        if matrix[xn, yn] == 'wall':
            continue

        queue.put((xn, yn, dist + 1))
        prev_matrix[xn, yn] = ((x, y), dist + 1)

    # Process the current point
    prev_point, prev_dist = prev_matrix[x, y]
    print(f'({prev_point[0]:0>2},{prev_point[1]:0>2}) -{dist:0>2}-> ({x:0>2}, {y:0>2})')


# Go backwards from end to start to build the shortest path
path = list()
point = end
while point != start:
    path.insert(0, point)
    prev_point, prev_dist = prev_matrix[point]
    point = prev_point
path.insert(0, start)

print(f'Shortest path ({len(path)} steps): {" -> ".join(map(str, path))}')

print('Done')
