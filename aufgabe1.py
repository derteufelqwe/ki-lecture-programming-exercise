from typing import List, Any, Optional
from queue import Queue, PriorityQueue
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
    PROFESSOR_OFFICE = 'professor_office'
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


class TileCost:
    """
    Cost for A* algorithm
    """

    WALL = 1
    OUTSIDE = 1
    FLOOR = 2
    TEA_KITCHEN = 3
    PROFESSOR_OFFICE = 4
    LABOR = 5

    @staticmethod
    def of(tile: TileType):
        """
        Pixel to tile type
        """

        return {
            TileType.WALL: TileCost.WALL,
            TileType.OUTSIDE: TileCost.OUTSIDE,
            TileType.FLOOR: TileCost.FLOOR,
            TileType.TEA_KITCHEN: TileCost.TEA_KITCHEN,
            TileType.PROFESSOR_OFFICE: TileCost.PROFESSOR_OFFICE,
            TileType.LABOR: TileCost.PROFESSOR_OFFICE,
        }[tile]


class Node:
    """
    To build a search tree
    """

    def __init__(self, parent: Optional['Node'], position):
        self.parent = parent
        self.position = position
        self.children = list()

    def add_child(self, node):
        self.children.append(node)

    def __gt__(self, other):
        """ Must be comparable for the priority queue """
        return 0

    def __lt__(self, other):
        """ Must be comparable for the priority queue """
        return 0

    def __repr__(self):
        return f'Node({self.position})'


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
def breadth_first_search():
    start = (3, 17)
    end = (1, 3)
    adj_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]    # For finding adjacent fields
    queue = Queue()
    queue.put(start)
    # Make the path is reconstructable
    previous_matrix = np.array([[(None, None) for _ in range(21)] for _ in range(21)])
    visited = set()   # Store already visited nodes, otherwise performance is terrible
    cnt = 0

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

        # print(f"{cnt}: {(node_x, node_y)}")
        cnt += 1

        if (node_x, node_y) == end:   # Stop when target is reached
            break

    # Reconstruct the path using the previous_matrix
    path = list()
    point = end
    while point != start:
        path.insert(0, point)
        point = tuple(previous_matrix[point])
    path.insert(0, start)

    print(f'Searched nodes : {cnt}')
    print(f'Solution length: {len(path)}')
    print(f'Shortest path  : ' + " -> ".join(map(str, path)))


# --- Aufgabe c ---
def a_star_heuristic(pos: tuple, end: tuple):
    """
    Goes directly to the end position and measures the distance.
    The heuristic is valid and consistent.
    :param pos: Current position
    :param end: Target position
    :return: The cost
    """
    sign = lambda x: -1 if x < 0 else 1

    cost = 0
    point = pos

    while point != end:
        x_diff = end[0] - point[0]
        y_diff = end[1] - point[1]

        if abs(x_diff) > abs(y_diff):
            x_increment = sign(x_diff)
            point = (point[0] + x_increment, point[1])
        else:
            y_increment = sign(y_diff)
            point = (point[0], point[1] + y_increment)

        cost += TileCost.of(plan_matrix[point])

    return cost


def a_star_search():
    start = (3, 17)
    end = (1, 3)
    adj_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # For finding adjacent fields
    root = Node(None, start)
    queue = PriorityQueue()
    # Queue elements: (Estimated Cost, node, current cost)
    queue.put((0 + a_star_heuristic(start, end), root, 0))
    cnt = 0

    while not queue.empty():
        _, node, current_cost = queue.get()
        node_pos_x, node_pos_y = node.position

        # Enqueue neighbours
        for adj_vec in adj_vectors:
            next_x, next_y = node_pos_x + adj_vec[0], node_pos_y + adj_vec[1]

            # Check if node is in range
            if next_x < 0 or next_x > mx or next_y < 0 or next_y > my:
                continue

            # Don't go through walls
            node_type = plan_matrix[next_x, next_y]
            if node_type == TileType.WALL:
                continue

            tile_cost = TileCost.of(node_type)
            child_node = Node(node, (next_x, next_y))
            node.add_child(child_node)
            queue.put((
                current_cost + a_star_heuristic((next_x, next_y), end),  # Estimated cost
                child_node,     # The node
                current_cost + tile_cost    # Current cost
            ))

        # print(f'{cnt}: {node}')
        cnt += 1

        if (node_pos_x, node_pos_y) == end:   # Stop when target is reached
            break

    path = list()
    parent = node
    while parent is not None:
        path.insert(0, parent)
        parent = parent.parent

    print(f'Searched nodes : {cnt}')
    print(f'Solution length: {len(path)}')
    print(f'Shortest path  : ' + ' -> '.join(map(str, [n.position for n in path])))

    return


print('--- BFS ---')
breadth_first_search()
print()

print('--- A* ---')
a_star_search()
print()


print('Done')
