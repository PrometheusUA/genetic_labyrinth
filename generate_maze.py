# -*- coding: utf-8 -*-
"""
Maze generator
"""

import random

from colorama import init

# print a maze
def printMaze(maze):
	for i in range(0, height):
		for j in range(0, width):
			if (maze[i][j] == 2):
				print(str(maze[i][j]), end=" ")
			elif (maze[i][j] == 0):
				print(str(maze[i][j]), end=" ")
			else:
				print(str(maze[i][j]), end=" ")

		print('\n')
	print(str('results: '))

# Find the number of surrounding cells
def surroundingCells(rand_wall):
	s_cells = 0
	if (maze[rand_wall[0]-1][rand_wall[1]] == 'c'):
		s_cells += 1
	if (maze[rand_wall[0]+1][rand_wall[1]] == 'c'):
		s_cells += 1
	if (maze[rand_wall[0]][rand_wall[1]-1] == 'c'):
		s_cells +=1
	if (maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
		s_cells += 1

	return s_cells

wall = 'w'
cell = 'c'
unvisited = 'u'
height = 6
width = 6
maze = []

# Init the maze with all cells marked as unvisited
for i in range(0, height):
	line = []
	for j in range(0, width):
		line.append(unvisited)
	maze.append(line)

# Find a random starting point
starting_height = int(random.random()*height)
starting_width = int(random.random()*width)
if (starting_height == 0):
	starting_height += 1
if (starting_height == height-1):
	starting_height -= 1
if (starting_width == 0):
	starting_width += 1
if (starting_width == width-1):
	starting_width -= 1

# Mark it as cell, make walls around and add surrounding walls to the list
maze[starting_height][starting_width] = cell
walls = []
walls.append([starting_height - 1, starting_width])
walls.append([starting_height, starting_width - 1])
walls.append([starting_height, starting_width + 1])
walls.append([starting_height + 1, starting_width])

# Add walls to maze
maze[starting_height-1][starting_width] = 'w'
maze[starting_height][starting_width - 1] = 'w'
maze[starting_height][starting_width + 1] = 'w'
maze[starting_height + 1][starting_width] = 'w'

while (walls):
	# Pick a random wall
	rand_wall = walls[int(random.random()*len(walls))-1]

	# Check if it is a left wall
	if (rand_wall[1] != 0):
		if (maze[rand_wall[0]][rand_wall[1]-1] == 'u' and maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
			# Find the number of surrounding cells
			s_cells = surroundingCells(rand_wall)

			if (s_cells < 2):
				# Find the new path
				maze[rand_wall[0]][rand_wall[1]] = 'c'

				# Mark the new walls
				# Upper cell
				if (rand_wall[0] != 0):
					if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
						maze[rand_wall[0]-1][rand_wall[1]] = 'w'
					if ([rand_wall[0]-1, rand_wall[1]] not in walls):
						walls.append([rand_wall[0]-1, rand_wall[1]])


				# Bottom cell
				if (rand_wall[0] != height-1):
					if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
						maze[rand_wall[0]+1][rand_wall[1]] = 'w'
					if ([rand_wall[0]+1, rand_wall[1]] not in walls):
						walls.append([rand_wall[0]+1, rand_wall[1]])

				# Left cell
				if (rand_wall[1] != 0):	
					if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
						maze[rand_wall[0]][rand_wall[1]-1] = 'w'
					if ([rand_wall[0], rand_wall[1]-1] not in walls):
						walls.append([rand_wall[0], rand_wall[1]-1])
			

			# Delete wall
			for wall in walls:
				if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
					walls.remove(wall)

			continue

	# Check if it is an upper wall
	if (rand_wall[0] != 0):
		if (maze[rand_wall[0]-1][rand_wall[1]] == 'u' and maze[rand_wall[0]+1][rand_wall[1]] == 'c'):

			s_cells = surroundingCells(rand_wall)
			if (s_cells < 2):
				# Find the new path
				maze[rand_wall[0]][rand_wall[1]] = 'c'

				# Mark the new walls
				# Upper cell
				if (rand_wall[0] != 0):
					if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
						maze[rand_wall[0]-1][rand_wall[1]] = 'w'
					if ([rand_wall[0]-1, rand_wall[1]] not in walls):
						walls.append([rand_wall[0]-1, rand_wall[1]])

				# Left cell
				if (rand_wall[1] != 0):
					if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
						maze[rand_wall[0]][rand_wall[1]-1] = 'w'
					if ([rand_wall[0], rand_wall[1]-1] not in walls):
						walls.append([rand_wall[0], rand_wall[1]-1])

				# Right cell
				if (rand_wall[1] != width-1):
					if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
						maze[rand_wall[0]][rand_wall[1]+1] = 'w'
					if ([rand_wall[0], rand_wall[1]+1] not in walls):
						walls.append([rand_wall[0], rand_wall[1]+1])

			# Delete wall
			for wall in walls:
				if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
					walls.remove(wall)

			continue

	# Check the bottom wall
	if (rand_wall[0] != height-1):
		if (maze[rand_wall[0]+1][rand_wall[1]] == 'u' and maze[rand_wall[0]-1][rand_wall[1]] == 'c'):

			s_cells = surroundingCells(rand_wall)
			if (s_cells < 2):
				# Find the new path
				maze[rand_wall[0]][rand_wall[1]] = 'c'

				# Mark the new walls
				if (rand_wall[0] != height-1):
					if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
						maze[rand_wall[0]+1][rand_wall[1]] = 'w'
					if ([rand_wall[0]+1, rand_wall[1]] not in walls):
						walls.append([rand_wall[0]+1, rand_wall[1]])
				if (rand_wall[1] != 0):
					if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
						maze[rand_wall[0]][rand_wall[1]-1] = 'w'
					if ([rand_wall[0], rand_wall[1]-1] not in walls):
						walls.append([rand_wall[0], rand_wall[1]-1])
				if (rand_wall[1] != width-1):
					if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
						maze[rand_wall[0]][rand_wall[1]+1] = 'w'
					if ([rand_wall[0], rand_wall[1]+1] not in walls):
						walls.append([rand_wall[0], rand_wall[1]+1])

			# Delete wall
			for wall in walls:
				if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
					walls.remove(wall)


			continue

	# Check the right wall
	if (rand_wall[1] != width-1):
		if (maze[rand_wall[0]][rand_wall[1]+1] == 'u' and maze[rand_wall[0]][rand_wall[1]-1] == 'c'):

			s_cells = surroundingCells(rand_wall)
			if (s_cells < 2):
				# Find the new path
				maze[rand_wall[0]][rand_wall[1]] = 'c'

				# Mark the new walls
				if (rand_wall[1] != width-1):
					if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
						maze[rand_wall[0]][rand_wall[1]+1] = 'w'
					if ([rand_wall[0], rand_wall[1]+1] not in walls):
						walls.append([rand_wall[0], rand_wall[1]+1])
				if (rand_wall[0] != height-1):
					if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
						maze[rand_wall[0]+1][rand_wall[1]] = 'w'
					if ([rand_wall[0]+1, rand_wall[1]] not in walls):
						walls.append([rand_wall[0]+1, rand_wall[1]])
				if (rand_wall[0] != 0):	
					if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
						maze[rand_wall[0]-1][rand_wall[1]] = 'w'
					if ([rand_wall[0]-1, rand_wall[1]] not in walls):
						walls.append([rand_wall[0]-1, rand_wall[1]])

			# Delete wall
			for wall in walls:
				if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
					walls.remove(wall)

			continue

	# Delete the wall from the list anyway
	for wall in walls:
		if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
			walls.remove(wall)
	


# Mark the rest of unvisited cells as walls
for i in range(0, height):
	for j in range(0, width):
		if (maze[i][j] == 'u'):
			maze[i][j] = 'w'

# Set entrance and exit
for i in range(0, width):
	if (maze[1][i] == 'c'):
		maze[0][i] = 'c'
		break

for i in range(width-1, 0, -1):
	if (maze[height-2][i] == 'c'):
		maze[height-1][i] = 'c'
		break

# Represent the maze with 0 and 1; enter/exit - 2
def normalize_maze(maze):
    for i in range(0, height):
    	for j in range(0, width):
            if (j == 0) | (j == width - 1) | (i == 0) | (i == height - 1):
                if (maze[i][j] == 'c'):
                    maze[i][j] = 2
                else: maze[i][j] = 1
            else:
                if (maze[i][j] == 'c'):
                    maze[i][j] = 0
                else: maze[i][j] = 1
    return maze
        
# Print result maze
res = normalize_maze(maze)
printMaze(res)
def get_maze():
	return res