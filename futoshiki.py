# Hadas Babayov 322807629
# Roni Gotlib 322805029

import sys
import numpy as np
import random
from itertools import repeat
import copy
import matplotlib.pyplot as plt

POP_SIZE = 100
CROSS_RATE = 0.5
num_of_gens = 70
num_of_restart = 2

# This function get input file, and save the dats in variables.
def get_input(file):
    # Open file.
    data = open(file, 'r')
    lines = data.readlines()
    data.close()

    # Parse data.
    N = int(lines[0])
    given_digits = int(lines[1])
    locations_and_values = []
    index = 0
    for i in range(2, len(lines)):
        if len(lines[i].split()) != 3:
            index = i
            break
        line = lines[i].split()
        locations_and_values.append([(int(line[0]) - 1, int(line[1]) - 1), int(line[2])])

    number_of_greater_than = int(lines[index])
    locations_of_constraints = []
    for i in range(index + 1, len(lines)):
        line = lines[i].split()
        locations_of_constraints.append([(int(line[0]), int(line[1])), (int(line[2]), int(line[3]))])

    return N, given_digits, locations_and_values, number_of_greater_than, locations_of_constraints


# This function initialize random board.
def init_board(N, locations_and_values):
    board = np.zeros((N, N))

    # Put the numbers from the file, in the initial matrix.
    for location, value in locations_and_values:
        i, j = location
        board[i][j] = value

    # Other locations -- random value.
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                board[i][j] = random.randint(1, N)
    return board


# This function calculate the evaluation for all population.
def evaluation(population, locations_of_constraints, N):
    evaluations = []
    # For each constraint violation we add 1. A perfect board will have a evaluation of 0.
    for board in population:
        eval = 0
        # Constraints (>)
        for tuple in locations_of_constraints:
            cell1, cell2 = tuple
            i1, j1 = cell1
            i2, j2 = cell2
            if board[i1 - 1][j1 - 1] <= board[i2 - 1][j2 - 1]:
                eval += 1

        # Count repeats in rows
        row_repeats = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                row_repeats[i][int(board[i][j] - 1)] += 1

        # Count repeats in cols
        col_repeats = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                col_repeats[int(board[i][j] - 1)][j] += 1

        for i in range(N):
            for j in range(N):
                if int(row_repeats[i][j]) > 1:
                    eval += int(row_repeats[i][j]) - 1
                if int(col_repeats[i][j]) > 1:
                    eval += int(col_repeats[i][j]) - 1

        evaluations.append(eval)
    return evaluations


def choose_parents(population, evaluations):
    # There will be a higher probability of being chosen for a better board.
    duplicates_by_eval = []
    for member, eval in zip(population, evaluations):
        number_of_duplicate = 0
        if 0 < eval < 3:
            number_of_duplicate = 1500
        if 3 <= eval < 5:
            number_of_duplicate = 1000
        if 5 <= eval < 10:
            number_of_duplicate = 180
        if 10 <= eval < 20:
            number_of_duplicate = 5
        if eval >= 20:
            number_of_duplicate = 1

        duplicates_by_eval.extend(repeat(member, number_of_duplicate))

    p1 = random.choice(duplicates_by_eval)
    p2 = random.choice(duplicates_by_eval)
    return p1, p2


# Elitism - We will pass the four best solutions to the next generation.
def get_top_four(population, evaluations):
    partition = np.argpartition(evaluations, 4)
    indexes = partition[:4]
    top_four = []
    for i in indexes:
        top_four.append(population[i])
    return top_four


def crossover_and_mutation(population, evaluations, N, locations_and_values):
    new_population = []
    # Get top four.
    top_four = get_top_four(population, evaluations)
    for member in top_four:
        new_population.append(member)

    # Produce 96 more by crossover and mutation.
    for i in range(POP_SIZE - 4):
        # Crossover.
        board = np.zeros((N, N))
        p1, p2 = choose_parents(population, evaluations)
        crossover_point = random.randint(0, N)
        for i in range(0, crossover_point):
            board[i] = p1[i]
        for i in range(crossover_point, N):
            board[i] = p2[i]

        # Mutations.
        for i in range(N):
            for j in range(N):
                if random.random() < 0.1:
                    board[i][j] = random.randint(1, N)

        # Save input numbers.
        for location, value in locations_and_values:
            i, j = location
            board[i][j] = value

        new_population.append(board)
    return new_population


# This function receives data and displays it in a graph.
def display_graph(regular_gens, darwin_gens, lamarck_gens, regular_eval, darwin_eval, lamarck_eval, ylabel, title):
    # plot lines
    plt.title(title)
    plt.plot(regular_gens, regular_eval, label="regular", color='red', linewidth=0.5)
    plt.plot(darwin_gens, darwin_eval, label="darwin", color='blue', linewidth=0.5)
    plt.plot(lamarck_gens, lamarck_eval, label="lamarck", color='green', linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def lamarck_algorithm(input_file):
    N, given_digits, locations_and_values, number_of_greater_than, locations_of_constraints = get_input(input_file)

    gen = 1
    found_solution = False
    restart = 0
    population = []
    evaluations = []
    min_evaluations = []
    ave_evaluations = []
    best_solution = []
    min_eval = float('inf')
    print('Lamarck algorithm solution :')
    while not found_solution:
        if gen > num_of_gens * N:
            break

        if gen == 1:
            for i in range(POP_SIZE):
                population.append(init_board(N, locations_and_values))

        evaluations = evaluation(population, locations_of_constraints, N)
        min_evaluations.append(min(evaluations))
        ave_evaluations.append(sum(evaluations) / len(evaluations))

        # Save the best solution.
        if min(evaluations) < min_eval:
            min_eval = min(evaluations)
            best_solution = population[np.argmin(evaluations)]

        # If found perfect solution - stop the loop and return this solution.
        if 0 in evaluations:
            found_solution = True
            sol_index = evaluations.index(0)
            solution = population[sol_index]
            print(solution)
            break

        # Optimization on the regular population - that pass to the next gen.
        new_population = crossover_and_mutation(population, evaluations, N, locations_and_values)
        for i in range(POP_SIZE):
            optimization(new_population[i], N, locations_of_constraints, locations_and_values)
        population = new_population

        # Restart.
        if not found_solution and gen == num_of_gens * N and restart < num_of_restart:
            restart += 1
            population = []
            evaluations = []
            gen = 0
        gen += 1

    if not found_solution:
        print(best_solution)
    print('Evaluation :', min_eval)

    return min_evaluations, ave_evaluations


def darwin_algorithm(input_file):
    N, given_digits, locations_and_values, number_of_greater_than, locations_of_constraints = get_input(input_file)

    gen = 1
    found_solution = False
    restart = 0
    population = []
    evaluations = []
    best_solution = []
    min_evaluations = []
    ave_evaluations = []
    population_copy = []
    min_eval = float('inf')
    print('Darwin algorithm solution :')
    while not found_solution:
        if gen > num_of_gens * N:
            break

        if gen == 1:
            for i in range(POP_SIZE):
                population.append(init_board(N, locations_and_values))
            evaluations = evaluation(population, locations_of_constraints, N)

        # Save the best solution.
        if min(evaluations) < min_eval and gen != 1:
            min_eval = min(evaluations)
            best_solution = population_copy[np.argmin(evaluations)]

        min_evaluations.append(min(evaluations))
        ave_evaluations.append(sum(evaluations) / len(evaluations))

        # If found perfect solution - stop the loop and return this solution.
        if 0 in evaluations:
            found_solution = True
            sol_index = evaluations.index(0)
            solution = population_copy[sol_index]
            print(solution)
            break

        new_population = crossover_and_mutation(population, evaluations, N, locations_and_values)
        # Optimization on copy. the population without optimization pass to the next gen.
        population_copy = copy.deepcopy(new_population)
        for i in range(POP_SIZE):
            optimization(population_copy[i], N, locations_of_constraints, locations_and_values)

        evaluations = evaluation(population_copy, locations_of_constraints, N)
        population = new_population

        # Restart.
        if not found_solution and gen == num_of_gens * N and restart < num_of_restart:
            evaluations = []
            population = []
            population_copy = []
            restart += 1
            gen = 0
        gen += 1

    if not found_solution:
        print(best_solution)
    print('Evaluation :', min_eval)

    return min_evaluations, ave_evaluations


def genetic_algorithm(input_file):
    N, given_digits, locations_and_values, number_of_greater_than, locations_of_constraints = get_input(input_file)

    gen = 1
    found_solution = False
    restart = 0
    population = []
    evaluations = []
    best_solution = []
    min_evaluations = []
    ave_evaluations = []
    min_eval = float('inf')
    print('Regular algorithm solution :')
    while not found_solution:
        if gen > num_of_gens * N:
            break

        if gen == 1:
            for i in range(POP_SIZE):
                population.append(init_board(N, locations_and_values))

        evaluations = evaluation(population, locations_of_constraints, N)

        # For graphs.
        min_evaluations.append(min(evaluations))
        ave_evaluations.append(sum(evaluations) / len(evaluations))

        # Save the best solution.
        if min(evaluations) < min_eval:
            min_eval = min(evaluations)
            best_solution = population[np.argmin(evaluations)]

        # If found perfect solution - stop the loop and return this solution.
        if 0 in evaluations:
            found_solution = True
            sol_index = evaluations.index(0)
            solution = population[sol_index]
            print(solution)
            break

        new_population = crossover_and_mutation(population, evaluations, N, locations_and_values)
        population = new_population

        # Restart.
        if not found_solution and gen == num_of_gens * N and restart < num_of_restart:
            evaluations = []
            population = []
            restart += 1
            gen = 0
        gen += 1

    if not found_solution:
        print(best_solution)
    print('Evaluation :',min_eval)

    return min_evaluations, ave_evaluations


# This function do optimizations for Lamarck and Darwin algorithms.
def optimization(board, N, locations_of_constraints, locations_and_values):
    count = 0
    for tuple in locations_of_constraints:
        cell1, cell2 = tuple
        i1, j1 = cell1
        i2, j2 = cell2
        # If a pair violates a constraint, replace placements.
        if board[i1 - 1][j1 - 1] < board[i2 - 1][j2 - 1] and count <= N:
            count += 1
            board[i1 - 1][j1 - 1], board[i2 - 1][j2 - 1] = board[i2 - 1][j2 - 1], board[i1 - 1][j1 - 1]

    # Save input numbers.
    for location, value in locations_and_values:
        i, j = location
        board[i][j] = value


if __name__ == '__main__':
    input_file = sys.argv[1]
    data = open(input_file, 'r')
    lines = data.readlines()
    data.close()
    N = int(lines[0])

    # Call the three algorithms.
    ga_min_evals, ga_ave_evals = genetic_algorithm(input_file)
    darwin_min_evals, darwin_ave_evals = darwin_algorithm(input_file)
    lamarck_min_evals, lamarck_ave_evals = lamarck_algorithm(input_file)

    # Display graphs.
    title = 'Futoshiki ' + str(N) + 'x' + str(N)
    display_graph(list(range(len(ga_min_evals))), list(range(len(darwin_min_evals))),
                  list(range(len(lamarck_min_evals))),
                  ga_min_evals, darwin_min_evals, lamarck_min_evals, 'Best evaluation', title)

    display_graph(list(range(len(ga_ave_evals))), list(range(len(darwin_ave_evals))),
                  list(range(len(lamarck_ave_evals))),
                  ga_ave_evals, darwin_ave_evals, lamarck_ave_evals, 'Average evaluation', title)
