import numpy as np

# Function to calculate distance between two cities
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Function to initialize pheromone trails
def init_pheromone(num_cities):
    return np.ones((num_cities, num_cities))

# Function to update pheromone trails
def update_pheromone(pheromone, delta_pheromone, rho):
    return (1 - rho) * pheromone + delta_pheromone

# Function to perform ant movement
def ant_movement(num_ants, pheromone, distances, alpha, beta):
    num_cities = len(distances)
    paths = []
    for ant in range(num_ants):
        path = []
        visited = set()
        current_city = np.random.randint(num_cities)
        visited.add(current_city)
        path.append(current_city)
        while len(visited) < num_cities:
            probabilities = []
            for city in range(num_cities):
                if city not in visited:
                    pheromone_factor = pheromone[current_city][city] ** alpha
                    distance_factor = (1.0 / distances[current_city][city]) ** beta
                    probabilities.append((city, pheromone_factor * distance_factor))
            probabilities = np.array(probabilities)
            probabilities[:, 1] /= np.sum(probabilities[:, 1])
            next_city = np.random.choice(probabilities[:, 0], p=probabilities[:, 1])
            visited.add(next_city)
            path.append(int(next_city))
            current_city = int(next_city)
        paths.append(path)
    return paths

# Function to calculate total distance of a path
def total_distance(path, distances):
    total = 0
    num_cities = len(path)
    for i in range(num_cities - 1):
        total += distances[path[i]][path[i + 1]]
    total += distances[path[-1]][path[0]]
    return total

# Function to evaporate pheromone trails
def evaporate_pheromone(pheromone, evaporation_rate):
    return (1 - evaporation_rate) * pheromone

# Function to solve TSP using Ant Colony Optimization
def solve_tsp(num_cities, num_ants, iterations, alpha, beta, rho, evaporation_rate):
    cities = np.random.rand(num_cities, 2)  # Generate random cities
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = distance(cities[i], cities[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    pheromone = init_pheromone(num_cities)
    best_distance = float('inf')
    best_path = None
    
    for _ in range(iterations):
        paths = ant_movement(num_ants, pheromone, distances, alpha, beta)
        for path in paths:
            path_distance = total_distance(path, distances)
            if path_distance < best_distance:
                best_distance = path_distance
                best_path = path
        delta_pheromone = np.zeros((num_cities, num_cities))
        for path in paths:
            for i in range(num_cities - 1):
                delta_pheromone[path[i]][path[i + 1]] += 1 / total_distance(path, distances)
            delta_pheromone[path[-1]][path[0]] += 1 / total_distance(path, distances)
        pheromone = update_pheromone(pheromone, delta_pheromone, rho)
        pheromone = evaporate_pheromone(pheromone, evaporation_rate)
    
    return best_distance, best_path

# Example usage
num_cities = 20
num_ants = 10
iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.1
evaporation_rate = 0.1

best_distance, best_path = solve_tsp(num_cities, num_ants, iterations, alpha, beta, rho, evaporation_rate)
print(f"Shortest distance: {best_distance}")
print(f"Best path: {best_path}")













































































# 1. Code Solves/Satisfies the Problem Statement:

# Yes, the code implements the Ant Colony Optimization (ACO) algorithm to find a solution to the Traveling Salesman Problem (TSP). It aims to find the shortest route for a salesman to visit a set of cities and return to the starting city.

# 2. Output Correct or Not:

# The output shows:

# Shortest distance: 4.026516848065792

# Best path: [14, 16, 1, 6, 15, 4, 19, 13, 0, 7, 12, 18, 2, 9, 5, 17, 11, 8, 3, 10]

# This output is *plausible*. The ACO algorithm finds a path and its total distance. The "correctness" is relative; ACO is a heuristic, so it doesn't guarantee the *absolute* shortest path, but it usually finds a good approximation. The distance value depends on the randomly generated city coordinates.

# 3. Why the Output is Like That and What it Represents:

# The ACO algorithm simulates ants finding a path.

# Ants deposit pheromone on paths; shorter paths get more pheromone.

# The algorithm iteratively:

# Ants construct routes based on pheromone and distance.

# The best route so far is tracked.

# Pheromone is updated (increased on good routes, evaporated elsewhere).

# The output shows the best route found by the ants and its total length. The specific path ([14, 16, ... , 10]) is the order in which the cities should be visited, starting from city 14, and the distance is the sum of the distances between the cities in that order.

# 4. Line-by-Line Explanation (Short):

# import numpy as np: Import NumPy.

# def distance(city1, city2): ...: Calculate distance between two cities.

# def init_pheromone(num_cities): ...: Initialize pheromone trails.

# def update_pheromone(pheromone, delta_pheromone, rho): ...: Update pheromone levels.

# def ant_movement(num_ants, ...): ...: Simulate ant movement to create paths.

# def total_distance(path, distances): ...: Calculate the total distance of a path.

# def evaporate_pheromone(pheromone, evaporation_rate): ...: Simulate pheromone evaporation

# def solve_tsp(...): ...: Solve TSP using ACO.

# Generates city coordinates.

# Calculates distances.

# Initializes pheromone.

# Iterates: ants move, paths are evaluated, pheromone is updated.

# Returns best path and distance.

# num_cities = 20...: Set ACO parameters.

# best_distance, best_path = solve_tsp(...): Run ACO.

# print(...): Print results.

# 5. Potential Oral Questions:

# What is Ant Colony Optimization (ACO)? How is it inspired by real ants?

# Answer: ACO is a metaheuristic optimization algorithm inspired by the foraging behavior of ants. Ants deposit pheromone on their paths, and other ants tend to follow paths with higher pheromone concentrations, leading to the discovery of shortest paths.

# Explain how pheromone is used in ACO.

# Answer: Pheromone represents the attractiveness of a path. Ants deposit pheromone as they travel, and the amount of pheromone influences the probability of other ants choosing that path.  Higher pheromone = more attractive path.

# What is the Traveling Salesman Problem (TSP)?

# Answer: The Traveling Salesman Problem is a classic optimization problem where the goal is to find the shortest possible route that visits each city in a given set exactly once and returns to the starting city.

# Explain the roles of alpha and beta in ACO.

# Answer:

# alpha:  Determines the relative importance of pheromone concentration.  Higher alpha makes ants more likely to follow paths with strong pheromone.

# beta:  Determines the relative importance of the distance to the next city. Higher beta makes ants more likely to choose closer cities.

# What is pheromone evaporation, and why is it important?

# Answer: Pheromone evaporation is the gradual decrease in pheromone concentration over time. It prevents the algorithm from getting stuck in a local optimum by discouraging premature convergence on a single path and encourages exploration of new paths.

# How does the code represent cities, paths, and pheromone?

# Answer:

# Cities: Represented as 2D coordinates (x, y) using NumPy arrays.

# Paths: Represented as a list of city indices, indicating the order in which cities are visited.

# Pheromone: Represented as a 2D NumPy array (pheromone matrix), where pheromone[i][j] stores the pheromone level on the path between city i and city j.

# What are the strengths and weaknesses of ACO?

# Answer:

# Strengths:  Effective for solving TSP and similar combinatorial optimization problems, relatively easy to implement, robust to getting stuck in local optima.

# Weaknesses:  Can be slower than other algorithms for some problems, performance depends on parameter tuning, doesn't guarantee the optimal solution.

# How do you choose the parameters in ACO (e.g., number of ants, evaporation rate)?

# Answer: Parameter tuning is often done experimentally.

# Number of ants:  More ants explore more paths but increase computation.

# Evaporation rate:  High rate reduces pheromone persistence, low rate may lead to premature convergence.

# Alpha and beta:  Balance the influence of pheromone and distance.

# Can ACO be applied to other optimization problems? Give examples.

# Answer: Yes, ACO can be applied to various optimization problems, including:

# Vehicle routing

# Job scheduling

# Graph coloring

# Network routing

# What is the difference between ACO and other optimization algorithms?

# Answer: ACO is inspired by insect behavior, while other algorithms have different sources of inspiration.  For example:

# Genetic Algorithms: Inspired by biological evolution.

# Simulated Annealing: Inspired by the annealing process in metallurgy.

# ACO uses a distributed, cooperative search strategy based on pheromone communication.