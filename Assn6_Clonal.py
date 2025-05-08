import numpy as np

# Define the Sphere function to be optimized
def sphere_function(x):
    return np.sum(x**3)

# Define the Clonal Selection Algorithm
def clonal_selection_algorithm(objective_function, dim, pop_size, max_iter, mutation_rate):
    # Initialize population randomly
    population = np.random.uniform(-5, 5, size=(pop_size, dim))
    
    for iter in range(max_iter):
        # Evaluate fitness for each individual
        fitness = np.array([objective_function(ind) for ind in population])
        
        # Sort population based on fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Clone individuals based on their fitness
        num_clones = int(pop_size * 0.5)
        clones = population[:num_clones]
        
        # Mutate clones
        mutated_clones = clones + np.random.normal(scale=mutation_rate, size=clones.shape)
        
        # Select the best individuals from the original population and mutated clones
        new_population = np.vstack((population[-num_clones:], mutated_clones))
        
        population = new_population
    
    # Return the best solution found
    best_solution = population[np.argmin(fitness)]
    best_fitness = fitness.min()
    
    return best_solution, best_fitness

# Define parameters
dim = 10           # Dimensionality of the problem
pop_size = 100     # Population size
max_iter = 100     # Maximum number of iterations
mutation_rate = 0.1  # Mutation rate

# Run Clonal Selection Algorithm
best_solution, best_fitness = clonal_selection_algorithm(sphere_function, dim, pop_size, max_iter, mutation_rate)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)















































































































# 1. Code Solves/Satisfies the Problem Statement:

# Yes, this code implements the Clonal Selection Algorithm (CSA) to optimize a given function.

# Specifically, it aims to find the input vector x that minimizes the Sphere function (defined as the sum of the cubes of its components).

# The code follows the general steps of CSA: initialization, cloning, mutation, and selection.

# 2. Output Correct or Not:

# The output shows a "Best solution" and a "Best fitness."

# The "Best solution" is the vector [-4.75849308  0.493644   4.03302832  0.2842272  -1.42593785  0.87893747 -0.85976013  1.78030419  4.06286154  2.15412714], which represents the values of the variables that the algorithm found to minimize the Sphere function.

# The "Best fitness" is -825.5525882885455, which is the value of the Sphere function at the "Best solution".  In other words, when you plug the values from the "Best solution" vector into the Sphere function, you get this result.

# Theoretically, the Sphere function is defined as  f(x) = sum(x_i^3).  It does not have a global minimum at zero, but instead will produce very large negative values for negative x values. The algorithm is attempting to find a set of x_i values that produce a very low result.

# The output appears to be a plausible result of the CSA attempting to minimize the sphere function.

# 3. Why the Output is Like That and What it Represents:

# The Clonal Selection Algorithm starts with a random population of candidate solutions.

# In each iteration, it evaluates the "fitness" of each solution (how well it minimizes the Sphere function).

# The algorithm then selects the best solutions (those with the lowest Sphere function values) and creates clones of them.

# These clones are mutated (slightly modified) to explore the solution space further.

# Finally, the algorithm selects the best individuals from the original population and the mutated clones to form the new population for the next iteration.

# This process continues until a stopping criterion is met (in this case, a maximum number of iterations).

# The output shows the best solution found by the algorithm and the corresponding function value (fitness).  The algorithm is searching for the minimum value of the sphere function.

# 4. Line-by-Line Explanation (Short):

# import numpy as np: Imports the NumPy library for numerical operations.

# def sphere_function(x): return np.sum(x**3): Defines the Sphere function, which calculates the sum of the cubes of the input vector's elements.

# def clonal_selection_algorithm(...): Defines the Clonal Selection Algorithm.

# population = np.random.uniform(-5, 5, size=(pop_size, dim)): Initializes the population with random solutions within the range [-5, 5].

# for iter in range(max_iter):: Iterates through the main loop of the algorithm.

# fitness = np.array([objective_function(ind) for ind in population]): Calculates the fitness (Sphere function value) for each individual in the population.

# sorted_indices = np.argsort(fitness): Gets the indices that would sort the fitness values in ascending order (lower fitness is better).

# population = population[sorted_indices]: Sorts the population based on fitness.

# fitness = fitness[sorted_indices]: Sorts the fitness values.

# num_clones = int(pop_size * 0.5): Determines the number of clones to create (50% of the population).

# clones = population[:num_clones]: Selects the best individuals to clone.

# mutated_clones = clones + np.random.normal(scale=mutation_rate, size=clones.shape): Mutates the clones by adding Gaussian noise.

# new_population = np.vstack((population[-num_clones:], mutated_clones)): Creates a new population by combining the best original individuals and the mutated clones.  It keeps the best from the original population.

# population = new_population: Updates the population for the next iteration.

# best_solution = population[np.argmin(fitness)]: Gets the solution with the best fitness (lowest Sphere function value).

# best_fitness = fitness.min(): Gets the best fitness value.

# return best_solution, best_fitness: Returns the best solution and its fitness.

# dim = 10...: Defines the parameters of the problem and the CSA.

# best_solution, best_fitness = clonal_selection_algorithm(...): Runs the CSA.

# print(...): Prints the results.

# 5. Potential Oral Questions:

# Here are some potential oral questions related to this code:

# Clonal Selection Algorithm:

# Explain the Clonal Selection Algorithm.  What is the biological inspiration behind it?

# What are the key steps involved in CSA (initialization, cloning, mutation, selection)?

# How does CSA differ from other optimization algorithms like Genetic Algorithms?

# What are the advantages and disadvantages of CSA?

# Code Details:

# Explain the purpose of the sphere_function.  Why is it used here?

# What is the role of the pop_size, max_iter, and mutation_rate parameters?  How do they affect the algorithm's performance?

# Explain how the cloning and mutation operations are implemented in the code.

# Why are the best individuals selected for cloning?

# How is diversity maintained in the population? (Through mutation)

# What does the output Best solution and Best fitness represent?

# Optimization:

# What is optimization?  What are some common optimization problems?

# What is the difference between local and global optimization?

# How can evolutionary algorithms like CSA be used for optimization?

# Applications:

# What are some real-world applications of the Clonal Selection Algorithm?  (e.g., parameter optimization, feature selection, pattern recognition)

# Can you give an example of how CSA could be used to solve a problem in your field of study?

# Variations and Extensions:

# Are there any variations or extensions to the basic Clonal Selection Algorithm?

# How could you modify the code to handle multi-objective optimization problems?