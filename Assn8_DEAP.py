import random
from deap import creator, base, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

NGEN=40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
top10 = tools.selBest(population, k=10)
print(len(top10[0]))


















































import random
from deap import creator, base, tools, algorithms

# 1. Define the Problem and Representation
# - Fitness Function: Maximization (weights=(1.0,))
# - Individual: A list of bits (0 or 1)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 2. Set up the Toolbox
toolbox = base.Toolbox()

# - Attribute Generator: Generates a random bit (0 or 1)
# - Individual Generator: Creates an individual (list of 100 bits)
# - Population Generator: Creates a population of individuals
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# - Evaluation Function: Counts the number of 1s in an individual
def evalOneMax(individual):
    return sum(individual),  # Returns a tuple (required by DEAP)

toolbox.register("evaluate", evalOneMax)

# - Genetic Operators: Crossover, Mutation, Selection
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Bit flip mutation with probability 0.05
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection with tournament size 3

# 3. Initialize the Population
population = toolbox.population(n=300)  # Create a population of 300 individuals

# 4. Run the Genetic Algorithm
NGEN = 40  # Number of generations
for gen in range(NGEN):
    # - Variation: Apply crossover and mutation to create offspring
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    #   - cxpb: crossover probability = 0.5
    #   - mutpb: mutation probability = 0.1

    # - Evaluation: Evaluate the fitness of the offspring
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit  # Assign fitness to the offspring

    # - Selection: Select the next generation from the offspring
    population = toolbox.select(offspring, k=len(population))  # Select 'k' individuals to replace the entire population

# 5. Get the Results
top10 = tools.selBest(population, k=10)  # Get the top 10 individuals from the final population
print(len(top10[0])) #prints the length of the first individual in the top 10.

# Output:
# 100

























# Import Libraries:
# random: For generating random numbers (used in bit generation).
# deap: The DEAP library itself.
# Define the Problem and Representation:
# creator.create("FitnessMax", base.Fitness, weights=(1.0,)): Creates a fitness class named FitnessMax. The weights=(1.0,) indicates that we want to maximize the fitness, and it's a single-objective problem.
# creator.create("Individual", list, fitness=creator.FitnessMax): Creates an Individual class, which is a list of values. Each individual will have a fitness attribute of type FitnessMax.
# Set up the Toolbox:
# toolbox = base.Toolbox(): Creates a Toolbox object, which holds functions for the genetic algorithm.
# toolbox.register("attr_bool", random.randint, 0, 1): Registers a function attr_bool to generate a random integer, either 0 or 1. This will be used to create the bits of our individuals.
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100): Registers a function individual to create an individual. It uses tools.initRepeat to repeatedly call toolbox.attr_bool (our bit generator) n=100 times. This creates a list of 100 bits.
# toolbox.register("population", tools.initRepeat, list, toolbox.individual): Registers a function population to create a population. It repeatedly calls toolbox.individual to create a list of individuals.
# def evalOneMax(individual): return sum(individual),: Defines the fitness function. It counts the number of 1s in the individual (the "OneMax" problem). The comma after sum(individual) is crucial; DEAP requires fitness functions to return a tuple, even for single-objective problems.
# toolbox.register("evaluate", evalOneMax): Registers the evalOneMax function as the evaluate function in the toolbox.
# toolbox.register("mate", tools.cxTwoPoint): Registers the cxTwoPoint function for crossover (mating). This performs two-point crossover between two parents to create two offspring.
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05): Registers the mutFlipBit function for mutation. This flips each bit in an individual with a probability of 0.05.
# toolbox.register("select", tools.selTournament, tournsize=3): Registers the selTournament function for selection. This performs tournament selection, where it chooses the best individual from a tournament of size 3.
# Initialize the Population:
# population = toolbox.population(n=300): Creates an initial population of 300 individuals (each with 100 bits).
# Run the Genetic Algorithm:
# NGEN = 40: Sets the number of generations to run the algorithm for.
# The for loop iterates through each generation:
# offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1): Applies variation (crossover and mutation) to the current population to create offspring.
# cxpb=0.5: Probability of crossover is 50%.
# mutpb=0.1: Probability of mutation is 10%.
# fits = toolbox.map(toolbox.evaluate, offspring): Evaluates the fitness of each individual in the offspring. toolbox.map efficiently applies the toolbox.evaluate function (which calls evalOneMax) to each offspring.
# The next for loop assigns the calculated fitness values to the corresponding offspring.
# population = toolbox.select(offspring, k=len(population)): Selects the individuals from the offspring that will survive to the next generation. The k=len(population) argument means that the new population will have the same size as the original.
# Get the Results:
# top10 = tools.selBest(population, k=10): Selects the top 10 individuals from the final population based on their fitness.
# print(len(top10[0])): Prints the length of the first individual in the top 10. Since each individual was created with 100 bits, the output is 100.
