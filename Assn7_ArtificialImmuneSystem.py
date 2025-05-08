import numpy as np

# Create random antibodies
def generate_antibodies(num, size):
    return [np.random.rand(size) for _ in range(num)]

# Affinity = closeness (1 / distance)
def affinity(a, d):
    return 1 / (1 + np.linalg.norm(a - d))

# Initial data
healthy = np.array([[1.0, 2.0, 3.0], [1.1, 1.9, 3.2]])
damaged = np.array([1.2, 1.7, 2.8])
antibodies = generate_antibodies(10, 3)

# Training phase (clone top 5 + mutate)
for _ in range(2):
    scores = [(a, affinity(a, dp)) for a in antibodies for dp in healthy]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    antibodies = [a + np.random.randn(3) * 0.1 for a, _ in top]  # mutated clones

# Detection phase
best = max(antibodies, key=lambda a: affinity(a, damaged))
print("Best matching antibody (possible damage):", best)















































































# Creates a population of random antibodies (vectors).

# num: how many antibodies to create.

# size: how many features (dimensionality).

# Example: np.random.rand(3) gives a vector like [0.5, 0.8, 0.3].

# Measures similarity (affinity) between antibody a and data d.

# Uses Euclidean distance: np.linalg.norm(a - d)

# Adds 1 in denominator to avoid division by zero.

# Higher affinity = closer = better match.

# healthy: training examples (normal patterns).

# damaged: test input to detect later.

# antibodies: random initial immune cells to learn from healthy data.

# Training: Run for 2 iterations.

# Evaluate affinity of each antibody against all healthy samples.

# Select top 5 highest affinity ones.

# Clone and mutate them:

# np.random.randn(3) * 0.1: adds Gaussian noise (mutation).

# Result: Antibodies learn to match the healthy patterns better.

# Now test against a new sample (damaged).

# Select the antibody with the highest affinity (closest match).

# If it's very close, it might be considered non-damaged.

# If it's far, it indicates an anomaly.

# affinity(a, dp) compares each antibody a to each healthy data point dp.

# The highest scoring antibodies are selected and mutated (learned).

# This means antibodies are adapting to recognize healthy patterns, which is pattern recognition.


















# import numpy as np

# def create_antibody(size):
#   return np.random.rand(size) 

# def euclidean_distance(a1, a2):
#   return np.linalg.norm(a1 - a2) 

# healthy_data = np.array([[1.0, 2.0, 3.0], [1.1, 1.9, 3.2]])
# num_antibodies = 10
# antibody_population = [create_antibody(healthy_data.shape[1]) for i in range(num_antibodies)]
# # Simulate sensor data with potential damage (replace with actual data)
# damaged_data = np.array([[1.2, 1.7, 2.8], [1.4, 1.5, 3.5]])

# # Affinity (closeness)
# def affinity(antibody, datapoint):
#   distance = euclidean_distance(antibody, datapoint)
#   return 1 / (1 + distance)

# for i in range(2):
#   healthy_affinities = [affinity(ab, datapoint) for ab in antibody_population for datapoint in healthy_data]
#   # Select top 'n' antibodies based on affinity (healthy data)
#   top_antibodies = sorted(zip(antibody_population, healthy_affinities), key=lambda x: x[1], reverse=True)[:5]
#   # Clone and introduce random mutations (simplified)
#   new_population = []
#   for ab, i in top_antibodies:
#     new_population.append(ab + np.random.randn(ab.shape[0]) * 0.1)  # Introduce small mutation
#     antibody_population = new_population
# # Update antibody population
# antibody_population.extend(new_population)
# # Check affinity for damaged data
# damaged_affinities = [affinity(ab, damaged_data[0]) for ab in antibody_population]
# # Identify potential damage based on high affinity for damaged data
# potential_damage_index = damaged_affinities.index(max(damaged_affinities))
# print(len(antibody_population))
# print("Potential damage detected with antibody:", antibody_population[potential_damage_index])