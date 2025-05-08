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

