# Define fuzzy sets
a = {'x1': 0.2, 'x2': 0.6, 'x3': 0.3, 'x4': 0.5, 'x5': 0.9}
b = {'x1': 0.4, 'x2': 0.7, 'x3': 0.5, 'x4': 0.1, 'x5': 0.2}

# Basic operations
union = {k: max(a[k], b[k]) for k in a}
intersection = {k: min(a[k], b[k]) for k in a}
complement = {k: 1 - a[k] for k in a}
difference = {k: min(a[k], 1 - b[k]) for k in a}

print("Union:", union)
print("Intersection:", intersection)
print("Complement of A:", complement)
print("Difference A-B:", difference)

# Cartesian product
a2 = {'x1': 0.2, 'x2': 0.6}
b2 = {'y1': 0.5, 'y2': 0.8}
product = {(i, j): min(a2[i], b2[j]) for i in a2 for j in b2}
print("Cartesian Product:", product)

# Min-Max Composition
R = {('x1', 'y1'): 0.2, ('x1', 'y2'): 0.4, ('x2', 'y1'): 0.6, ('x2', 'y2'): 0.8}
S = {('y1', 'z1'): 0.3, ('y1', 'z2'): 0.5, ('y2', 'z1'): 0.7, ('y2', 'z2'): 0.9}

composition = {}
for (x, y1), r_val in R.items():
    for (y2, z), s_val in S.items():
        if y1 == y2:
            composition[(x, z)] = max(composition.get((x, z), 0), min(r_val, s_val))
print("Min-Max Composition:", composition)
