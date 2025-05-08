# A = {'x1': 0.2, 'x2': 0.4, 'x3': 0.6, 'x4': 0.8}
# B = {'x1': 0.3, 'x2': 0.5, 'x3': 0.7, 'x4': 0.9}

# R = {('x1', 'y1'): 0.2, ('x1', 'y2'): 0.4, ('x2', 'y1'): 0.6, ('x2', 'y2'): 0.8}
# S = {('x1', 'y1'): 0.3, ('x1', 'y2'): 0.5, ('x2', 'y1'): 0.7, ('x2', 'y2'): 0.9}

# def fuzzy_union(A, B):
#     union = {}
#     for key in A.keys():
#         union[key] = max(A[key], B[key])
#     return union

# # Example usage
# union_result = fuzzy_union(A, B)
# print("Union:", union_result)


# def fuzzy_intersection(A, B):
#     intersection = {}
#     for key in A.keys():
#         intersection[key] = min(A[key], B[key])
#     return intersection

# # Example usage
# intersection_result = fuzzy_intersection(A, B)
# print("Intersection:", intersection_result)


# def fuzzy_complement(A):
#     complement = {}
#     for key in A.keys():
#         complement[key] = 1 - A[key]
#     return complement

# # Example usage
# complement_result_A = fuzzy_complement(A)
# complement_result_B = fuzzy_complement(B)
# print("Complement A:", complement_result_A)
# print("Complement B:", complement_result_B)


# def fuzzy_difference(A, B):
#     difference = {}
#     for key in A.keys():
#         difference[key] = min(A[key], 1 - B[key])
#     return difference

# # Example usage
# difference_result = fuzzy_difference(A, B)
# print("Difference:", difference_result)


# def max_min_composition(R, S):
#     composition = {}
#     for (x, z) in R.keys():
#         composition[(x, z)] = 0
#         for y in [key[1] for key in R.keys() if key[0] == x]:
#             composition[(x, z)] = max(composition[(x, z)], min(R[(x, y)], S[(y, z)]))
#     return composition

# # Example usage
# composition_result = max_min_composition(R, S)
# print("Max-Min Composition:", composition_result)





# NO EXTRA STEPS REQUIRED, SIMPLY RUN THIS PY CODE 


#  Create 2 dictionaries representing 2 sets a & b and there elements
a = {'x1':0.2,'x2':0.6,'x3':0.3,'x4':0.5,'x5':0.9}
b = {'x1':0.4,'x2':0.7,'x3':0.5,'x4':0.1,'x5':0.2}

# Union operation -> Return max value by comapring corresponding elements from 2 sets
def fuzzy_union(a,b):
    union ={}
    for key in a.keys():
        union[key] = max(a[key],b[key])
    return union
print("Union :\n",fuzzy_union(a,b))


# Intersection Operation -> return min value by comparing .....
def fuzzy_intersection(a,b):
    intersection = {}
    for key in a.keys():
        intersection[key] = min(a[key],b[key])
    return intersection
print("\nIntersection:\n",fuzzy_intersection(a,b))

# Complement Operation
def complement(a): 
    comp_a ={}
    for key in a.keys():
        comp_a[key] = 1- a[key]
    return comp_a
print("\nComplement :\n",complement(a))

# Difference operation for Fuzzy set -> min(a[key],1-b[key])
def diff(a,b):
    diff ={}
    for key in a.keys():
        diff[key] = min(a[key],1-b[key])
    return diff
print("\nDifference : \n",diff(a,b))


a = {'x1': 0.2, 'x2': 0.6}
b = {'y1': 0.5, 'y2': 0.8}
# Catersian Product -> same as intersection but here are 2 variables -> x,y (x and y variables have different values )instead of only x in above case of intersection
def cart_product(a,b):
    prod = {}
    for a_key in a.keys():
        for b_key in b.keys():
            prod[(a_key,b_key)] = min(a[a_key],b[b_key])
    return prod
print("\nProduct : \n",cart_product(a,b))

# Create 2 fuzzy relations
# R: from X → Y
R = {('x1', 'y1'): 0.2,('x1', 'y2'): 0.4,('x2', 'y1'): 0.6,('x2', 'y2'): 0.8}

# S: from Y → Z
S = {('y1', 'z1'): 0.3,('y1', 'z2'): 0.5,('y2', 'z1'): 0.7,('y2', 'z2'): 0.9}

# min-max composition
def max_min_composition(R, S):
    result = {}
    for x, y1 in R:
        for y2, z in S:
            if y1 == y2:
                pair = (x, z)
                value = min(R[(x, y1)], S[(y2, z)])
                result[pair] = max(result.get(pair, 0), value)
    return result
print("\nMin Max Composition : \n",max_min_composition(R,S))