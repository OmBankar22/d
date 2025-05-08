# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from geneticalgorithm import geneticalgorithm as ga
import warnings

# Define functions for genetic algorithm fitness evaluation and neural network training
def fitness_function(params):
    # Decode parameters
    hidden_layer_sizes = (int(params[0]),) * int(params[1])
    activation = ['identity', 'logistic', 'tanh', 'relu'][int(params[2])]
    solver = ['lbfgs', 'sgd', 'adam'][int(params[3])]
    
    # Create and train neural network
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)
    model.fit(X_train, y_train)
    
    # Evaluate fitness (example: mean squared error)
    fitness = -model.score(X_val, y_val)
    
    return fitness

def train_neural_network(params):
    # Decode parameters
    hidden_layer_sizes = (int(params[0]),) * int(params[1])
    activation = ['identity', 'logistic', 'tanh', 'relu'][int(params[2])]
    solver = ['lbfgs', 'sgd', 'adam'][int(params[3])]
    
    # Create and train neural network
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)
    model.fit(X_train, y_train)
    
    # Evaluate neural network performance on validation data
    validation_error = -model.score(X_val, y_val)
    
    return validation_error

# Load your dataset
np.random.seed(42)

# Assuming you have features like temperature, humidity, etc., and you want to predict drying time
num_samples = 1000
num_features = 5  # Adjust based on your actual features

# Generate synthetic features
X = np.random.rand(num_samples, num_features)

# Generate synthetic target variable (drying time)
# Here, we'll assume a linear relationship with some noise
true_coefficients = np.random.rand(num_features) * 10  # Random coefficients for features
noise = np.random.normal(loc=0, scale=1, size=num_samples)  # Gaussian noise
y = np.dot(X, true_coefficients) + noise


# Example: Replace X_train, X_val, y_train, y_val with your dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

warnings.filterwarnings("ignore")

# Define genetic algorithm parameters
number_of_parameters = 4  # Adjust based on the number of parameters you want to optimize
varbound = np.array([[5, 50],  # Range for number of neurons in each hidden layer
                     [1, 5],    # Range for number of hidden layers
                     [0, 3],    # Activation function: 0 for identity, 1 for logistic, 2 for tanh, 3 for relu
                     [0, 2]])   # Solver: 0 for lbfgs, 1 for sgd, 2 for adam
algorithm_param = {'max_num_iteration': 1, 'population_size': 100, 'mutation_probability': 0.1,
                   'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3,
                   'crossover_type': 'uniform', 'max_iteration_without_improv': None}
model = ga(function=fitness_function, dimension=number_of_parameters, variable_type='int', variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

# Run the genetic algorithm
model.run()

# Get the best parameters found by the genetic algorithm
best_params = model.output_dict['variable']

# Train neural network with the best parameters
validation_error = train_neural_network(best_params)

print("Best parameters found by genetic algorithm:", best_params)
print("Validation error of neural network:", validation_error)

# Optionally, you can further evaluate the neural network on a test set
# and perform any additional analysis or visualization




















































































# 1. Code Solves/Satisfies the Problem Statement:

# Yes, the code aims to optimize the parameters of a neural network using a genetic algorithm. This is a hybrid approach, as it combines an evolutionary optimization technique (GA) with a machine learning model (NN).

# The code defines a fitness_function that trains a neural network with given parameters and returns its performance (in this case, the negative of the R2 score). The genetic algorithm tries to minimize this fitness function.

# The train_neural_network function also trains a neural network with given parameters and returns the validation error.

# The code sets up a genetic algorithm to optimize four parameters: the number of neurons per hidden layer, the number of hidden layers, the activation function, and the solver for the neural network.

# The code then runs the genetic algorithm, retrieves the best parameters, trains a final neural network with those parameters, and prints the validation error.

# 2. Output Correct or Not:

# The code appears to run correctly.

# The image shows that the genetic algorithm found a solution with an objective function value of approximately -0.954.

# The code then trains a neural network with the parameters [32, 3, 3, 1] and reports a validation error of -0.9536.

# The negative values for the objective function and validation error indicate that the  R² score is being used as the performance metric, and the GA is minimizing the negative of the  R² score.  An R2 score close to 1 indicates a good fit, so values near -1 indicate poor fit.

# 3. Why the Output is Like That and What it Represents:

# Genetic Algorithm:

# The genetic algorithm explores the parameter space to find a combination of neural network parameters that results in the best performance.

# The "Objective function" in the output refers to the fitness function that the GA is trying to minimize. In this case, it's the negative of the R² score.  The graph shows how the best objective function value evolves over iterations (though in the provided image, it seems the GA ran for only one iteration).

# The best solution found by the GA is [32. 3. 3. 1.].  These values correspond to:

# 32 neurons per hidden layer.

# 3 hidden layers.

# Activation function index 3 (which is 'relu').

# Solver index 1 (which is 'sgd').

# Neural Network Training:

# After the GA finds the best parameters, the code trains a neural network using those parameters.

# The "Validation error of neural network" is -0.9536.  This is the negative R² score of the neural network trained with the optimized parameters on the validation set.

# A validation error of -0.9536 suggests the model is performing poorly.

# 4. Line-by-Line Explanation (Short):

# import numpy as np...: Imports necessary libraries.

# def fitness_function(params):: Defines the fitness function for the GA.

# hidden_layer_sizes = (int(params[0]),) * int(params[1]): Decodes the number of neurons and layers from the parameter vector.

# activation = ['identity', 'logistic', 'tanh', 'relu'][int(params[2])]: Decodes the activation function.

# solver = ['lbfgs', 'sgd', 'adam'][int(params[3])]: Decodes the solver.

# model = MLPRegressor(...): Creates an MLPRegressor (neural network) model.

# model.fit(X_train, y_train): Trains the neural network.

# fitness = -model.score(X_val, y_val): Calculates the negative R² score (fitness).

# def train_neural_network(params):: Defines a function to train the neural network and get the validation error

# The lines in  train_neural_network are similar to fitness_function but return the validation error.

# np.random.seed(42):  Sets the random seed for reproducibility.

# X = np.random.rand(num_samples, num_features):  Generates synthetic feature data.

# true_coefficients = np.random.rand(num_features) * 10: Generates random coefficients.

# noise = np.random.normal(loc=0, scale=1, size=num_samples): Generates Gaussian noise.

# y = np.dot(X, true_coefficients) + noise: Generates synthetic target variable (drying time).

# X_train, X_val, y_train, y_val = train_test_split(...): Splits the data into training and validation sets.

# number_of_parameters = 4: Defines the number of parameters to optimize.

# varbound = np.array([...]): Defines the bounds for each parameter.

# algorithm_param = {...}: Sets the parameters for the genetic algorithm.

# model = ga(...): Creates a genetic algorithm model.

# model.run(): Runs the genetic algorithm optimization.

# best_params = model.output_dict['variable']: Gets the best parameters found.

# validation_error = train_neural_network(best_params): Trains the NN with the best parameters.

# print(...): Prints the results.

# 5. Potential Oral Questions:

# Here are some potential oral questions:

# Hybrid Modeling:

# What is hybrid modeling? Why combine a genetic algorithm with a neural network?

# What are the advantages of using a genetic algorithm for parameter optimization?

# What are the limitations of this approach?

# Genetic Algorithms:

# Explain the basic principles of genetic algorithms (selection, crossover, mutation).

# What are the key parameters of a genetic algorithm (population size, mutation rate, etc.)?

# What is the role of the fitness function?  How is it defined in this code?

# What are the different types of crossover and mutation?

# Neural Networks:

# Explain the basic architecture of a multi-layer perceptron (MLP).

# What are activation functions?  What are the common types, and which ones are used here?

# What are different optimization algorithms (solvers) for neural networks (lbfgs, sgd, adam)?

# What is the difference between training, validation, and testing data?

# What does the R2 score measure?

# Code Details:

# Explain how the parameters are encoded for the genetic algorithm.

# Explain the purpose of the fitness_function.  Why is the R² score negated?

# How are the neural network parameters (number of layers, neurons, etc.) determined from the GA's output?

# What are the ranges specified in the varbound array?

# What do the different values in the best_params array represent?

# Application to Spray Drying:

# How can this model be applied to the spray drying of coconut milk?

# What are the important input parameters in the spray drying process?

# What is the desired output or quality parameter to be predicted?

# How can this model help in optimizing the spray drying process?

# Output Interpretation:

# Explain the output of the code.  What do the "Best parameters" and "Validation error" mean?

# What does the value of the validation error (-0.9536) indicate about the model's performance?

# How could the model's performance be improved?