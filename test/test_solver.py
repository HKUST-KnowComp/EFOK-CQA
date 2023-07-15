import copy

from constraint import *

# Create the problem instance
problem = Problem()

# Define the variables and their domains
problem.addVariables(["f1", "f2", "f3"], range(1, 6))

# Define the constraint using InSetConstraint
a_constraint = {(1, 2), (3, 4), (5, 4)}
b_constraint = {(2, 2)}
constraint_list = [{(1, 2), (3, 4), (5, 4)}, {(2, 2)}]
for i, constraint in enumerate(constraint_list):
    if i == 0:
        problem.addConstraint(lambda x, y: (x, y) in copy.deepcopy(constraint_list[i]), ["f1", "f2"])
    if i == 1:
        problem.addConstraint(lambda x, y: (x, y) in copy.deepcopy(constraint_list[i]), ["f2", "f3"])
# Find a solution to the problem
solution = problem.getSolutions()

# Print the solution
print(solution)