from platypus import NSGAII, Problem, Real
from generate_fis import evaluate_fuzzy,initialize_fuzzy

initialize_fuzzy(0)
problem = Problem(24, 1)
problem.types[:] = Real(0, 10)
problem.function = evaluate_fuzzy

algorithm = NSGAII(problem)
algorithm.run(2)