import time
import numpy as np

def FFA(population,objective_function,lb,ub,max_generations):
    alpha = 0.5
    beta = 1
    gamma = 1
    population_size,dimension = population.shape[0],population.shape[1]
    intensity = np.zeros(population_size)
    best_solution = None
    Convergence_curve = np.zeros((1, max_generations))
    best_fitness = float('inf')
    # Main loop
    ct = time.time()
    r = np.sqrt()
    for generation in range(max_generations):
        # Evaluate fitness and update best solution
        for i in range(population_size):
            intensity[i] = objective_function(population[i])

            if intensity[i] < best_fitness:
                best_fitness = intensity[i]
                best_solution = population[i].copy()
        # Update fireflies' positions
        for i in range(population_size):
            for j in range(population_size):
                if intensity[i] > intensity[j]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta_new = beta * np.exp(-gamma * r ** 2)
                    population[i] += alpha * (population[j] - population[i]) + beta_new * np.random.uniform(low=-1,high=1,size=dimension)
        # Apply boundary constraints
        population = np.clip(population, lb, ub)
        Convergence_curve[generation] = best_fitness
    ct=time.time()-ct
    return best_solution, best_fitness,Convergence_curve,ct