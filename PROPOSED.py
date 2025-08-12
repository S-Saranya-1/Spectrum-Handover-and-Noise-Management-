import time
import numpy as np


def PROPOSED(population, objective_function, lb, ub, max_generations):
    alpha = 0.5
    beta = 1
    FADs = 0.2
    gamma = 1
    population_size, dimension = population.shape[0], population.shape[1]
    intensity = np.zeros(population_size)
    best_solution = None
    Xmin = np.matlib.repmat(np.multiply(np.ones((1, dimension)), lb), population_size, 1)
    Xmax = np.matlib.repmat(np.multiply(np.ones((1, dimension)), ub), population_size, 1)
    Convergence_curve = np.zeros((1, max_generations))
    best_fitness = float('inf')
    # Main loop
    ct = time.time()
    for generation in range(max_generations):
        CF = (1 - generation / max_generations) ** (2 * generation / max_generations)
        r = np.sqrt((generation+alpha+beta+gamma))/population_size
        if r >= 0.5:
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
        else:
            # ------------------ Detecting top predator ------------------
            for i in np.arange(1, population.shape[1 - 1] + 1).reshape(-1):
                Flag4ub = population[i, :] > ub
                Flag4lb = population[i, :] < lb
                population[i, :] = (np.multiply(population[i, :], (not (Flag4ub + Flag4lb)))) + np.multiply(ub, Flag4ub) + np.multiply(
                    lb, Flag4lb)
                intensity[i, 1] = objective_function(population[i, :])
                if intensity[i, 1] < Top_predator_fit:
                    Top_predator_fit = intensity[i, 1]
                    Top_predator_pos = population[i, :]
            # ---------------------- Marine Memory saving ----------------
            if Iter == 0:
                fit_old = intensity
                Prey_old = population
            Inx = (fit_old < intensity)
            Indx = np.matlib.repmat(Inx, 1, dimension)
            population = np.multiply(Indx, Prey_old) + np.multiply(not Indx, population)
            intensity = np.multiply(Inx, fit_old) + np.multiply(not Inx, intensity)
            fit_old = intensity
            Prey_old = population
            # ---------- Eddy formation and FADs  effect (Eq 16) -----------
            if np.random.rand() < FADs:
                U = np.random.rand(population_size, dimension) < FADs
                population = population + CF * (
                    np.multiply((Xmin + np.multiply(np.random.rand(population_size, dimension), (Xmax - Xmin))), U))
            else:
                r = np.random.rand()
                Rs = population.shape[1 - 1]
                stepsize = (FADs * (1 - r) + r) * (population[np.random.permutation(Rs), :] - population[np.random.permutation(Rs), :])
                population = population + stepsize
            Iter = Iter + 1
            Convergence_curve[Iter] = Top_predator_fit
    ct = time.time() - ct
    return best_solution, best_fitness, Convergence_curve, ct