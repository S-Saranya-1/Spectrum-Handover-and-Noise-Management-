import time

import numpy as np
import numpy.matlib
from numpy import inf
from scipy.stats import levy


def MPA(Prey,fobj,lb ,ub,Max_iter):
    SearchAgents_no, dim = Prey.shape[0], Prey.shape[1]
    Top_predator_pos = np.zeros((1,dim))
    Top_predator_fit = inf
    Convergence_curve = np.zeros((1,Max_iter))
    stepsize = np.zeros((SearchAgents_no,dim))
    fitness = inf(SearchAgents_no,1)
   
    Xmin = np.matlib.repmat(np.multiply(np.ones((1,dim)),lb),SearchAgents_no,1)
    Xmax = np.matlib.repmat(np.multiply(np.ones((1,dim)),ub),SearchAgents_no,1)
    Iter = 0
    FADs = 0.2
    P = 0.5
    ct = time.time()
    while Iter < Max_iter:
        #------------------- Detecting top predator -----------------
        for i in np.arange(1,Prey.shape[1-1]+1).reshape(-1):
            Flag4ub = Prey[i,:] > ub
            Flag4lb = Prey[i,:] < lb
            Prey[i,:] = (np.multiply(Prey[i,:],(not (Flag4ub + Flag4lb) ))) + np.multiply(ub,Flag4ub) + np.multiply(lb,Flag4lb)
            fitness[i,1] = fobj(Prey[i,:])
            if fitness(i,1) < Top_predator_fit:
                Top_predator_fit = fitness[i,1]
                Top_predator_pos = Prey[i,:]
        #------------------- Marine Memory saving -------------------
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey
        Inx = (fit_old < fitness)
        Indx = np.matlib.repmat(Inx,1,dim)
        Prey = np.multiply(Indx,Prey_old) + np.multiply(not Indx ,Prey)
        fitness = np.multiply(Inx,fit_old) + np.multiply(not Inx ,fitness)
        fit_old = fitness
        Prey_old = Prey
        #------------------------------------------------------------
        Elite = np.matlib.repmat(Top_predator_pos,SearchAgents_no,1)
        CF = (1 - Iter / Max_iter) ** (2 * Iter / Max_iter)
        RL = 0.05 * levy(SearchAgents_no,dim,1.5)
        RB = np.random.randn(SearchAgents_no,dim)
        for i in np.arange(1,Prey.shape[1-1]+1).reshape(-1):
            for j in np.arange(1,Prey.shape[2-1]+1).reshape(-1):
                R = np.random.rand()
                #------------------ Phase 1 (Eq.12) -------------------
                if Iter < Max_iter / 3:
                    stepsize[i,j] = RB[i,j] * (Elite[i,j] - RB[i,j] * Prey[i,j])
                    Prey[i,j] = Prey(i,j) + P * R * stepsize[i,j]
                    #--------------- Phase 2 (Eqs. 13 & 14)----------------
                else:
                    if Iter > Max_iter / 3 and Iter < 2 * Max_iter / 3:
                        if i > Prey.shape[1-1] / 2:
                            stepsize[i,j] = RB[i,j] * (RB[i,j] * Elite[i,j] - Prey(i,j))
                            Prey[i,j] = Elite[i,j] + P * CF * stepsize[i,j]
                        else:
                            stepsize[i,j] = RL(i,j) * (Elite[i,j] - RL(i,j) * Prey(i,j))
                            Prey[i,j] = Prey(i,j) + P * R * stepsize[i,j]
                        #----------------- Phase 3 (Eq. 15)-------------------
                    else:
                        stepsize[i,j] = RL(i,j) * (RL(i,j) * Elite[i,j] - Prey[i,j])
                        Prey[i,j] = Elite[i,j] + P * CF * stepsize[i,j]
        #------------------ Detecting top predator ------------------
        for i in np.arange(1,Prey.shape[1-1]+1).reshape(-1):
            Flag4ub = Prey[i,:] > ub
            Flag4lb = Prey[i,:] < lb
            Prey[i,:] = (np.multiply(Prey[i,:],(not (Flag4ub + Flag4lb) ))) + np.multiply(ub,Flag4ub) + np.multiply(lb,Flag4lb)
            fitness[i,1] = fobj(Prey[i,:])
            if fitness(i,1) < Top_predator_fit:
                Top_predator_fit = fitness(i,1)
                Top_predator_pos = Prey[i,:]
        #---------------------- Marine Memory saving ----------------
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey
        Inx = (fit_old < fitness)
        Indx = np.matlib.repmat(Inx,1,dim)
        Prey = np.multiply(Indx,Prey_old) + np.multiply(not Indx ,Prey)
        fitness = np.multiply(Inx,fit_old) + np.multiply(not Inx ,fitness)
        fit_old = fitness
        Prey_old = Prey
        #---------- Eddy formation and FADs  effect (Eq 16) -----------
        if np.random.rand() < FADs:
            U = np.random.rand(SearchAgents_no,dim) < FADs
            Prey = Prey + CF * (np.multiply((Xmin + np.multiply(np.random.rand(SearchAgents_no,dim),(Xmax - Xmin))),U))
        else:
            r = np.random.rand()
            Rs = Prey.shape[1-1]
            stepsize = (FADs * (1 - r) + r) * (Prey[np.random.permutation(Rs),:] - Prey[np.random.permutation(Rs),:])
            Prey = Prey + stepsize
        Iter = Iter + 1
        Convergence_curve[Iter] = Top_predator_fit
    ct = time.time()-ct

    
    return Top_predator_fit,Top_predator_pos,Convergence_curve,ct