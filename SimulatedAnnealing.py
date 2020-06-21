"""
ModelMap

A toolkit for Model Mapping experimentation.

Simulated Annealing support class

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


import random
import math
import numpy as np

def ObjectiveFunc(confs, models, constraints, log_domain=False):
    performance = 0

    # Resource counters
    res_usage = [0] * len(constraints)

    # Calculate objective function as the aggregated throughput/latency
    for i in range(len(models)):
        # Get a performance model from the list
        fun = models[i]

        if log_domain:
            eval_conf = np.log2(confs[i])
        else:
            eval_conf = confs[i]

        # Sample the model at the current coordinates
        performance += fun(eval_conf)

        # Accumulate resource usage for all parameters
        for p in range(len(confs[i])):
            res_usage[p] += confs[i][p]

    # Constraints violation penalty for overallocation of resources
    penalty = 0
    for p in range(len(res_usage)):
        if res_usage[p] > constraints[p]:
            penalty += 100000

    return performance - penalty

def SimulatedAnnealing(confs, domain, models, constraints, opt='max', initial_temperature=0.7, final_temperature=0.001,
                       num_steps=1000, num_substeps=100, log_domain=False):


    def neighbour(confs, domain):
        '''
        Get a configuration set and return a neighbour configuration set from the feasible neighbour set.
        '''
        import copy

        # Collect all indices of the current confs values
        confs_indices = []
        for c in confs:
            temp = []
            for p in range(len(c)):
                temp.append(list(domain[p].keys())[list(domain[p].values()).index(c[p])])
            confs_indices.append(temp)

        # Find all the immediate neighbours of the current state
        neighbour_confs = []
        for idx, ci in enumerate(confs_indices):
            for p in range(len(ci)):
                # Find valid neighbours
                if (domain[p].get(ci[p] + 1)):
                    temp_confs = copy.deepcopy(confs)
                    temp_confs[idx][p] = domain[p].get(ci[p] + 1)
                    neighbour_confs.append(temp_confs[:])
                if (domain[p].get(ci[p] - 1)):
                    temp_confs = copy.deepcopy(confs)
                    temp_confs[idx][p] = domain[p].get(ci[p] - 1)
                    neighbour_confs.append(temp_confs[:])

        return random.choice(neighbour_confs)

    curr_performance = ObjectiveFunc(confs, models, constraints, log_domain)

    T0 = -1.0 / math.log(initial_temperature)
    TF = -1.0 / math.log(final_temperature)

    current_temperature = T0

    DeltaE_avg = 0.0

    na = 1.0

    best_performance = 0.0
    bestc = confs

    for step in range(num_steps):
        for step2 in range(num_substeps):

            temp_confs = neighbour(confs, domain)

            new_performance = ObjectiveFunc(temp_confs, models, constraints, log_domain)

            DeltaE = abs(new_performance - curr_performance)

            if (step == 0 and step2 == 0): DeltaE_avg = DeltaE

            if (new_performance > curr_performance and opt == 'max') or \
                    (new_performance < curr_performance and opt == 'min'):
                accept = True
            else:
                p = math.exp(-DeltaE / (DeltaE_avg * current_temperature))
                if random.random() < p:
                    accept = True
                else:
                    accept = False

            if accept == True:
                curr_performance = new_performance
                confs = temp_confs

                na = na + 1.0
                DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na

                if (curr_performance > best_performance and opt == 'max') or \
                        (curr_performance < best_performance and opt == 'min'):
                    best_performance = curr_performance
                    bestc = confs

        print("Step " + str(step) + "/" + str(num_steps) + " --- Best:" + str(best_performance))

        current_temperature = T0 + ((TF-T0) * (step / num_steps))

    return bestc, best_performance
