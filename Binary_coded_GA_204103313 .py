
# General Binary Coded Genetic algorithm for n numbers of variables
# By: Manpreet Singh
#     Roll No. 204103313

import numpy as np
import matplotlib.pyplot as plt


# Input Parameters
print("\n\n*********** Binary Coded genetic Algorithm ************* ")
print("\n\t Provide the Input Parameters \n")

N = int(input("Enter The Size of Population (N): "))                         # Initial Population size
Cross_over_probability = float(input("Enter Cross Over Probability (Pc): "))   # Cross Over Probability
Mutation_probability = float(input("Enter Mutation Probability (Pc): "))          # Mutation Probability
Max_nos_generation = int(input("Enter Maximum limit on maximum no of generation: "))    # Limit for Maximum Number of Generation to run for

# ================================= Define Objective function and constrains ================================


def objective_function(x, for_fitness=False):   # x is array of variables i.e x = [x1, x2 ..xn]

    func = x[0] + x[1] + - 2*pow(x[0], 2) - pow(x[1], 2) + x[0]*x[1]     # user defined

    objective_maximize = False    # if Maximization Problem then set -> True ; for Minimization set -> False

    if objective_maximize is False and for_fitness is True:
        return 1/(func+1)    # Converting minimization to maximization problem
    return func


def variable_range(epsilon=False):
    x1 = [0, 0.5]         # user defined -> range of variable 1
    x2 = [0, 0.5]         # user defined -> range of variable 2
    # x3 = [0.0, 0.8]     # if there is third variable

    e = [0.0001, 0.0001]        # user defined -> epsilon value for  x1 and x2
    if epsilon is True:
        return e
    return [x1, x2]

# ===========================================================================================================
#                                   Binary Coded Genetic Algorithm
# -----------------------------------------------------------------------------------------------------------


# String length calculation
def string_length(sub_string=False):    # calculating string length and sub string lengths
    epsilon = variable_range(epsilon=True)
    v_range = variable_range()
    sub_str = [0]*len(epsilon)
    for v in range(len(v_range)):
        x = v_range[v]
        sub_str[v] += int(np.ceil((np.log2((x[1] - x[0])/epsilon[v]))))
    if sub_string is True:
        return sub_str
    return sum(sub_str)


# Random Population Generation
def generate_population(population_size):
    str_length = int(string_length())   # taking string length
    return np.random.randint(2, size=(population_size, str_length))   # generating random population of given size (N)


# Binary to Decimal converter
def binary_to_decimal(string):   # -> takes a binary sting and return equivalent decimal value
    decimal, i = 0, 0
    while string != 0:
        remainder = string % 10
        decimal = decimal + remainder * pow(2, i)
        string = string // 10
        i += 1
    return decimal


# Normalized decoder function
def get_real_decoded_val(x, sub_str_size):
    real_decoded = [0] * len(x)
    var_range = variable_range()
    for i in range(len(x)):
        x_range = var_range[i]
        real_decoded[i] = x_range[0] - ((x_range[0] - x_range[1]) / (pow(2, sub_str_size[i]) - 1)) * x[i]

    return real_decoded


# Fitness Evaluation function
def get_fitness(population, get_additional_data=False):
    population_fitness = [0]*len(population)
    x = [0] * len(variable_range())     # to store binary to decimal converted value of string

    real_decoded_value = [[0] * len(x)] * len(population)
    sub_str_size = string_length(sub_string=True)
    for index in range(len(population)):
        s = population[index]
        i, j = 0, 0
        for n in range(len(x)):
            j += sub_str_size[n]
            sub_string = s[i:j]
            i = j
            binary = int("".join(map(str, sub_string)))
            x[n] = binary_to_decimal(binary)
        real_decoded_value[index] = get_real_decoded_val(x, sub_str_size)   # -> normalized decoded value of each string
        population_fitness[index] = objective_function(real_decoded_value[index], for_fitness=True)  # -> getting fitness value

    real_decoded_value = np.asarray(real_decoded_value)
    min_max_avg_fitness = [np.min(population_fitness), np.max(population_fitness), np.average(population_fitness)]
    best_sol_of_gen = real_decoded_value[population_fitness.index(max(population_fitness))]

    if get_additional_data is True:   # -> to return additional value along with fitness value
        additional_data = [population_fitness,    # -> Population Fitness values
                           min_max_avg_fitness,   # -> Min, Max and average fitness value
                           best_sol_of_gen,       # -> Best solution of current generation
                           real_decoded_value]    # -> Real decoded values of variable

        return additional_data

    return population_fitness


# Roulette Wheel Selection function
def roulette_wheel(population, fitness_value, selected_string_with_list=False):
    modified_fitness = [0] * len(fitness_value)   # -> modified fitness value to avoid negative value in fitness
    for i in range(len(fitness_value)):
        modified_fitness[i] = fitness_value[i] + abs(min(fitness_value)-1)

    total_f = sum(modified_fitness)
    probability = [100/total_f * element for element in modified_fitness]  # -> getting probability as per fitness value
    c_probability = probability

    for i in range(1, len(probability)):
        c_probability[i] = c_probability[i] + c_probability[i-1]  # -> converting probability to cumulative probability

    selected_str = np.array([[0]*string_length()]*len(population))   # -> to store selected stings for mating pool
    selected_list = [0]*len(population)      # list to store the index of selected string

    # selection through roulette wheel
    for i in range(len(probability)):
        r = np.random.random() * 100        # r -> random number for selection
        for j in range(len(c_probability)):
            if r <= c_probability[j]:       # checking r on cumulative probability scale
                selected_str[i] = population[j]   # selecting wining string for mating pool
                selected_list[i] = (j+1)
                break
    if selected_string_with_list is True:
        return [selected_str,    # -> selected strings for mating pool
                selected_list]   # -> index list of selected string from given population
    return selected_str


# Two-Point Cross Over Operator
def cross_over(mating_pool, cross_over_probability):
    r = np.arange(int(len(mating_pool)))
    np.random.shuffle(r)
    nos_pairs = int(len(mating_pool)/2)
    child_population = mating_pool

    for n in range(nos_pairs):
        parent1 = mating_pool[r[n]]
        parent2 = mating_pool[r[n+nos_pairs]]

        if np.random.random() <= cross_over_probability:   # checking cross over probability
            site = np.random.randint(string_length(), size=2)
            site.sort()
            for bit in range(int(site[0]), int(site[1])):  # swapping cross over sites
                temp_bit = parent1[bit]
                parent1[bit] = parent2[bit]
                parent2[bit] = temp_bit
            child1, child2 = parent1, parent2     # new solution as child1 and child2
            child_population[r[n]] = child1
            child_population[r[n+nos_pairs]] = child2
    return child_population


# Mutation Operator for child solutions
def mutation(child_population, mutation_probability):
    new_generation = child_population
    for child_index in range(len(new_generation)):
        for bit in range(string_length()):
            if np.random.random() <= mutation_probability:
                if new_generation[child_index][bit] == 0:
                    new_generation[child_index][bit] = 1
                else:
                    new_generation[child_index][bit] = 0
    return new_generation


def natural_selection(parent_population, offspring_population,  fitness_value):
    total_population = np.concatenate((parent_population, offspring_population))
    survivals = np.array([[0]*string_length()]*len(parent_population))

    for i in range(len(parent_population)):  # -> selecting best as per fitness value
        survivals[i] = total_population[fitness_value.index(max(fitness_value))]
        fitness_value[fitness_value.index(max(fitness_value))] = 0

    return survivals


# -----------------------------------------------------------------------------------------------------------
min_fitness, max_fitness, avg_fitness = [], [], []
generations, solutions = [], []

use_natural_selection_operator = True    # to select different approach
live_fitness_plot = True   # for live plot of fitness vs generations

if live_fitness_plot is True:
    plt.figure(figsize=(10, 6))       # plot size


generation_count = 0
P = generate_population(N)        # -> generating initial population of size N


# *************************** Main While Loop ***************************

while generation_count < Max_nos_generation:   # Main While loop over generations

    Fitness = get_fitness(P, get_additional_data=True)
    Mating_pool = roulette_wheel(P, Fitness[0])

    off_spring = cross_over(Mating_pool, Cross_over_probability)
    off_spring_sol = mutation(off_spring, Mutation_probability)

    if use_natural_selection_operator is True:
        total_fitness = get_fitness(np.concatenate((P, off_spring_sol)))   # - > getting total fitness of parent + child
        next_generation = natural_selection(P, off_spring_sol, total_fitness)  # -> best survival as per fitness value

        P = next_generation
    else:
        P = off_spring_sol

    generation_count += 1
    print(str(generation_count) + " ->  Fitness (min, max, avg) " + str(Fitness[1]) + " -> best sol " + str(Fitness[2]))

    if generation_count < 6 or generation_count == N-1:
        solutions.append(Fitness[3])

    # -> **************************** Data for plot ******************************

    generations.append(generation_count)
    min_fitness.append(Fitness[1][0])
    max_fitness.append(Fitness[1][1])
    avg_fitness.append(Fitness[1][2])

    if live_fitness_plot is True:
        plt.style.use('seaborn-darkgrid')
        plt.cla()
        plt.plot(generations, max_fitness, linewidth=2, label='Max Fitness')
        plt.plot(generations, avg_fitness, linewidth=2, label='Avg Fitness')
        plt.plot(generations, min_fitness, linewidth=2, label='Min Fitness')

        plt.title('Fitness value v/s Generations',  fontsize=15)
        plt.xlabel('Generations',  fontsize=15)
        plt.ylabel('Fitness Value',  fontsize=15)
        plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0, prop={'size': 15})
        plt.tight_layout()
        plt.pause(0.001)

plt.show()

# -----------------------------------------------------------------------------------------------------------
#                                  Plotting post analysis data
# -----------------------------------------------------------------------------------------------------------
# Plot -> Fitness vs generations
if live_fitness_plot is False:
    plt.figure(figsize=(10, 6))  # plot size
    plt.style.use('seaborn-darkgrid')

    plt.plot(generations, max_fitness, linewidth=2, label='Max Fitness')
    plt.plot(generations, avg_fitness, linewidth=2, label='Avg Fitness')
    plt.plot(generations, min_fitness, linewidth=2, label='Min Fitness')

    plt.title('Fitness value v/s Generations',  fontsize=15)
    plt.xlabel('Generations',  fontsize=15)
    plt.ylabel('Fitness Value',  fontsize=15)
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0, prop={'size': 15})
    plt.tight_layout()

# Plot -> Solution vs Generations
if len(variable_range()) == 2:
    plt.style.use('default')
    plt.figure(figsize=(12, 6))

    lim = variable_range()
    x1 = np.linspace(lim[0][0], lim[0][1], 22)
    x2 = np.linspace(lim[1][0], lim[1][1], 22)

    p, q = np.meshgrid(x1, x2)
    z = p + q + - 2 * p ** 2 - q ** 2 + p * q       # objective function

    ax1 = plt.subplot(2, 3, 1)
    ax1.contour(p, q, z, 20)
    ax1.scatter(solutions[0][:, 0], solutions[0][:, 1], c='r', edgecolors='k', label=' 1st generation')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('variable x2', fontsize=12)
    plt.title('1st generation')

    ax2 = plt.subplot(232, sharey=ax1)
    plt.contour(p, q, z, 20)
    plt.scatter(solutions[1][:, 0], solutions[1][:, 1], c='r', edgecolors='k',  label=' 2nd generation')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.title('2nd generation')

    ax3 = plt.subplot(233, sharey=ax1)
    plt.contour(p, q, z, 20)
    plt.scatter(solutions[2][:, 0], solutions[2][:, 1], c='r', edgecolors='k', label=' 3rd generation')
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.title('3rd generation')
    plt.colorbar()

    ax4 = plt.subplot(234, sharey=ax1)
    plt.contour(p, q, z, 20)
    plt.scatter(solutions[3][:, 0], solutions[3][:, 1], c='r', edgecolors='k', label=' 4th generation')
    plt.title('4th generation')
    plt.xlabel('variable x1', fontsize=12)
    plt.ylabel('variable x2', fontsize=12)

    ax5 = plt.subplot(235, sharey=ax1)
    plt.contour(p, q, z, 20)
    plt.scatter(solutions[4][:, 0], solutions[4][:, 1], c='r', edgecolors='k', label=' 5th generation')
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.xlabel('variable x1', fontsize=12)
    plt.title('5th generation')

    ax6 = plt.subplot(236, sharey=ax1)
    plt.contour(p, q, z, 20)
    plt.scatter(solutions[5][:, 0], solutions[5][:, 1], c='r', edgecolors='k', label=' Nth generation')
    plt.setp(ax6.get_yticklabels(), visible=False)
    plt.title('200th generation')
    plt.xlabel('variable x1', fontsize=12)
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# ------------------------------------- End of Program --------------------------------------
