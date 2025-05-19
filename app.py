import random
import numpy as np

def read_input(file_location):
    with open(file_location, 'r') as file:
        num_cities = int(file.readline().strip())
        cities = [tuple(map(int, file.readline().strip().split())) 
        for _ in range(num_cities)]
    return num_cities, cities


def nnh(distance_matrix):
    num_cities = len(distance_matrix)


    unvisited = set(range(num_cities))
    tour = [random.choice(list(unvisited))]

    unvisited.remove(tour[0])

    while unvisited:

        last_city = tour[-1]
        next_city = min(unvisited, key=lambda city: distance_matrix[last_city, city])
        tour.append(next_city)
        unvisited.remove(next_city)

    return tour + [tour[0]]



def initialize_pop(num_cities, population_size, distance_matrix):

    population = []

    nnh_percentage = 0.6

    nnh_count = int(nnh_percentage * population_size)
    random_count = population_size - nnh_count

    for _ in range(nnh_count):
        population.append(nnh(distance_matrix))

    for _ in range(random_count):
        population.append(np.append(np.random.permutation(num_cities), 0).tolist())

    return population


def euclidean_dist(cities):

    cities = np.array(cities)
    return np.linalg.norm(cities[:, None, :] - cities[None, :, :], axis=-1)


def cal_path_dist(route, distance_matrix):

    return distance_matrix[route[:-1], route[1:]].sum()



def evaluate_pop(population, distance_matrix):
    fit_scores = 1 / np.array([cal_path_dist(route, distance_matrix) for route in population])


    min_fit, max_fitness = fit_scores.min(), fit_scores.max()

    normalized_fitness = np.ones_like(fit_scores) if max_fitness == min_fit else (fit_scores - min_fit) / (max_fitness - min_fit)
    
    return normalized_fitness.tolist(), fit_scores.tolist()



def roul_wheel_sel(population, fit_scores, num_parents):
    fit_scores = np.array(fit_scores)
    cumulative_probab = np.cumsum(fit_scores) / np.sum(fit_scores)

    indices = np.searchsorted(cumulative_probab, np.random.rand(num_parents))

    return [population[i] for i in indices]



def adaptive_mutation(route, mutation_rate):

    if np.random.rand() < mutation_rate:
        i, j = np.sort(np.random.choice(range(1, len(route) - 1), 2, replace=False))
        route[i:j] = route[i:j][::-1]
    return route


def two_pt_cross(parent1, parent2):
    size = len(parent1) - 1  
    
    point1, point2 = sorted(np.random.choice(range(1, size), 2, replace=False))

    child = np.full(size, -1)
    child[point1:point2] = parent1[point1:point2]

    parent2_genes = [gene for gene in parent2 if gene not in child]
    missing_indices = np.where(child == -1)[0]

    child[missing_indices] = parent2_genes[:len(missing_indices)]

    return np.append(child, child[0]).tolist()  


def create_nxt_gen(population, fit_scores, num_cities, mutation_rate, distance_matrix):

    elite_fract = {50: 0.10, 100: 0.07, 200: 0.05, 500: 0.025}.get(num_cities, 0.02)

    elite_count = int(elite_fract * len(population))

    elites = [population[i] for i in np.argsort(fit_scores)[-elite_count:]]


    num_parents = len(population) - elite_count
    parents = roul_wheel_sel(population, fit_scores, num_parents)


    offspring = []
    for _ in range(num_parents):
        retries = 5
        while retries > 0:

            try:
                child = two_pt_cross(*random.sample(parents, 2))  


                child = adaptive_mutation(child, mutation_rate)  
                offspring.append(child)
                break
            except ValueError:
                retries -= 1

        if retries == 0:
            offspring.append(random.choice(elites)) 

    return elites + offspring





def write_output(best_route, best_distance, cities, output_path="output.txt"):
    with open(output_path, 'w') as file:
        file.write(f"{best_distance:.3f}\n")  
        for city_index in best_route:
            file.write(f"{cities[city_index][0]} {cities[city_index][1]} {cities[city_index][2]}\n")






def main():
    file_location = "input.txt"
    num_cities, cities = read_input(file_location)

    distance_matrix = euclidean_dist(cities)

    pop_config = [
        (40, 100, 1000),
        (90, 300, 870),
        (190, 400, 450),
        (400, 600, 280),
        (float('inf'), 200, 10) 
    ]
    
    for threshold, pop_size, gens in pop_config:

        if num_cities < threshold:
            population_size, generations = pop_size, gens
            break

    mutation_rates = [
        (40, 0.3, 0.02),
        (90, 0.4, 0.03),
        (190, 0.5, 0.04),
        (400, 0.7, 0.06),
        (float('inf'), 0.7, 0.06) 
    ]


    for threshold, start_mut, end_mut in mutation_rates:
        if num_cities < threshold:
            initial_mut_rate, final_mutation_rate = start_mut, end_mut
            break


    elite_fract = 0.05  
    no_improvement_limit = 50


    population = initialize_pop(num_cities, population_size, distance_matrix)


    best_distance_overall = float('inf')
    no_improve_counter = 0

    for generation in range(generations):
        current_mutation_rate = initial_mut_rate - (
            (generation / generations) * (initial_mut_rate - final_mutation_rate)

        )

        normalized_fitness, raw_fitness = evaluate_pop(population, distance_matrix)


        population = create_nxt_gen(population, raw_fitness, num_cities, current_mutation_rate, distance_matrix)


        best_index = np.argmax(raw_fitness)

        best_route = population[best_index]
        best_distance = cal_path_dist(best_route, distance_matrix)

        if best_distance < best_distance_overall:
            best_distance_overall = best_distance
            no_improve_counter = 0
        else:
            no_improve_counter += 1


        

        if no_improve_counter >= no_improvement_limit:
            break

    write_output(best_route, best_distance_overall, cities)
    




if __name__ == "__main__":
    main()
