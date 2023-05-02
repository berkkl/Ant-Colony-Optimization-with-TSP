import random
import numpy as np
import matplotlib.pyplot as plt


# Constructor for the AntColonyOptimization class
# Initializes the class with the given parameters:
# num_ants: Number of ants in the colony
# num_iterations: Number of iterations for the optimization process
# alpha: Pheromone influence coefficient
# beta: Distance influence coefficient
# rho: Pheromone evaporation rate
# q: Constant used to calculate the amount of pheromone deposited
class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, q):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q


    # Main optimization method
    # Takes the distance matrix as input and returns the best route and its total distance
    # The method iteratively constructs routes for each ant, updates pheromone levels,
    # and keeps track of the best route found so far
    def optimize(self, distances):
        num_cities = len(distances)
        pheromones = np.ones((num_cities, num_cities))
        best_distance = float('inf')
        best_route = []

        for _ in range(self.num_iterations):
            all_routes = []
            for i in range(self.num_ants):
                route = self.construct_route(pheromones, distances)
                all_routes.append(route)

            pheromones = self.update_pheromones(pheromones, all_routes, distances)

            min_distance, min_route = self.get_best_route(all_routes, distances)
            if min_distance < best_distance:
                best_distance = min_distance
                best_route = min_route

        return best_distance, best_route

    # Constructs a route for a single ant using the pheromone and distance information
    # The next city is chosen based on a probabilistic transition rule, which considers both
    # pheromone levels and distances
    # Returns the generated route as a list of city indices
    def construct_route(self, pheromones, distances):
        num_cities = len(distances)
        start_city = random.randint(0, num_cities - 1)
        route = [start_city]
        available_cities = set(range(num_cities)) - {start_city}

        for _ in range(num_cities - 1):
            current_city = route[-1]
            probabilities = []

            for next_city in available_cities:
                tau = pheromones[current_city][next_city] ** self.alpha
                eta = (1 / distances[current_city][next_city]) ** self.beta
                probabilities.append(tau * eta)

            probabilities = [p / sum(probabilities) for p in probabilities]
            chosen_city = random.choices(list(available_cities), weights=probabilities)[0]
            available_cities.remove(chosen_city)
            route.append(chosen_city)

        return route

    # Updates the pheromone levels on the edges based on the routes taken by all ants
    # Evaporates the pheromones by a factor of rho and deposits new pheromones based on
    # the quality of the routes (shorter routes deposit more pheromones)
    # Returns the updated pheromone matrix
    def update_pheromones(self, pheromones, all_routes, distances):
        num_cities = len(distances)
        updated_pheromones = (1 - self.rho) * pheromones

        for route in all_routes:
            total_distance = self.calculate_distance(route, distances)
            delta_pheromones = self.q / total_distance

            for i in range(num_cities):
                a = route[i]
                b = route[(i + 1) % num_cities]
                updated_pheromones[a][b] += delta_pheromones
                updated_pheromones[b][a] += delta_pheromones

        return updated_pheromones

    # Calculates the total distance of a given route using the distance matrix
    # The distance is calculated as the sum of the distances between consecutive cities
    # in the route
    # Returns the total distance as a float
    def calculate_distance(self, route, distances):
        num_cities = len(distances)
        distance = 0

        for i in range(num_cities):
            distance += distances[route[i]][route[(i + 1) % num_cities]]

        return distance


    # Finds the best (shortest) route among all routes provided
    # Compares the total distance of each route and selects the one with the minimum distance
    # Returns the shortest distance and the corresponding route
    def get_best_route(self, all_routes, distances):
        best_distance = float('inf')
        best_route = None

        for route in all_routes:
            distance = self.calculate_distance(route, distances)
            if distance < best_distance:
                best_distance = distance
                best_route = route

        return best_distance, best_route

# Creating random distances for each city for testing purposes
def create_distances_matrix(num_cities):
    distances = np.zeros((num_cities, num_cities))
    city_coords = []

    for _ in range(num_cities):
        x, y = random.uniform(0, 100), random.uniform(0, 100)
        city_coords.append((x, y))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.sqrt((city_coords[i][0] - city_coords[j][0])**2 + (city_coords[i][1] - city_coords[j][1])**2)
            distances[i][j] = distance
            distances[j][i] = distance
            print(f"Distance between city {i} and city {j}: {distance}")

    return distances, city_coords

def plot_route(city_coords, best_route, distances):
    route_coords = [city_coords[city] for city in best_route]
    route_coords.append(city_coords[best_route[0]])

    xs, ys = zip(*route_coords)

    plt.scatter(xs, ys)
    plt.plot(xs, ys)

    # Increase the font size of city names
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, f"{i}", fontsize=12, color="red")

    # Add distance labels to the edges
    for i in range(len(best_route)):
        city1 = best_route[i]
        city2 = best_route[(i + 1) % len(best_route)]
        edge_midpoint = ((city_coords[city1][0] + city_coords[city2][0]) / 2,
                         (city_coords[city1][1] + city_coords[city2][1]) / 2)
        edge_distance = np.round(distances[city1][city2], 1)
        plt.text(edge_midpoint[0], edge_midpoint[1], f"{edge_distance}", fontsize=10, color="blue")

    plt.show()


def main(num_cities):
    distances, city_coords = create_distances_matrix(num_cities)

    num_ants = 20
    num_iterations = 100
    alpha = 1
    beta = 5
    rho = 0.5
    q = 100

    aco = AntColonyOptimization(num_ants, num_iterations, alpha, beta, rho, q)
    best_distance, best_route = aco.optimize(distances)

    print("Best distance:", best_distance)
    print("Best route:", best_route)

    plot_route(city_coords, best_route, distances)

if __name__ == "__main__":
    num_cities = 10
    main(num_cities)
