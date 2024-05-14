import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import argparse


def cost_fin_mem(decisions, optimal, mem_length):
    # Modified loss function with Finite Memory
    cost = 0
    for i in range(min(mem_length, len(decisions))):
        cost += pow(pow(0.5, i) * np.linalg.norm(decisions[i] - optimal), 2)
    return np.sqrt(max(0, cost))


def cost(decisions, optimal, rho: float):
    # Our constructed loss function for Unbounded Memory
    cost = 0
    # Computing norm between our history and a history that picks the optimal
    # decision in each round
    for i in range(len(decisions)):
        cost += pow(pow(rho, i) * np.linalg.norm(decisions[i] - optimal), 2)
    return np.sqrt(max(0, cost))


def second_order_bco(d: int, x, A, eta, optimal, rho, decisions, is_finite, mem_length = 0):
    # This method uses the Bandit Newton Step algorithm to select decisions
    A_square_root = (scipy.linalg.sqrtm(A))

    random_directions = np.random.normal(size=(d,2))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_directions = random_directions.T
    y = x + 1/2 * np.linalg.inv(A_square_root) @ (random_directions[0] + random_directions[1])
    decisions.insert(0, y)
    if is_finite:
        loss = cost_fin_mem(decisions, optimal, mem_length)
    else:
        loss = cost(decisions, optimal, rho)

    gradient_est = 2 * d * loss * A_square_root @ random_directions[0]
    hessian_est = 2 * d * d * loss * A_square_root * (random_directions[0] @ random_directions[1].T + random_directions[1] @ random_directions[0].T) * A_square_root
    A = (A + eta * hessian_est).real
    x = x - eta * np.linalg.inv(A) @ gradient_est
    x = np.clip(x.real, -1, 1)

    return x, A


def run_experiment_two(T: int, d: int, rho: float, mem_lengths: list[int], iterations):
    # Comparing our algorithm's performance with bounded memory versus unbounded memory
    costs = {}

    # BCO-UM with [rho]
    optimal = np.random.uniform(-1, 1, d)
    aggregate_losses = []
    for i in range(iterations):
        print(f"Iteration {i+1} for rho = {rho}")
        decisions = [np.zeros(d)]
        losses = []
        eta = 1 / (np.sqrt(pow(d, 3)) * np.sqrt(T))
        x = np.zeros(d)
        A = np.identity(d)

        for _ in range(T):

            coin = np.random.uniform(0, 1)

            if coin > 1/np.cbrt(T):
                decisions.insert(0, decisions[0])
            else:
                x, A = second_order_bco(d, x, A, eta, optimal, rho, decisions, False)

            losses.append(cost(decisions, optimal, rho))
        aggregate_losses.append(losses)

    cumulative_cost = []
    cumulative_loss = 0
    for i in range(len(aggregate_losses[0])):
        for losses in aggregate_losses:
            cumulative_loss += losses[i] / iterations
        cumulative_cost.append(cumulative_loss)
        costs[rho] = cumulative_cost

    # BCO-M with various memory lengths
    for mem_length in mem_lengths:
        aggregate_losses = []
        for i in range(iterations):
            print(f"Iteration {i+1} for memory length = {mem_length}")
            decisions = [np.zeros(d)]
            losses = []
            eta = 1 / (np.sqrt(pow(d, 3)) * np.sqrt(T))
            x = np.zeros(d)
            A = np.identity(d)

            for k in range(T):

                coin = np.random.uniform(0, 1)

                if coin > 1/np.cbrt(T):
                    decisions.insert(0, decisions[0])
                else:
                    x, A = second_order_bco(d, x, A, eta, optimal, rho, decisions, True, mem_length)

                losses.append(cost_fin_mem(decisions, optimal, mem_length))
            aggregate_losses.append(losses)

        # Computing the average regret of our algorithm for each memory length
        cumulative_cost = []
        cumulative_loss = 0
        for i in range(len(aggregate_losses[0])):
            for losses in aggregate_losses:
                cumulative_loss += losses[i] / iterations
            cumulative_cost.append(cumulative_loss)
            costs[mem_length] = cumulative_cost

    for mem_length, cost_values in costs.items():
        if type(mem_length) == float:
            plt.plot(
            cost_values,
            label = 'rho = ' + str(rho),
            linestyle = 'dashed'
        )
        else:
            plt.plot(
            cost_values,
            label = 'memory length = ' + str(mem_length),
            linestyle = 'dashed'
        )
        # Plot regret.

        plt.xlabel(f'Time')
        plt.ylabel(f'Regret')
        plt.title(f'BCO-UM with rho = 0.5 versus BCO-M')
        plt.legend()

    plt.savefig('plots/experiment_1.png', bbox_inches='tight')    
    plt.show()

if __name__ == '__main__':
    # Running both experiments
    parser = argparse.ArgumentParser()
    parser.add_argument('--d',
                        type = int,
                        default = 2,
                        help = 'dimension.')
    parser.add_argument('--rho',
                        type = float,
                        default = 0.5,
                        help = 'value for rho.')
    parser.add_argument('--T',
                        type = int,
                        default = 1000,
                        help = 'time horizon.')
    parser.add_argument('--iterations',
                        type = int,
                        default = 25,
                        help = 'number of tests.')
    parser.add_argument('--mem_length',
                        nargs='+',
                        type = int,
                        default = 4,
                        help = 'memory length.')
    args = parser.parse_args()

    run_experiment_two(args.T, args.d, args.rho, args.mem_length, args.iterations)