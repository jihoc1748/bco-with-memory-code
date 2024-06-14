import numpy as np
import random
import time
import scipy
import matplotlib.pyplot as plt


def word_stream(file_path):
# Used for parsing the dataset
    with open(file_path, 'r') as file:
        for line in file:
                yield line



def projection(A, tau):
    # This function takes a matrix A and projects it onto the space of matrices with
    # rank less than tau using the standard singular value decomposition method.
    u, s, v = np.linalg.svd(A)

    def soft_threshold(s, tau):
        return np.maximum(s - tau, 0)
    
    def find_lambda(s, tau):
        def f(lam):
            return np.sum(np.maximum(s - lam, 0)) - tau
    
        lower, upper = 0, np.max(s)
        while upper - lower > 1e-5:
            mid = (lower + upper) /2
            if f(mid) > 0:
                lower = mid
            else:
                upper = mid
        return (lower + upper) / 2
    
    lambda_val = find_lambda(s, tau)
    s_thresh = soft_threshold(s, lambda_val)
    S_thresh = np.zeros((u.shape[1], v.shape[0]))
    np.fill_diagonal(S_thresh, s_thresh)

    return (u @ S_thresh) @ v

def ogd(file, tau, T):
    A = np.full((943, 1682), 0)

    data = word_stream(file)
    t = 0
    cost = 0
    running_average = []
    for line in data:
        if t > T:
            break
        t += 1
        eta = 1 / (np.sqrt(t))
        user, item, rating, _ = line.split()
        user = int(user)
        item = int(item)
        rating = int(rating)
        cost += pow((A[user][item] - rating), 2)
        print("Round " + str(t))
        print(cost/t)
        print("")
        running_average.append(cost/t)
        A_prime = A - eta * 2 * (A[user][item] - rating)
        A = projection(A_prime, tau)
    return running_average

def fkm(file, tau, T):
    d = 943 * 1682
    decision = np.full((943, 1682), 0)

    eta = np.sqrt(np.sqrt(T))
    delta = 1 / (np.sqrt(np.sqrt(T)))
    data = word_stream(file)
    running_average = []
    t = 0
    cost = 0
    prev_decisions = []
    for line in data:
        if t > T:
            break
        t+=1
        user, item, rating, _ = line.split()
        user = int(user)
        item = int(item)
        rating = int(rating)

        random_directions = np.random.normal(size=(d,1))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        random_directions = random_directions.T       
        y = decision + random_directions[0].reshape((943, 1682))
        prev_decisions.insert(0, y)
        round_cost = 0
        for i in range(len(prev_decisions)):
            round_cost += pow(prev_decisions[i][user][item] - rating, 2) * pow(0.5, i)

        cost += (round_cost)
        print("Round " + str(t))
        print(cost/2/t)
        print("")
        running_average.append(cost/2/t)
        grad = d / delta * round_cost * random_directions[0]

        decision = projection(decision - (eta * grad).reshape((943, 1682)), tau)
    return running_average

def main():
    # Obtaining the average loss values after running either the ogd or fkm algorithm
    ogd_costs = ogd('ml-100k\\ml-100k\\u.data', 20, 10000)
    fkm_costs = fkm('ml-100k\\ml-100k\\u.data', 20, 10000)

    plt.plot(ogd_costs,
             label = "OGD iterates",
             linestyle = 'dashed')
    plt.plot(fkm_costs,
             label = "BCO iterates",
             linestyle = 'dashed')
    plt.xlabel(f'Time')
    plt.ylabel(f'Average Squared Loss')
    plt.title("MovieLens 100k")
    plt.legend()

    plt.savefig('plots/recommendation.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
