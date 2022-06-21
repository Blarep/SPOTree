'''
Generic file to set up the decision problem (i.e., optimization problem) under consideration
Must have functions: 
  get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector)
  find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors

This particular file sets up a shortest path decision problem over a 4 x 4 grid network, where driver starts in
southwest corner and tries to find shortest path to northeast corner.
'''

from gurobipy import *
import numpy as np
import random

dim = 4 #(creates dim * dim grid, where dim = number of vertices)
Edge_list = [(i,i+1) for i in range(1, dim**2 + 1) if i % dim != 0]
Edge_list += [(i, i + dim) for i in range(1, dim**2 + 1) if i <= dim**2 - dim]
Edge_dict = {} #(assigns each edge to a unique integer from 0 to number-of-edges)
for index, edge in enumerate(Edge_list):
    Edge_dict[edge] = index
D = len(Edge_list) # D = number of decisions
print(Edge_list)
print(Edge_dict)
def get_num_decisions():
  return D

Edges = tuplelist(Edge_list)
# Find the optimal total cost for an observation in the context of shortes path
m_shortest_path = Model('shortest_path')
m_shortest_path.Params.OutputFlag = 0
flow = m_shortest_path.addVars(Edges, ub = 1, name = 'flow')
m_shortest_path.addConstrs((quicksum(flow[i,j] for i,j in Edges.select(i,'*')) - quicksum(flow[k, i] for k,i in Edges.select('*',
  i)) == 0 for i in range(2, dim**2)), name = 'inner_nodes')
m_shortest_path.addConstr((quicksum(flow[i,j] for i,j in Edges.select(1, '*')) == 1), name = 'start_node')
m_shortest_path.addConstr((quicksum(flow[i,j] for i,j in Edges.select('*', dim**2)) == 1), name = 'end_node')

def shortest_path(cost):

    # m_shortest_path.setObjective(quicksum(flow[i,j] * cost[Edge_dict[(i,j)]] for i,j in Edges), GRB.MINIMIZE)
    m_shortest_path.setObjective(LinExpr( [ (cost[Edge_dict[(i,j)]],flow[i,j] ) for i,j in Edges]), GRB.MINIMIZE)
    m_shortest_path.optimize()
    return {'weights': m_shortest_path.getAttr('x', flow), 'objective': m_shortest_path.objVal}

def find_opt_decision(cost):
    weights = np.zeros(cost.shape)
    objective = np.zeros(cost.shape[0])
    for i in range(cost.shape[0]):
        temp = shortest_path(cost[i,:])
        for edge in Edges:
            weights[i, Edge_dict[edge]] = temp['weights'][edge]
        objective[i] = temp['objective']
    return {'weights': weights, 'objective':objective}


def evFunc(cromosome):
    #WIP
    retun 0

def genInitPob(pobSize):
    # Generate random feasible paths of length 6
    initPob = []
    AdjacencyList = []
    for i in range(1, 25):
        neighbors = []
        for edge in Edge_list:
            if i == edge[0]:
                neighbors.append(edge)
        AdjacencyList.append(neighbors)
    print(AdjacencyList)
    for i in range(pobSize):
        cromosome = [0] * 24
        actualNode = 1    
        for j in range(6):
            if len(AdjacencyList[actualNode-1]) == 2:
                UpOrRight = random.randint(0,1)
                if UpOrRight == 0:
                    #Go up
                    selectedEdge = AdjacencyList[actualNode-1][1]
                else:
                    #Go right
                    selectedEdge = AdjacencyList[actualNode-1][0]
                #Mark edge and update node
                cromosome[Edge_dict[selectedEdge]] = 1
                actualNode = selectedEdge[1]
            else:
                #Only can go in one direction
                selectedEdge = AdjacencyList[actualNode-1][0]
                cromosome[Edge_dict[selectedEdge]] = 1
                actualNode = selectedEdge[1]

def doOperator(prob):
    num = random.random()
    if num < prob:
        return True
    return False

def mutation(cromosome, pMut):
    for geneIndex in range(len(cromosome)):
        if doOperator(pMut):
            if cromosome[geneIndex]:
                cromosome[geneIndex] = 0
            else:
                cromosome[geneIndex] = 1

def crossover(crom1, crom2, pCross):
    child1 = crom1
    child2 = crom2
    if doOperator(pCross):
        point = random.randint(1, len(crom1)-1)
        child1 = crom1[:punto] + crom2[punto:]
        child2 = crom2[:punto] + crom1[punto:]
    return [child1,child2]


def rouletteWheel(population):
    #WIP, roulette wheel for maximization, needs to be changed to minimization
    fitnesses = [evFunc(crom) for crom in population]
    totalFit = sum(fitnesses)
    selProbs = [evFunc(crom)/totalFit for crom in population]
    selected = np.random.choice(len(population), size = len(population), p = selProbs)
    fathers = []
    for selIndex in selected:
        fathers.append(copy.deepcopy(population[selIndex]))
    return fathers

def bestCromosome(population, bestCrom, maxFit):
    fitnesses = [evFunc(crom) for crom in population]
    for i in range(len(population)):
        if fitnesses[i] > maxFit;
        bestCrom = copy.deepcopy(population[i])
        maxFit = fitnesses[i]
    return bestCrom

def evAlg(seed, pMut, pCross, gens):
    random.seed(100)
    initPob = genInitPob(20)
    selected = copy.deepcopy(initPob)
    bestCrom = bestCromosome(initPob, initPob[0], evFunc(initPob[0]))
    for i in range(gens);
        oldGen = copy.deepcopy(selected)
        bestCrom = bestCromosome(oldGen, bestCromm evFunc(bestCromosome))
        selected = rouletteWheel(oldGen)
        random.shuffle(selected)
        children = []

        #Crossover
        for j in range(0, len(selected), 2):
            children = crossover(selected[j], selected[j+1], pCross)
        bestCrom = bestCromosome(children, bestCromm evFunc(bestCromosome))
        #Mutation
        for cromosome in children:
            mutation(cromosome, pMut)
        keepBest = random.randint(0, len(selected)-1)
        children[keepBest] = bestCrom
        selected = children


#Tests...
genInitPob(20)


