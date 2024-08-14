# Branch and Bound Algorithm

# -----problem------
# max 100x1+150x2
# 2x1+x2 <= 10
# 3x1+6x2 <= 40
# x1,x2>0 and integer
# ------------------

from gurobipy import *
import numpy as np
import copy
import matplotlib.pyplot as plt

RLP = Model("relaxed MIP")

x = {}
for i in range(2):
    x[i] = RLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='X_{str(i)}')

RLP.setObjective(100*x[0]+150*x[1], GRB.MAXIMIZE)

RLP.addConstr(2*x[0]+x[1] <= 10, name='c_1')
RLP.addConstr(3*x[0]+6*x[1] <= 40, name='c_2')

RLP.optimize()


class Node:
    def __init__(self):
        self.local_LB = 0
        self.local_UB = np.inf
        self.x_solution = {}
        self.x_int_solution = {}
        self.branch_var_list = []
        self.model = None
        self.cnt = None
        self.is_integer = False

    def deepcopy_node(node):
        new_node = Node()
        new_node.local_LB = 0
        new_node.local_UB = np.inf
        new_node.x_int_solution = copy.deepcopy(node.x_int_solution)
        new_node.x_solution = copy.deepcopy(node.x_solution)
        new_node.branch_var_list = []
        new_node.model = node.model.copy()
        new_node.cnt = node.cnt
        new_node.is_integer = node.is_integer

        return new_node
    
def branch_and_bound(model):
    model.optimize()
    global_UB = model.ObjVal
    global_LB = 0
    eps = 1e-3
    incument_node = None
    gap = np.inf

    # create initial node
    queue = []
    node = Node()
    node.local_LB = 0
    node.local_UB = global_UB   
    node.model = model.copy()
    node.model.setParam("OutputFlag",0)
    node.cnt = 0
    queue.append(node)

    cnt = 0
    global_UB_change = []
    global_LB_change = []

    while len(queue) > 0 and global_UB - global_LB > eps:
        # select the cyurrent node
        current_node = queue.pop()
        cnt += 1

        # solve the current model
        current_node.model.optimize()
        solution_status = current_node.model.Status

        """
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUND = 5
        """
        # is_integer: mark whether the current solution is integer solution
        # is_pruned: mark whether the current solution is pruned
        is_integer = True
        is_pruned = False
        if solution_status == 2:
            for var in current_node.model.getVars():
                current_node.x_solution[var.varName] = var.x
                print(var.VarName,' = ',var.x)

                # round the current solution to get the integer
                current_node.x_int_solution[var.varName] = (int) (var.x)
                if abs(int(var.x) - var.x) >= eps:
                    is_integer = False
                    current_node.branch_var_list.append(var.VarName)
            
            # update LB and UB
            if is_integer:
                # for each integer variable, update LB and UB
                current_node.is_integer = True
                current_node.local_UB = current_node.model.ObjVal
                current_node.local_LB = current_node.model.ObjVal
                if current_node.local_LB > global_LB:
                    global_LB = current_node.local_LB
                    incurrent_node = node.deepcopy_node(current_node)
            if not is_integer:
                current_node.is_integer = False
                current_node.local_UB = current_node.model.ObjVal
                current_node.local_LB = 0
                for var_name in current_node.x_int_solution.keys():
                    var = current_node.model.getVarByName(var_name)
                    current_node.local_LB += current_node.x_int_solution[var_name]*var.Obj
                if current_node.local_LB > global_LB or (current_node.local_LB == global_LB and current_node.is_integer):
                    global_LB = current_node.local_LB
                    incumbent_node = node.deepcopy_node(current_node)
                    incumbent_node.local_LB = current_node.local_LB
                    incumbent_node.local_UB = current_node.local_UB

            """
            prune step
            """
            # prune by optimility
            if is_integer:
                is_pruned = True
            
            # prune by bound
            if (not is_integer and current_node.local_LB < global_LB):
                is_pruned = True

            gap = round(100*(global_UB-global_LB)/global_LB, 2)
            print('\n -------- \n', cnt, '\t Gap = ',gap,' %s')
        elif solution_status != 2:
            # the current node is infeasible or unbound
            is_integer = False
            # prune step
            # prune by infeasibility
            is_pruned = True
        
        '''
        Branch step
        '''
        if is_pruned:
            # select the branch variables
            branch_var_name = current_node.branch_var_list[0]
            left_var_bound = (int) (current_node.x_sol[branch_var_name])
            right_var_bound = (int) (current_node.x_sol[branch_var_name]) + 1
        
            # create two child nodes
            left_node = node.deepcopy_node(current_node)
            right_node = node.deepcopy_node(current_node)

            # create left child node
            temp_var = left_node.model.getVarByName(branch_var_name)
            left_node.model.addConstr(temp_var <= left_var_bound, name='branch_left_'+str(cnt))
            left_node.model.setParam('OutputFlag', 0)
            left_node.model.update()
            cnt += 1
            left_node.cnt = cnt

            # create right child node
            temp_var = right_node.model.getVarByName(branch_var_name)
            right_node.model.addConstr(temp_var >= right_var_bound, name='branch_right_'+str(cnt))
            right_node.model.setParam('OutputFlag', 0)
            right_node.model.update()
            cnt += 1
            right_node.cnt = cnt

            queue.append(left_node)
            queue.append(right_node)

            # update the global UB, explore all the leaf nodes
            temp_global_UB = 0
            for node in queue:
                node.model.optimize()
                if node.model.status == 2:
                    if node.model.ObjVal >= temp_global_UB:
                        temp_global_UB = node.model.ObjVal

            global_UB = temp_global_UB
            global_UB_change.append(global_UB)
            global_LB_change.append(global_LB)

    # all the nodes are explores, uodate the LB and UB
    global_UB = global_LB
    gap = round(100*(global_UB-global_LB)/global_LB, 2)
    global_UB_change.append(global_UB)
    global_LB_change.append(global_LB)

    print('----------------------------------------')
    print('        Branch and Bound terminates     ')
    print('        Optimal solution found          ')
    print('----------------------------------------')
    print('\n final gap = ', gap, ' %')
    print('Optimal solution: ', incumbent_node.x_int_solution)
    print('Optimal Objective: ',    global_LB)

    return incumbent_node, gap, global_UB_change, global_LB_change


if __name__ == "__main__":
    incumbent_node, gap, global_UB_change, global_LB_change = branch_and_bound(RLP)
    # plot the results
    font_dict = {"family":'Arial',
                 "style":'oblique',
                 "weight":"normal",
                 "color":"green",
                 "size":20}

    plt.rcParams['figure.figsize'] = (12.0,8.0)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16

    x_cor = range(1,len(global_LB_change) + 1)
    plt.plot(x_cor, global_LB_change, label='LB')
    plt.plot(x_cor, global_UB_change, label='UB')
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dict)
    plt.ylabel('Bound update', fontdict=font_dict)
    plt.title('Bounds update during branch and bound procedure \n',fontsize=23)
    plt.show()
