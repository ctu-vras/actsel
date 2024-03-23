import pandas as pd


def create_log_df(iterations, steps, actions, columns):
    # Generate the tuples for the MultiIndex
    index_tuples = []
    for iteration in iterations:
        for step in steps:
            # 'before' entropy types only have one row
            index_tuples.append((iteration, step, 'before', 'true'))
            # 'expected' entropy type has multiple rows, one for each action
            for action in actions:
                index_tuples.append((iteration, step, 'expected', action))
            for action in actions:
                index_tuples.append((iteration, step, 'ig', action))
            # 'after' entropy types only have one row
            index_tuples.append((iteration, step, 'after', 'true'))
    
    # Create the MultiIndex
    index = pd.MultiIndex.from_tuples(index_tuples, names=['iteration', 'step', 'entropy', 'action'])
    
    # Create the DataFrame
    log_df = pd.DataFrame(index=index, columns=columns)
    
    return log_df


def pretty(mode, opt_type, parts, sums, sorted, ordered, single_node_optim=False):
    if single_node_optim:
        string = f"{mode} | Single node optimization"
    else:
        string = f"{mode} | {opt_type}"

    print()
    print("-"*len(string))
    print(string)
    print("-"*len(string))
    print(f"Parts:\n{parts}")

    if single_node_optim:
        print(f"\nIG sums:\n--------\nNO SUMS FOR SINGLE NODE OPTIM")
    else:
        print(f"\nIG sums:\n--------\n{sums}")
    print(f"\nSorted IGs:\n-----------\n{sorted}")
    print(f"\nOrdered actions:\n----------------\n{ordered}")
    print("-"*len(string))


# EXAMPLE USAGE 
if __name__ == "__main__":

    node_names = ['Material', 'Category', 'Density', 'Elasticity', 'Volume']
    columns = ['mat-vision', 'mat-sound', 'cat-vision', 'density', 'elasticity']

    data = {
    'mat-vision': [0.580468, 0.414406, 0.463058, 0.413425, 0.464032],
    'mat-sound': [1.069927, 0.364844, 0.632229, 0.268226, 0.313797],
    'cat-vision': [0.423533, 1.557094, 0.431163, 0.637112, 0.333979],
    'density': [1.301956, 0.374922, 0.361072, 0.297424, 0.375235],
    'elasticity': [0.848486, 0.368449, 0.545605, 0.352743, 0.409459]
    }

    df = pd.DataFrame(data, index=node_names)

    # Create an initial log DataFrame with one set of iterations
    log_df = create_log_df([0], [0], ['before', 'expected', 'after'], node_names, columns)
    print(log_df)
    new_df = create_log_df([1], [0], ['before', 'expected', 'after'], node_names, columns)
    new_df.loc[(1, 0, 'before')] = df.values
    print(new_df)
    log_df = pd.concat([log_df, new_df])

    print(log_df)