
# from __future__ import print_function, division
from network import Network
from experiments import ExperimentObject
import json, copy, os
import numpy as np
import pandas as pd
import warnings
from acsel_logging import pretty, create_log_df
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
import time

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

experiment_object_names = [
    "experiment_banana",
    "experiment_banana_ycb",
    "experiment_black_kinova_cube_heavy",
    "experiment_ceramic_mug",
    "experiment_cracker_box_ycb",
    "experiment_enamel_mug_ycb",
    "experiment_plastic_cup_ycb",
    "experiment_soft_brick_ycb",
    "experiment_soft_yellow_sponge",
    "experiment_stiff_yellow_sponge",
    "experiment_strawberry_fake",
    "experiment_strawberry_ycb",
    "experiment_white_kinova_cube_light",
    "experiment_white_tile",
    "experiment_wineglass",
    "experiment_wineglass_ycb",
    "experiment_wooden_box_ycb"
]

action_to_node = {
    "mat-vision": "Material",
    "mat-sound": "Material",
    "cat-vision": "Category",
    "density": "Density",
    "elasticity": "Elasticity"
}

BASE_PATH = 'your/path/to/experiment/data'
MAX_ITER = 5
VERBOSITY = False
NO_NEGATIVE_IG = True


# Load actions configuration
with open('configs/actions.json', 'r') as f:
    actions_cfg = json.load(f)

# Load nodes configuration
with open('configs/nodes.json', 'r') as f:
    nodes_cfg = json.load(f)

# Now, actions_cfg and nodes_cfg can be passed to the Network constructor
template_network = Network(actions_cfg, nodes_cfg)

experiment_objects = [ExperimentObject(exp_name, BASE_PATH) for exp_name in experiment_object_names]

mode_options = ["Category", "Material", "Density", "Elasticity", "Volume", "Network-continuous", "Network-categorical", "Random"]


cumulative_iters = 0

mother_folder = f"data/logs/{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
T_START = time.time()

for MODE in mode_options:

    daughter_folder = mother_folder + f"/{MODE}"

    # only whole network has optimization options
    # otherwise its defaulted by the node type
    if MODE == "Network-continuous":
        MODE = "Network"
        OPTIM_TYPE = "continuous"
    elif MODE == "Network-categorical":
        MODE = "Network"
        OPTIM_TYPE = "categorical"
    elif MODE == "Random":
        OPTIM_TYPE = "Random"
    else:
        OPTIM_TYPE = "node-implicit"

    for exp_num, exp in enumerate(experiment_objects):
        experiment_timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        exp_log_path = f"{daughter_folder}/exp_{exp.object_name}_{experiment_timestamp}"
        os.makedirs(exp_log_path, exist_ok=True)

        experiment_log_df = create_log_df([0], [0], actions=template_network.action_names, columns=template_network.node_names)

        experiment_info = {
            "object_name": exp.object_name,
            "timestamp": experiment_timestamp,
            "MODE": MODE,
            "OPTIM_TYPE": OPTIM_TYPE,
            "MAX_ITER": MAX_ITER,
            "iters":[],
            "NO_NEGATIVE_IG": NO_NEGATIVE_IG
        }
        for iter_n in range(MAX_ITER):
            # reload measurements that were popped in previous iteration
            exp.reload_measurements()
            network = copy.deepcopy(template_network)
            available_actions = copy.copy(network.action_names)
            print(f"\n{'-'*15}Summary{'-'*15}")
            print(f"| Exp. object: {exp.object_name}")
            print(f"| Optimization mode: {MODE}.")
            print(f"| Optimization type: {OPTIM_TYPE}.")
            print(f"| Iteration num: {iter_n}/{MAX_ITER}.")
            t_delta = time.time() - T_START
            if cumulative_iters > 0:
                one_iter_estim = t_delta / cumulative_iters
                total_iter = MAX_ITER * len(experiment_objects) * 7
                remaining_iters = total_iter - cumulative_iters
                remaining_time = one_iter_estim * remaining_iters
                print(f"| ETA: {remaining_time:.2f} [s]")
            else:
                print(f"| ETA: N/A")
                
            print(f"| Elapsed time: {t_delta:.2f} [s]")
            print("-"*(30+len("Summary")))
            print()

            network_path = exp_log_path + f"/network/iter_{iter_n}"
            os.makedirs(network_path, exist_ok=True)
            iter_info = {}
            taken_actions = []
            taken_measurements = []
            step_n = 0
            while len(available_actions) > 0:

                net_before = network.serialize()
                with open(network_path + f"/before_{step_n}.json", 'w') as f:
                    json.dump(net_before, f, indent=4)

                before_entropies = network.get_true_entropies()
                expected_entropies = network.get_expected_entropies()
                IGs = network.expected_IGs_per_nodes(true_entropies=before_entropies,
                                                        expected_entropies=expected_entropies)
    
                chosen_action = None
                if MODE == "Network":  # whole network optimization
                    if OPTIM_TYPE == "continuous":
                        continuous_sum = IGs.iloc[:,2:].sum(axis=1)
                        sorted_IGs = continuous_sum.sort_values(ascending=False)
                        ordered_actions = sorted_IGs.index.tolist()
                        ordered_igs = sorted_IGs.values
                        if VERBOSITY:
                            pretty(MODE, OPTIM_TYPE, IGs.iloc[2:,:], continuous_sum, sorted_IGs, ordered_actions, False)

                    elif OPTIM_TYPE == "categorical":
                        categorical_sum = IGs.iloc[:,:2].sum(axis=1)
                        
                        sorted_IGs = categorical_sum.sort_values(ascending=False)
                        ordered_actions = sorted_IGs.index.tolist()
                        ordered_igs = sorted_IGs.values
                        if VERBOSITY:
                            pretty(MODE, OPTIM_TYPE, IGs.iloc[:2,:], categorical_sum, sorted_IGs, ordered_actions, False)

                    else:
                        raise ValueError(f"Unrecognized OPTIM_TYPE {OPTIM_TYPE}.")

                elif MODE in network.node_names:  # single node optimization

                    # This does not need split into continuous and categorical optim type
                    # because the optim type is automatically based on the node choice
                    # i.e. when category node is chosen, its IGs are categorical by definition
                    # or when density node is chosen, its IGs are continuous by definition
                    row = IGs.loc[:,MODE]
                    sorted_IGs = row.sort_values(ascending=False)
                    ordered_actions = sorted_IGs.index.tolist()
                    ordered_igs = sorted_IGs.values
                    if VERBOSITY:
                        pretty(MODE, OPTIM_TYPE, row, None, sorted_IGs, ordered_actions, True)

                elif MODE == "Random":
                    ordered_actions = [np.random.choice(available_actions)]
                    print("ORDERED ACTIONS: ", ordered_actions)

                else:
                    raise ValueError(f"Unknown MODE: '{MODE}'.")
                
                # go in order of the best actions
                for i, a in enumerate(ordered_actions):
                    # if best action was not yet taken
                    if a in available_actions:
                        available_actions.remove(a)
                        chosen_action = a
                        if NO_NEGATIVE_IG:
                            if MODE != "Random":
                                # if the best available action has negative IG,
                                if ordered_igs[i] <= 0.0:
                                    print("Warning! Best action has negative expected information gain.")
                                    chosen_action = None
                                    break

                        taken_actions.append(chosen_action)
                        break
                    # if taken, go to next best
                    else:
                        continue
                
                if chosen_action is None:
                    print("    > No more actions provided positive IG, ending step.")
                    print(f"    > Ordered actions: {ordered_actions}")
                    print(f"    > Ordered IGs: {ordered_igs}")
                else:
                    print(f"    > Chosen action: {chosen_action}")

                    # take the best action here
                    taken_measurement = network.execute_action(chosen_action, experiment_pntr=exp)
                    taken_measurements.append(taken_measurement.tolist())
                    
                    entropies_after_update = network.get_true_entropies()

                # log data
                if iter_n == 0 and step_n == 0:  # the beginning of a new experiment
                    experiment_log_df.loc[(0, 0, 'before')] = before_entropies.values
                    experiment_log_df.loc[(0, 0, 'expected')] = expected_entropies.values
                    experiment_log_df.loc[(0, 0, 'ig')] = IGs.values
                    if chosen_action is not None:
                        experiment_log_df.loc[(0, 0, 'after')] = entropies_after_update.values
                    
                else:
                    new_df = create_log_df([iter_n], [step_n], actions=network.action_names, columns=network.node_names)
                    new_df.loc[(iter_n, step_n,'before')] = before_entropies.values
                    new_df.loc[(iter_n, step_n,'expected')] = expected_entropies.values
                    new_df.loc[(iter_n, step_n,'ig')] = IGs.values
                    if chosen_action is not None:
                        new_df.loc[(iter_n, step_n,'after')] = entropies_after_update.values

                    # append to the new log
                    experiment_log_df = pd.concat([experiment_log_df, new_df])

                # log to file
                experiment_log_df.to_csv(exp_log_path + f'/{exp.object_name}.csv')

                if chosen_action is None:
                    break

                net_after = network.serialize()
                with open(network_path + f"/after_{step_n}.json", 'w') as f:
                    json.dump(net_after, f, indent=4)
                    
        
                step_n += 1
            iter_info["taken_measurements"] = taken_measurements
            iter_info["taken_actions"] = taken_actions
            experiment_info["iters"].append(iter_info)
            cumulative_iters += 1
    
        with open(exp_log_path + f"/{exp.object_name}_experiment_info.json", 'w') as f:
            json.dump(experiment_info, f, indent=4)
                # else take the next best action


