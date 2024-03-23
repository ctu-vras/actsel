
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import norm

np.set_printoptions(precision=2, suppress=True)


def sampled_cfm_gmm(ax, mus, sigmas, n_samples=1000, plot=False):

    rng = ax[-1] - ax[0]
    bin_size = rng / len(ax)

    gaussians = []
    sample_sets = []
    for i in range(len(mus)):
        mu = mus[i]
        sigma = sigmas[i]
        pdf = norm.pdf(ax, mu, sigma)
        sample_sets.append(norm.rvs(loc=mu, scale=sigma, size=n_samples))
        gaussians.append(pdf)
        if plot:
            plt.plot(ax, pdf, lw=2)

    colors = ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']
    cfm = np.zeros([len(mus), len(mus)])
    
    for r, samples in enumerate(sample_sets):
        for sample in samples:
                # correct for the binning
                x = int(sample / bin_size)
                ys = [pdf[x] for pdf in gaussians]
                y_hat_id = np.argmax(ys)
                if plot:
                    plt.scatter(sample, ys[y_hat_id], color = "#" + colors[y_hat_id], s=12)
                cfm[r, y_hat_id] += 1
        if plot:
            plt.title("Source Gaussian {}".format(r+1))
            plt.xlabel("$x$")
            plt.ylabel("$f(x)$")
            # plt.savefig(saving_path + "/source_gaussian_{}.pdf".format(r), dpi=300)
            plt.show()

            for pdf in gaussians:
                plt.plot(ax, pdf, lw=2)

    cfm = cfm / np.sum(cfm, axis=0).reshape(1,-1)  # columns sum to one
    if plot:
        plt.show()
    return cfm


with open('configs/templates/confusion_matrices.json', 'r') as file:
    cfm_data = json.load(file)

with open('configs/templates/action_templates.json', 'r') as file:
    action_templates = json.load(file)

with open('configs/templates/transition_templates.json', 'r') as file:
    transition_templates = json.load(file)

# first check whether the nodes mus and elasticities have been changed
with open('configs/nodes.json', 'r') as file:
    nodes = json.load(file)

changed_at_least_once = False
for node_name, node in nodes.items():
    if node["type"] == 'categorical' or node_name == "Volume":
        continue

    changed = False
    if node["params"]["means"] != cfm_data[f"{node_name.lower()}-specs"]["mu"]:
        cfm_data[f"{node_name.lower()}-specs"]["mu"] = node["params"]["means"]
        changed = True
    if node["params"]["stds"] != cfm_data[f"{node_name.lower()}-specs"]["sigma"]:
        cfm_data[f"{node_name.lower()}-specs"]["sigma"] = node["params"]["stds"]
        changed = True

    if changed:
        changed_at_least_once = True
        print(f"Recomputing the cfm for {node_name}, because either means or stds have changed.")
        # now recompute the cfm
        coverage_factor=5.0
        step_factor=10.0

        largest_mean = max(node["params"]['means'])
        std_of_largest_mean = node["params"]['stds'][node["params"]['means'].index(largest_mean)]
        axis_max = largest_mean + coverage_factor * std_of_largest_mean

        lowest_mean = min(node["params"]['means'])
        std_of_lowest_mean = node["params"]['stds'][node["params"]['means'].index(lowest_mean)]
        axis_min = lowest_mean - coverage_factor * std_of_lowest_mean

        step_size = min(node["params"]['stds']) / step_factor
        num_points = int((axis_max - axis_min) / step_size) 
        axis = np.linspace(axis_min, axis_max, num_points)

        cfm = sampled_cfm_gmm(ax=axis, mus=node["params"]["means"], sigmas=node["params"]["stds"])

        cfm_data[f"{node_name.lower()}"] = cfm.tolist()
    

for key, val in cfm_data.items():
    if "names" in key or "specs" in key or "given" in key:
        continue
    action_templates[key]["cfm"] = val

    val = np.array(val)
    accuracies = []
    for r in range(np.shape(val)[0]):
        for c in range(np.shape(val)[1]):
            if r == c:
                accuracies.append(val[r,c])
    action_templates[key]["accuracy"] = np.mean(accuracies)


# update the transitions if cfms were changed
transition_templates["category-material"] = cfm_data["mat-given-cat"]
transition_templates["material-category"] = cfm_data["cat-given-mat"]
transition_templates["density-material"] = cfm_data["density"]
transition_templates["elasticity-material"] = cfm_data["elasticity"]

with open('configs/actions.json', 'w') as file:
    json.dump(action_templates, file, indent=4)

with open('configs/transitions.json', 'w') as file:
    json.dump(transition_templates, file, indent=4)

if changed_at_least_once:
    with open('configs/confusion_matrices.json', 'w') as file:
        json.dump(cfm_data, file, indent=4)

