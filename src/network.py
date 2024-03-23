from __future__ import print_function, division
import numpy as np 
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import warnings, copy
import pandas as pd
import json
from collections import deque

class ContinuousDistribution(object):
    """
    Represents a continuous distribution defined over a given axis, with values 
    corresponding to probabilities or density values, and defined means and standard deviations.

    Attributes:
        step_size (float): The distance between consecutive points on the axis.
        size (int): The number of points on the axis.
        axis (numpy.ndarray): The points defining the axis of the distribution.
        vals (numpy.ndarray): The probability or density values corresponding to each point on the axis.
        means (numpy.ndarray): The means of the Gaussian distributions used to model the continuous distribution.
        stds (numpy.ndarray): The standard deviations of the Gaussian distributions used to model the continuous distribution.

    """
    def __init__(self, name, step_size, size, axis, vals, means, stds):
        """
        Initializes the ContinuousDistribution with the given parameters.
        
        Parameters:
            step_size (float): The distance between consecutive points on the axis.
            size (int): The total number of points on the axis.
            axis (numpy.ndarray): The points defining the axis of the distribution.
            vals (numpy.ndarray): The probability or density values for each point on the axis.
            means (numpy.ndarray): The means of the Gaussian distributions.
            stds (numpy.ndarray): The standard deviations of the Gaussian distributions.
        """
        self.name  = name
        self.step_size = step_size
        self.size  = size
        self.axis  = axis
        self.vals  = vals
        self.means = means
        self.stds  = stds

    def serialize(self):
        return {
            "name" : self.name,
            "step_size" : self.step_size,
            "size" : self.size,
            "axis" : np.array(self.axis).tolist(),
            "vals" : np.array(self.vals).tolist(),
            "means" : np.array(self.means).tolist(),
            "stds" : np.array(self.stds).tolist(),
        }

    def find_zero_on_axis(self):
        """Returns zero id of axis.
        """
        zero_bin = 100
        zero_id = 0
        for id, bin in enumerate(self.axis):
            if bin < 1 and bin > 0:
                if bin < zero_bin:
                    zero_bin = bin
                    zero_id = id
        return zero_id

    def sampler(self, n_samples):
        """
        Draws random samples from the distribution.

        Parameters:
            n_samples (int): Number of samples to draw.

        Returns:
            numpy.ndarray: Random samples drawn from the distribution.
        """
        ptx = np.random.choice(self.axis,
                               p=self.vals/np.sum(self.vals),
                               size=n_samples)
        
        return ptx
    
    def inverse_cdf_sampler(self, n_samples):
        """
        Samples from the distribution using the inverse CDF method.

        Parameters:
            n_samples (int): Number of samples to draw.

        Returns:
            numpy.ndarray: Random samples drawn from the distribution using the inverse CDF method.
        """
        cumulative = np.cumsum(self.vals)
        cumulative -= cumulative.min()
        f = interp1d(cumulative/cumulative.max(), self.axis)

        return f(np.random.random(n_samples))

    def get_gaussian_weights(self, n_samples=5000):
        """
        Computes the weights for the Gaussian components of the distribution.

        Parameters:
            n_samples (int, optional): Number of samples to use for fitting the Gaussian Mixture Model (GMM). Default is 5000.

        Returns:
            numpy.ndarray: The weights of the Gaussian components.

        Note:
            If the GMM does not converge, the function will attempt to refit the model with increased sample size and iterations.
        """
        
        self.vals = np.where(self.vals < 0, 1e-10, self.vals)
        sum_normalized_gmm = self.vals / np.sum(self.vals)
        basic_samples = np.random.choice(self.axis, p=sum_normalized_gmm, size=n_samples)
        gmm = GaussianMixture(n_components=len(self.means), means_init=self.means.reshape([-1,1]), tol=1e-6).fit(basic_samples.reshape([-1,1]))
        init_max_iter = gmm.n_iter_
        if gmm.converged_ == False:
            max_trials = 3

            for multiplier in range(2, 1+max_trials):
                new_samples = np.random.choice(self.axis, p=sum_normalized_gmm, size=n_samples)
                basic_samples = np.append(basic_samples, new_samples)
                gmm = GaussianMixture(n_components=len(self.means),
                                      means_init=self.means.reshape([-1,1]),
                                      tol=1e-6,
                                      max_iter=init_max_iter*multiplier
                                      ).fit(basic_samples.reshape([-1,1]))
                if gmm.converged_:
                    return gmm.weights_
            warnings.warn(f"GMM did not additionally converge after adding {multiplier}x more samples to the fit.")


        return gmm.weights_
    
    def plot(self):
        """
        Plots the Gaussian Mixture Model for continuous nodes.
        """
        
        plt.figure(figsize=(8, 3))
        plt.plot(self.axis, self.vals, label='Gaussian Mixture Model', lw=2)
        plt.title('GMM')
        plt.xlabel('{}'.format(self.name))
        plt.ylabel('p(x)')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Action(object):
    """
    Represents an action with its configuration including type, confusion matrix, accuracy, error, units, and target.

    Attributes:
        name (str): Name of the action.
        type (str): Type of the action.
        cfm (numpy.ndarray): Confusion matrix for the action's outcomes.
        accuracy (float): Accuracy of the action.
        error (float): Measurement error associated with the action.
        units (str): Units of measurement for the action's outcomes.
        target (str): Target node affected by the action.

    Parameters:
        cfg (dict): Configuration dictionary for the action, containing 'type', 'cfm', 'accuracy', 'error', 'units', and 'target'.
        name (str): Name of the action.
    """
    def __init__(self, cfg, name):
        """
        Initializes the Action object with the given configuration and name.
        """
        self.name = name
        self.type = cfg['type']
        self.cfm = cfg['cfm']  # confusion matrix
        self.accuracy = cfg['accuracy']
        self.error = cfg['error']
        self.units = cfg['units']
        self.target = cfg['target']
    
    def serialize(self):
        return self.__dict__


class Node(object):
    """
    Represents a node in a Bayesian network, encapsulating both continuous and categorical data.
    
    Attributes:
        name (str): The name of the node.
        type (str): The type of the node, either 'continuous' or 'categorical'.
        params (dict): Parameters of the node, including means, stds, and probabilities for categorical distributions.
        parents (list): A list of names of the parent nodes.
        children (list): A list of names of the child nodes.
        pdf (ContinuousDistribution): The probability distribution function for continuous nodes.
        pmf (numpy.ndarray): The probability mass function for categorical nodes.
        df (pd.DataFrame): DataFrame associated with the node, initially None.

    """
    def __init__(self, cfg, name):
        """
        Initializes the Node object with configuration settings and name.
        
        Args:
            cfg (dict): Configuration dictionary for the node containing type, params, parents, and children.
            name (str): Name of the node.
        """
        self.name = name
        self.type = cfg['type']
        self.params = cfg['params']
        self.parents = cfg['parents']
        self.children = cfg['children']        
        self.pdf = None
        self.pmf = None
        

    def serialize(self):
        return {
            "name" : self.name,
            "type" : self.type,
            "params" : self.params,
            "parents" : self.parents,
            "children" : self.children,
            "pdf" : self.pdf.serialize() if self.pdf else None,
            "pmf" : np.array(self.pmf).tolist(),
        }


    def init_pdf(self, weights, coverage_factor=5.0, step_factor=10.0):
        """
        Initializes the PDF for a continuous node based on the given weights and parameters.
        
        Args:
            weights (numpy.ndarray): The weights to use for initializing the PDF.
            coverage_factor (float): Determines the range of the axis as a multiple of the standard deviation.
            step_factor (float): Determines the step size in the axis as a fraction of the smallest standard deviation.
        
        >NOTE: This method is specifically designed for continuous nodes and will initialize the `pdf` attribute.
        """
    
        largest_mean = max(self.params['means'])
        std_of_largest_mean = self.params['stds'][self.params['means'].index(largest_mean)]
        axis_max = largest_mean + coverage_factor * std_of_largest_mean

        lowest_mean = min(self.params['means'])
        std_of_lowest_mean = self.params['stds'][self.params['means'].index(lowest_mean)]
        axis_min = lowest_mean - coverage_factor * std_of_lowest_mean

        step_size = min(self.params['stds']) / step_factor
        num_points = int((axis_max - axis_min) / step_size) 
        axis = np.linspace(axis_min, axis_max, num_points)

        # creation of the gaussian mixture model
        gmm = self.create_gmm(axis, weights)

        self.pdf = ContinuousDistribution(name=self.name, step_size=step_size, size=num_points, axis=axis, vals=gmm,
                                          means=np.array(self.params['means']),
                                          stds=np.array(self.params['stds']))
        
    def create_gmm(self, axis, weights):
        """
        Creates a Gaussian Mixture Model based on the provided axis and weights.
        
        Args:
            axis (numpy.ndarray): The axis over which the GMM is defined.
            weights (numpy.ndarray): The weights of the individual Gaussian components in the GMM.
        
        Returns:
            numpy.ndarray: The values of the GMM evaluated at each point in the axis.
        """
        gmm = np.zeros_like(axis)
        for i, (mean, std) in enumerate(zip(self.params['means'], self.params['stds'])):
            gmm += norm.pdf(axis, loc=mean, scale=std) * weights[i]

        return gmm
    
    def get_entropy(self):
        """
        Calculates the entropy of the node, differentiated by node type (categorical or continuous).
        
        Returns:
            float: The entropy of the node.
        
        Raises:
            ValueError: If the node type is unknown.
        """
        if self.type == 'categorical':

            return -np.sum(self.pmf * np.log(self.pmf, where=(self.pmf > 0)))
        
        elif self.type == 'continuous':
            product = self.pdf.vals * np.log(self.pdf.vals, where=(self.pdf.vals > 0))

            return -np.trapz(product, self.pdf.axis)
        else:
            raise ValueError("Unknown node type: {}.".format(self.type))
        
    def emulate_update(self, post_emul):
        """
        Updates the node's probability distribution (PDF or PMF) with the post-emulation data.
        
        Args:
            post_emul (numpy.ndarray): The post-emulation data to update the node's distribution.
        
        >NOTE: This method directly modifies the `pdf.vals` or `pmf` attribute based on the node's type.
        """
        if self.type == 'categorical':
            self.pmf = post_emul

        elif self.type == 'continuous':
            self.pdf.vals = post_emul
            
        else:
            raise ValueError("Unknown node type: {}.".format(self.type))

    
        
    def get_expected_entropy_given_action(self, action_pntr):
        """
        Calculates the expected entropy given an action for the node.
        
        Args:
            action_pntr (Action): The action pointer for which the expected entropy is calculated.
        
        Returns:
            tuple: A tuple containing the expected entropy and the posterior emulation.
        
        Raises:
            ValueError: If the node type is unknown.
        """

        if self.type == "categorical":
            cfm = np.array(action_pntr.cfm)
            entropies = -np.sum(cfm * np.log(cfm, where=(cfm > 0)), axis=0)
            meas_emulation = cfm @ self.pmf
            entropy_expectation = np.sum(meas_emulation * entropies)

            posterior_emulation = self.pmf * meas_emulation
            posterior_emulation /= np.sum(posterior_emulation)

            return entropy_expectation, posterior_emulation
            
        elif self.type == "continuous":
            # retrieve error gaussian
            sigma_m = action_pntr.error
            middle = int((self.pdf.axis[-1] + self.pdf.axis[0])/2)
            error_gaussian = norm.pdf(self.pdf.axis, loc=middle, scale=sigma_m)

            # convolve original gmm with the centered gaussian with error std
            # fftconvolve is more than 10 times as fast as np.convolve
            emulation = fftconvolve(self.pdf.vals, error_gaussian, mode='same')

            # no need to normalize here when normalizing done before entropy computation
            # eumlation /= np.trapz(emulation, self.pdf.axis)  

            posterior_emulation = emulation * self.pdf.vals
            posterior_emulation /= np.trapz(posterior_emulation, self.pdf.axis)

            product = posterior_emulation * np.log(posterior_emulation, where=(posterior_emulation > 0))
            entropy_expectation = -np.trapz(product, self.pdf.axis)

            return entropy_expectation, posterior_emulation
        else:
            raise ValueError("Unknown node type: {}.".format(self.type))
        
    def get_expected_entropy_given_transition(self, sending_node_pntr, trans_mat=None):
        """
        Calculates the expected entropy given a transition from a sending node to this node.
        
        Args:
            sending_node_pntr (Node): The pointer to the sending node.
            trans_mat (numpy.ndarray, optional): The transition matrix. Defaults to None.
        
        Returns:
            tuple: A tuple containing the expected entropy and the posterior emulation.
        
        Raises:
            ValueError: If the transition matrix is not provided for categorical target nodes.
        """

        # if current node is categorical
        if self.type == "categorical":
            if trans_mat is None:
                raise ValueError(f"Invalid message pass, transition matrix cannot be None for target node '{self.name}' of type '{self.type}'!")
            
            trans_mat = np.array(trans_mat)
            # if non-square trans_mat, make it square:
            if trans_mat.shape[0] != trans_mat.shape[1]:
                sq_trans_mat = trans_mat @ trans_mat.T
                sq_trans_mat /= np.sum(sq_trans_mat, axis=0)
                entropies = -np.sum(sq_trans_mat * np.log(sq_trans_mat, where=(sq_trans_mat > 0)), axis=0)

            else:
                entropies = -np.sum(trans_mat * np.log(trans_mat, where=(trans_mat > 0)), axis=0)

            # categorical to categorical
            if sending_node_pntr.type == "categorical":  
                message_emulation = trans_mat @ sending_node_pntr.pmf
                posterior_emulation = message_emulation * self.pmf
                posterior_emulation /= np.sum(posterior_emulation)

                return np.sum(message_emulation * entropies), posterior_emulation
            
            # continuous to categorical
            elif sending_node_pntr.type == "continuous":
                # gather the weights from the gmm
                gmm_fit_weights = sending_node_pntr.pdf.get_gaussian_weights()
                message_emulation = trans_mat @ gmm_fit_weights

                posterior_emulation = message_emulation * self.pmf
                posterior_emulation /= np.sum(posterior_emulation)

                return np.sum(message_emulation * entropies), posterior_emulation
            
        # if current node is continuous
        elif self.type == "continuous":

            # categorical to continuous
            if sending_node_pntr.type == "categorical":

                # i) create the gmm from the sending node pmf and data stored in receiveing node's pdf
                gmm_message_emulation = self.create_gmm(self.pdf.axis, sending_node_pntr.pmf)
                # gmm_message already contains the "emulation trans_mat" in the stds and means of the receiving node
                # therefore no need for next steps
                posterior_emulation = gmm_message_emulation * self.pdf.vals
                posterior_emulation /= np.trapz(posterior_emulation, self.pdf.axis)

                product = posterior_emulation * np.log(posterior_emulation, where=(posterior_emulation > 0))

                return -np.trapz(product, self.pdf.axis), posterior_emulation

            # continuous to continuous -> not present in the bayes net
            else:
                print(f"Error: Sending node {sending_node_pntr.name}, of type {sending_node_pntr.type}")
                print(f"for current node {self.name}, of type {self.type}")
                raise NotImplementedError(f"There is currently no other relationship from continuous node than continuous to categorical. Other are not implemented.")
        else:
            raise ValueError("Unknown node type: {}.".format(self.type)) 
        

class Network(object):
    """
    Represents a Bayesian network, containing nodes and actions, and manages their interactions.
    
    Attributes:
        action_names (list): A list of action names.
        actions (list): A list of Action objects associated with the network.
        node_names (list): A list of node names.
        nodes (list): A list of Node objects in the network.
        transitions (dict): A dictionary representing the transitions between nodes.

    """
    def __init__(self, actions_cfg, nodes_cfg):
        """
        Initializes the Network object with configurations for actions, nodes and transitions.
        
        Args:
            actions_cfg (dict): Configuration dictionary for actions.
            nodes_cfg (dict): Configuration dictionary for nodes.
        """
        self.action_names = []
        self.actions = []
        self.init_actions(actions_cfg)

        self.node_names = []
        self.nodes = []
        self.init_nodes(nodes_cfg)

        self.transitions = None
        self.init_transitions()

    def serialize(self):
        return {
            "action_names" : self.action_names,
            "actions" : [action.serialize() for action in self.actions],
            "node_names" : self.node_names,
            "nodes" : [node.serialize() for node in self.nodes],
            "transitions" : self.transitions
        }


    def init_transitions(self):
        """
        Initializes the transition probabilities between nodes from a configuration file (`transitions.json`).
        
        >NOTE: This method loads transitions from a JSON file and stores them in the `transitions` attribute.
        """
        with open('configs/transitions.json', 'r') as file:
            self.transitions = json.load(file)


    def update_neighbor(self, sending_node, neighbor_node):
        assert sending_node.type in ["categorical", "continuous"], f"sending_node.type '{sending_node.type}' is incorrect"
        assert neighbor_node.type in ["categorical", "continuous"], f"neighbor_node.type '{neighbor_node.type}' is incorrect"
        trans = self.get_transition(sending_node, neighbor_node)
        if sending_node.type == "categorical":
            if neighbor_node.type == "categorical":
                message = trans @ sending_node.pmf 
                posterior = message * neighbor_node.pmf
                neighbor_node.pmf = posterior / np.sum(posterior)

            elif neighbor_node.type == "continuous":
                gmm_message = neighbor_node.create_gmm(neighbor_node.pdf.axis, sending_node.pmf)
                posterior = gmm_message * neighbor_node.pdf.vals
                neighbor_node.pdf.vals = posterior / np.trapz(posterior, neighbor_node.pdf.axis)
            
        elif sending_node.type == "continuous":
            if neighbor_node.type == "categorical":
                gmm_fit_weights = sending_node.pdf.get_gaussian_weights()
                message = trans @ gmm_fit_weights
                posterior = neighbor_node.pmf * message
                neighbor_node.pmf = posterior / np.sum(posterior)

            # continuous to continuous -> not present in the bayes net
            else:
                print(f"Error: Sending node {sending_node.name}, of type {sending_node.type}")
                print(f"for neighbor node {neighbor_node.name}, of type {neighbor_node.type}")
                raise NotImplementedError(f"There is currently no other relationship from continuous node than continuous to categorical. Other are not implemented.")
            
        
    def get_transition(self, source_node, target_node):
        """
        Retrieves the transition matrix between two nodes, if it exists.
        
        Args:
            source_node (Node): The source node.
            target_node (Node): The target node.
        
        Returns:
            numpy.ndarray or None: The transition matrix if exists, otherwise None.
        
        Raises:
            ValueError: If transitions are not yet initialized or the desired key is not found.
        """
        if self.transitions is None:
            raise ValueError("Trnsition dictionary self.transitions was not yet initialized!")
        trans_keys = self.transitions.keys()
        desired_key = f"{source_node.name.lower()}-{target_node.name.lower()}"
        if desired_key not in trans_keys:
            return None
        else:
            return self.transitions[desired_key]
        

    def init_actions(self, cfg):
        """
        Initializes actions based on a configuration dictionary.
        
        Args:
            cfg (dict): The configuration dictionary for actions.
        """
        for key, action_cfg in cfg.items():
            self.action_names.append(key)
            self.actions.append(Action(action_cfg, key))

    def init_nodes(self, cfg):
        """
        Initializes nodes based on a configuration dictionary.
        
        Args:
            cfg (dict): The configuration dictionary for nodes.
        """
        for key, node_cfg in cfg.items():
            self.node_names.append(key)
            self.nodes.append(Node(node_cfg, key))

        # init continuous nodes according to categorical nodes
        for node in self.nodes:
            if node.type == "categorical":
                # init categorical distribution first
                node.pmf = np.array(node.params['probability'])

                # init continuous children
                for child_name in node.children:
                    child = self.get_node(child_name)
                    if child.type == "continuous":
                        child.init_pdf(weights=node.pmf)                       

    def get_node(self, name):
        """
        Retrieves a node by its name.
        
        Args:
            name (str): The name of the node to retrieve.
        
        Returns:
            Node: The node with the specified name.
        """
        return self.nodes[self.node_names.index(name)]
    
    def get_filtered_nodes(self, filter=None):

        options = ["continuous", "categorical"]
        if filter is None:
            return self.nodes
        if filter not in options:
            raise ValueError(f"Filter {filter} did not match any option from: {options}")
        
        if filter == "continuous":
            return [node for node in self.nodes if node.type == "continuous"]
        elif filter == "categorical":
            return [node for node in self.nodes if node.type == "categorical"]

    @staticmethod
    def __private_get_node(node_pntrs, name):
        """
        A private method to retrieve a node from a list of node pointers by name.
        
        Args:
            node_pntrs (list): A list of node pointers (Node objects).
            name (str): The name of the node to find.

        Returns:
            Node: The node with the given name, if found.

        >NOTE: Private, not intended for use outside of the Ntwork class.
        """
        for node in node_pntrs:
            if node.name == name:
                return node
            
    def get_action(self, name):
        return self.actions[self.action_names.index(name)]

    def expected_IGs_per_nodes(self, true_entropies, expected_entropies):
        """
        Calculates the expected information gain for each node given each action.
        
        Args:
            true_entropies: dictionary of true entropy values for each node, keyed by node names
            expected_entropies: pandas DataFrame where rows are actions, columns are nodes, values are expected entropy after measurement
        Returns:
            df_IGs: A pandas DataFrame where rows are actions and columns are nodes, values are expected information gains
        """
        IGs = {}
        for action_name in self.action_names:
            nodes_dict = {}
            for node in self.nodes:
                nodes_dict[node.name] = true_entropies[node.name].values[0] - expected_entropies[node.name][action_name]
            IGs[action_name] = nodes_dict
        
        df_IGs = pd.DataFrame(IGs)

        return df_IGs.T

    def get_true_entropies(self):
        """
        Calculates and returns the true entropies of all nodes in the network.
        
        Returns:
            df_true_ents: A pandas DataFrame with node names as keys and their entropies as values.
        """
        true_entropies = {}
        for node in self.nodes:
            true_entropies[node.name] = node.get_entropy()
        df_true_ents = pd.DataFrame([true_entropies])
        
        return df_true_ents
    
    def get_expected_entropies(self, debug_print=False):
        """
        Calculates and returns expected entropies for each action in the network.
        
        This method performs a breadth-first search (BFS) through the network,
        starting from the target node of each action and propagating through the network
        to estimate the entropies based on potential actions and transitions.

        Args:
            debug_print (bool, optional): If True, prints debug information during the execution. Useful for understanding the process and for debugging purposes. Defaults to False.

        Returns:
            dict: A nested dictionary where the first level keys are action names, and the second level keys are node types ('categorical' or 'continuous'). Each entry contains the expected entropy value for that action and node type.
            > NOTE: Returned dictionary also saved privately inside class.
        """
        
        expected_network_entropies = {}
        # i)  choose action
        for action in self.actions:

            action_entropies = {
                    "Material": 0.0,
                    "Category": 0.0,
                    "Density": 0.0,
                    "Elasticity": 0.0,
                    "Volume": 0.0
            }

            private_nodes = copy.deepcopy(self.nodes)
            target_node = self.__private_get_node(node_pntrs=private_nodes, name=action.target)

            # BFS init
            queue = deque([target_node])
            visited = set()
            if debug_print:
                print(f"\n> Beginning BFS for action: {action.name} at node {target_node.name}")
                print(f"{'it':<5}{'Current Node':<15}{'Queue':<40}{'Visited':<60}{'Neighbors':<30}")
            it = 0
            while queue:
                current_node = queue.popleft()
                neighbor_names = [current_node.children + current_node.parents][0]
                neighbor_names = [name for name in neighbor_names if name is not None]

                queue_names = [node.name for node in queue]  # Extract names for printing   
                # firt update
                if len(visited) == 0:
                    action_entropies[current_node.name], posterior_emulation = current_node.get_expected_entropy_given_action(action)
                    visited.add(current_node.name)
                    current_node.emulate_update(post_emul=posterior_emulation)
                    if debug_print:
                        print(f"{it:3d} {current_node.name:<15}{str(queue_names):<40}{str(list(visited)):<60}{str(neighbor_names):<30}")    

                # For debug print
                it += 1
                for neighbor_name in neighbor_names:

                    if neighbor_name not in visited and neighbor_name is not None:
                        visited.add(neighbor_name)
                        neighbor_node = self.__private_get_node(private_nodes, neighbor_name)
                        queue.append(neighbor_node)
                        expected_entropy, posterior_emulation = neighbor_node.get_expected_entropy_given_transition(sending_node_pntr=current_node,
                                                                                                                    trans_mat=self.get_transition(current_node, neighbor_node),
                                                                                                                    )
                        action_entropies[neighbor_node.name] = expected_entropy
                        neighbor_node.emulate_update(post_emul=posterior_emulation)
                        
                    queue_names = [node.name for node in queue]  # Extract names for printing   
                    if debug_print:
                        print(f"{it:2d} {current_node.name:<15}{str(queue_names):<40}{str(list(visited)):<60}{str(neighbor_names):<30}")    
            
            # add action_entropies to an action into network entropies dict
            expected_network_entropies[action.name] = action_entropies

            if debug_print:
                print("\nFinished BFS for action:", action.name)
                print("-" * 100)  # Separator for clarity
        df_exp_ents = pd.DataFrame(expected_network_entropies)

        return df_exp_ents.T
            

    
    def execute_action(self, action, experiment_pntr):
        
        # i)  pop the measurement
        meas = experiment_pntr.get(action)
        target_node = None

        # ii) incorporate the measurement
        if action == "cat-vision":
            category_node = self.get_node("Category")
            # introduce small error to measurement vector in case its in the form [1,0,...,0]
            noisy_meas = meas.introduce_error(meas.measurement_vector)
            actual_meas = noisy_meas
            category_posterior = category_node.pmf * noisy_meas
            category_node.pmf = category_posterior / np.sum(category_posterior)
            target_node = category_node

        elif action == "mat-vision" or action == "mat-sound":
            material_node = self.get_node("Material")
            # introduce small error to measurement vector in case its in the form [1,0,...,0]
            noisy_meas = meas.introduce_error(meas.measurement_vector)
            actual_meas = noisy_meas
            material_posterior = material_node.pmf * noisy_meas
            material_node.pmf = material_posterior / np.sum(material_posterior)
            target_node = material_node

        elif action == "density":
            volume_node = self.get_node("Volume")
            density_node = self.get_node("Density")

            #cut volume at 0 and normalize
            zero_id = volume_node.pdf.find_zero_on_axis()
            above_zero_volume_axis = volume_node.pdf.axis[zero_id:]
            above_zero_volume_gmm = volume_node.pdf.vals[zero_id:] 
            
            mass_samples = norm.rvs(loc=meas.meas_mean, scale=meas.meas_std, size=10000)  # grams
            
            volume_cm_samples = np.random.choice(above_zero_volume_axis,
                                    p=above_zero_volume_gmm/np.sum(above_zero_volume_gmm),
                                    size=10000)
            density_samples = mass_samples / volume_cm_samples  # g/cm^3
            density_samples *= 1000
            filtered_density_samples = density_samples[density_samples <= 11000]


            # now fit a density GMM to those samples
            reconstructed_gmm = GaussianMixture(
                                                n_components=len(density_node.params["means"]),
                                                means_init=np.array(density_node.params["means"]).reshape([-1,1]),
                                                tol=1e-6,
                                                max_iter=500
                                                ).fit(filtered_density_samples.reshape([-1,1]))
            reconstructed_density_gmm = np.zeros_like(density_node.pdf.axis)
            for i, (mean, std) in enumerate(zip(density_node.params["means"], density_node.params["stds"])):
                reconstructed_density_gmm += norm.pdf(density_node.pdf.axis, loc=mean, scale=std) * reconstructed_gmm.weights_[i]

            # now use the reconstructed density gmm as a measurement
            actual_meas = reconstructed_density_gmm
            density_posterior = density_node.pdf.vals * reconstructed_density_gmm
            density_node.pdf.val = density_posterior / np.trapz(density_posterior, density_node.pdf.axis)
            target_node = density_node
    
        elif action == "elasticity":
            elasticity_node = self.get_node("Elasticity")
            action_error = self.get_action(action).error
            actual_meas = norm.pdf(elasticity_node.pdf.axis, loc=meas.meas_mean, scale=action_error)
            elasticity_posterior = elasticity_node.pdf.vals * actual_meas
            elasticity_node.pdf.vals = elasticity_posterior / np.trapz(elasticity_posterior, elasticity_node.pdf.axis)
            target_node = elasticity_node

        else:
            raise ValueError(f"Unknown action '{action}'.")
        
        # iii) BFS

        queue = deque([target_node])
        visited = set()

        while queue:
            current_node = queue.popleft()
            neighbor_names = [current_node.children + current_node.parents][0]
            neighbor_names = [name for name in neighbor_names if name is not None]

            if len(visited) == 0:
                visited.add(current_node.name)
            
            for neighbor_name in neighbor_names:
                if neighbor_name not in visited and neighbor_name is not None:
                    visited.add(neighbor_name)
                    neighbor_node = self.get_node(neighbor_name)
                    queue.append(neighbor_node)

                    # update neighbour
                    self.update_neighbor(sending_node=current_node, neighbor_node=neighbor_node)
        
        return actual_meas

        

        
            

        