<p align="center">
  <img src="_src_images/actsel_github_banner.png" alt="Action Selection algorithm to explore the physical properties in broader term.">
</p>

# ACTSEL source code repository
We introduce ACTSEL. A method for automatic selection of actions that help optimally determine physical object properties that are not readily available through vision.

## Graphical model overview
<p align="center">
  <img src="_src_images/actsel_general_diagram.png" alt="General diagram of ACTSEL algorithm in action."
  height="300"
  style="margin-right: 20px;">
  <img src="_src_images/actsel_actions_network.png" alt="Bayesian network" height="300">
</p>
<p align="center">
  <em>General overview of the algorithm (left), Bayesian network and action relations (right)</em>
</p>

## Running the model
> For best experience install conda environment as `numpy`, `scipy` and `scikit-learn` are needed for algorithm operation
  1) To run the model, fill in the templates for nodes, actions and their relevant confusion matrices in `configs/templates`. In order to update the actual config `.json` files, run the `scripts/templates_to_cfgs.py` from root directory as:
  ```
  python3 scripts/templates_to_cfgs.py
  ```

  2) Customize the `main.py` to meet your action and object requirements byt customizing `experiment_object_names` and action to node mapping.

### Implementation remarks
The algorithm and results presented in the paper were obtained offline on pre-measured dataset for broader statistical understanding. This fact is reflected in `main.py`.

## Publication, video, data
Kruzliak, A.; Hartvich, J.; Patni, S. P.; Rustler, L.; Behrens, J. K.; Abu-Dakka, F. J.; Mikolajczyk, K.; Kyrki, V. & Hoffmann, M. (2024). Interactive Learning of Physical Object Properties Through Robot Manipulation and Database of Object Measurements, in 'Intelligent Robots and Systems (IROS), IEEE/RSJ International Conference on', pp. 7596-7603.
* Full text: [DOI - IEEE Xplore](https://doi.org/10.1109/IROS58592.2024.10802249) , [pdf-arxiv](https://arxiv.org/abs/2404.07344)
* Video: [youtube](https://youtu.be/h_ZIYUmzv-8)
* Database of object measurements and its source code: [link](https://cmp.felk.cvut.cz/ipalm/) 
