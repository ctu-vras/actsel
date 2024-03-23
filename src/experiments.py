import os, json, random
import numpy as np

MAT_NAMES = ["ceramic", "glass", "metal", "soft-plastic", "hard-plastic", "wood", "paper", "foam"]
CAT_NAMES = ["bottle", "bowl", "box", "can", "dice", "fruit", "mug", "plate", "sodacan", "wineglass"]

class Measurement:
    def __init__(self, measurement_vector, meas_mean, meas_std, unit, names):
        self.measurement_vector = measurement_vector
        self.meas_mean = meas_mean
        self.meas_std = meas_std
        self.unit = unit
        self.names = names

    def __repr__(self):
        address = hex(id(self))
        return f"<Measurement(mean={self.meas_mean}, std={self.meas_std}, unit={self.unit}) at {address}>"

    def __str__(self):
        address = hex(id(self))
        return (f"<Measurement at {address}>\n"
                f"  Measurement Vector: {self.measurement_vector}\n"
                f"  Mean: {self.meas_mean}, Standard Deviation: {self.meas_std}"
                f"  Unit: {self.unit}")
    
    def introduce_error(self, vector, error=0.01):
        assert(error >= 0 and error <= 1)
        vector_len = len(vector)
        cnm = np.eye(vector_len) * (1 - error)
        piecewise_error = error / (vector_len - 1)
        for r in range(vector_len):
            for c in range(vector_len):
                if r == c:
                    continue
                cnm[r, c] = piecewise_error

        noisy_vector = cnm @ vector
        return noisy_vector

class ExperimentObject:
    
    def __init__(self, object_name, base_path):
        self.object_name = object_name.replace('experiment_', '')
        self.base_path = os.path.join(base_path, object_name)
        self.actions = {'mat-vision': [], 'mat-sound': [], 'cat-vision': [], 'density': [], 'elasticity': []}
        self._load_actions_data()

    def reload_measurements(self):
        self.actions = {'mat-vision': [], 'mat-sound': [], 'cat-vision': [], 'density': [], 'elasticity': []}
        self._load_actions_data()
    
    def _load_actions_data(self):
        for action in self.actions.keys():
            for i in range(5):  # Assuming there are 5 instances per action
                action_instance_path = os.path.join(self.base_path, f"{action}_{i}", "data")
                measurement = self._load_measurement_from_path(action_instance_path)
                if measurement:
                    self.actions[action].append(measurement)
            random.shuffle(self.actions[action])  # Shuffle the list for random access

    def _order_dict_to_vector(self, data_dict, ordered_names):

        other_category = 0
        meas_vector = np.zeros_like(ordered_names, dtype=np.float32)
        for vision_name, val in data_dict.items():
            if vision_name in ordered_names:
                meas_vector[ordered_names.index(vision_name)] = val
            elif vision_name == "plastic":
                # split general plastic into soft and hard equally
                meas_vector[ordered_names.index("soft-plastic")] = val/2
                meas_vector[ordered_names.index("hard-plastic")] = val/2
            else:
                other_category += val

        if other_category > 0:
            untouched_categories = np.where(meas_vector < 1e-5)[0]
            uniform_addition = other_category / len(untouched_categories)

            # add the uniform addition to the unseen materials from our vector
            # to account for difference betewen vision module materials and used subset of materials
            meas_vector[untouched_categories] += uniform_addition
        
        meas_vector /= np.sum(meas_vector)
        return meas_vector


    def _load_measurement_from_path(self, path):
        measurement_path = os.path.join(path, "measurement.json")
        if os.path.exists(measurement_path):
            with open(measurement_path, 'r') as file:
                data = json.load(file)
                action = data["meas_prop"]

                meas = None

                # preprocessing data based on type of action
                if action == "mat-vision":
                    meas_vector = self._order_dict_to_vector(data["values"]["prediction"], MAT_NAMES)
                    meas = Measurement(measurement_vector=meas_vector, meas_mean=None, meas_std=None, unit="", names=MAT_NAMES)

                elif action == "mat-sound":
                    # reorder dict data according to names
                    meas_vector = np.zeros_like(MAT_NAMES, dtype=np.float32)
                    for sound_name, val in data['values'].items():
                        meas_vector[MAT_NAMES.index(sound_name)] = val

                    meas = Measurement(measurement_vector=meas_vector, meas_mean=None, meas_std=None, unit="", names=MAT_NAMES)

                elif action == "cat-vision":
                    
                    meas_vector = self._order_dict_to_vector(data["values"]["prediction"], CAT_NAMES)
                    meas = Measurement(measurement_vector=meas_vector, meas_mean=None, meas_std=None, unit="", names=MAT_NAMES)

                elif action == "density":
                    # even though named torques, it is values in grams actually
                    if "arm" in data['values']:
                        mean_weight = np.mean(data["values"]["arm"]["joint4_torque"])
                        std_weight = np.std(data["values"]["arm"]["joint4_torque"])
                    else:
                        mean_weight = np.mean(data["values"])
                        std_weight = np.std(data["values"])
                    meas = Measurement(meas_mean=mean_weight, meas_std=std_weight, unit="g", measurement_vector=None, names=None)

                elif action == "elasticity":
                    meas = Measurement(meas_mean=data["params"]["mean"], meas_std=data["params"]["sigma"], unit="kPa", measurement_vector=None, names=None)
                else:
                    raise ValueError(f"Unknown action: `{action}`.")
     
                return meas
        return None

    def get(self, action):
        if action in self.actions and self.actions[action]:
            measurement = self.actions[action].pop()
            if len(self.actions[action]) < 3:
                print(f"Warning! Number of measurements for {action} is less than 3!")
            return measurement
        return None
    
    def __repr__(self):
        address = hex(id(self))
        return f"<ExperimentObject(object_name={self.object_name}) at {address}>"

    def __str__(self):
        experiments_summary = {action: len(measurements) for action, measurements in self.experiments.items()}
        address = hex(id(self))
        return (f"<ExperimentObject at {address}>\n"
                f"  Object Name: {self.object_name}\n"
                f"  Experiments: {experiments_summary}")


if __name__ == "__main__":

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
        
    # Assuming the directory structure is /base_path/object_name/action_i/data
    base_path = '/home/andrej/Documents/CODE/ipalm_acsel2/data/source/experiment_17_objects_clean'

    experiment_object = ExperimentObject("experiment_banana_ycb", base_path)

    for name in experiment_object_names:
        exp = ExperimentObject(name, base_path)
        print(f"Experiment {exp.object_name}; elasticity:")
        for a in exp.actions['cat-vision']:
            print(a.measurement_vector)


    