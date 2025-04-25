import random
from src.data import Data
from src.utils import Time
from . import log
from src.filter import Filter
import src.formula as fx
from matplotlib.patches import Polygon
import src.multitask_formula as mtl

SEED = 42 # for reproducibility

def evaluate_image_serval(formula, simulated_assignment, debug=False):
    compute_time = 0
    filter_results = {}

    for filter_group, group_priority in formula:
        satisfied = True
        for filter_id, _ in filter_group:
            if filter_id not in filter_results:
                filter : Filter = Filter.get_filter(filter_id)
                compute_time += filter.time
                filter_results[filter_id] = simulated_assignment[filter_id]

            if debug:
                print(f"Filter ID: {filter_id}, Result: {filter_results[filter_id]}")
            
            if not filter_results[filter_id]:
                satisfied = False
                break
            
        if satisfied:
            return group_priority, compute_time
    return 0, compute_time

def evaluate_image_earthsight(formula, simulated_assignment, unique_filters, registry, debug=False):
    if len(formula) == 0:
        return 0, 0
                
    if registry:
        _assignment, compute_time, confidence, priority = mtl.evaluate_formula_dnf_multitask(formula, registry.copy(), lower_threshold=0.0, upper_threshold=0.7, simulated_assignment=simulated_assignment, debug=False)
    else:
        _assignment, compute_time, confidence, priority = fx.evaluate_formula_dnf(formula, unique_filters, lower_threshold=0.0, upper_threshold=0.7, simulated_assignment=simulated_assignment, mode=2, verbose=debug)
                
    if debug:
        print(f"ES Assignment: {_assignment}")
        print(f"ES Confidence: {confidence}")        
    return priority, compute_time


def evaluate_image(formula, mode, registry, include_fnr=False, compare = True):
        if len(formula) != 0 and random.random() < 0:
            return 0, 0, 0

        global SEED
        random.seed(SEED)
        SEED += 1

        simulated_assignment = {}
        unique_filters = set(filter for term in formula for filter, polarity in term[0])
        
        for filter in unique_filters:
            simulated_assignment[filter] = (random.random() < Filter.get_filter(filter).pass_probs['pass'])
        ground_truth_pri = fx.ground_truth_priority(formula, simulated_assignment)

        if include_fnr:
            for filter in unique_filters:
                if simulated_assignment[filter]:
                    fpr = Filter.get_filter(filter).false_negative_rate
                    simulated_assignment[filter] = [random.random() >= fpr] # keep it true with 1 - false negative rate

        if not compare:
            if mode == "serval":
                priority, compute_time = evaluate_image_serval(formula, simulated_assignment)
    
            elif mode == "earthsight":
                priority, compute_time = evaluate_image_earthsight(formula, simulated_assignment, unique_filters, registry)
                
                if ground_truth_pri > 1 and ground_truth_pri > priority:
                    log.Log("PRIORITIZATION ERROR", {"ground_truth_pri": ground_truth_pri, "computed_priority": priority})

            return priority, compute_time, ground_truth_pri
        
        if compare:
            serval_priority, serval_compute_time = evaluate_image_serval(formula, simulated_assignment)
            earthsight_priority, earthsight_compute_time = evaluate_image_earthsight(formula, simulated_assignment, unique_filters, registry)


            if mode == "serval":
                priority, compute_time = serval_priority, serval_compute_time
            elif mode == "earthsight":
                priority, compute_time = earthsight_priority, earthsight_compute_time

            return priority, compute_time, ground_truth_pri

            

class Image(Data):
    id = 0
    def __init__(self, size: int, time: 'Time', coord=[0,0],
                 name="", satellite = None):
        
        """
        Arguments:
                size (float) - size of image in m
                region (Polygon) - region of image
                time (datetime) - time of image
                name: name of the image
        """
        super().__init__(size)
        self.satellite = satellite
        self.time = time
        self.coord = coord
        self.size = size
        self.id = Image.id
        self.score = -1 # priority score
        self.compute_time = 0 # time taken to compute the score
        Image.id += 1
        self.name = name
        self.descriptor = -1 # ground truth value
        self.earliest_possible_transmit_time = None

    def set_score(self, value):
        self.score = value

    @classmethod
    def set_id(cls, value):
        cls.id = value

    @staticmethod
    def from_dict(data):
        min_x, min_y, max_x, max_y = data['region']
        region = Polygon([(min_x, min_y), (max_x, min_y),
                         (max_x, max_y), (min_x, max_y)])
        return Image(
            **data,
            region=region
        )

    # To implement custom comparator (on the score) for the priority queue in the detector
    def __lt__(self, obj):
        """self < obj."""
        # Priority queue is a min heap while we want to put the highest score first
        # So we reverse the comparison
        return self.score > obj.score if not self.score == obj.score else self.time < obj.time

    def __le__(self, obj):
        """self <= obj."""
        return self < obj or self == obj

    def __eq__(self, obj):
        """self == obj."""
        return self.score == obj.score and self.time == obj.time

    def __ne__(self, obj):
        """self != obj."""
        return not self == obj

    def __gt__(self, obj):
        """self > obj."""
        return not self <= obj

    def __ge__(self, obj):
        """self >= obj."""
        return not self < obj

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return "{{imageId: {}, imageSize: {}, imageScore: {}, imageName: {}}}".format(self.id, self.size, self.score, self.name)
