import math
import random
from copy import deepcopy
from collections import defaultdict
from src.filter import Filter

import math
import random
from copy import deepcopy
from collections import defaultdict
from src.filter import Filter

class Model:
    """Base class for all models (filters)"""
    def __init__(self, name, execution_time):
        self.name = name
        self.execution_time = execution_time
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

class BackboneModel(Model):
    """Backbone model that must be executed before its dependent child models"""
    def __init__(self, name, execution_time):
        super().__init__(name, execution_time)
        self.child_models = []
        
    def add_child(self, child_model):
        """Add a child model that depends on this backbone"""
        self.child_models.append(child_model)
        child_model.backbone = self

class ClassifierModel(Model):
    """Model that produces true/false classification outputs"""
    def __init__(self, name, execution_time, pass_probability):
        super().__init__(name, execution_time)
        self.pass_probability = pass_probability
        self.backbone = None  # Will be set if this model depends on a backbone

class ModelRegistry:
    """Registry to manage all models and their dependencies"""
    def __init__(self):
        self._models = {}
        self._dependency_graph = defaultdict(list)
        self._execution_status = {}  # Tracks which models have been executed
        self._filter_to_model = {}   # Maps filter IDs to model names
        
    def register_model(self, model):
        """Register a model in the registry"""
        self._models[model.name] = model
        self._execution_status[model.name] = False
        
        # If it's a classifier with a backbone, update dependency graph
        if isinstance(model, ClassifierModel) and model.backbone:
            self._dependency_graph[model.name].append(model.backbone.name)
            
    def get_model(self, name):
        """Get a model by name"""
        return self._models.get(name)
    
    def get_all_classifier_models(self):
        """Get all classifier models"""
        return {name: model for name, model in self._models.items() 
                if isinstance(model, ClassifierModel)}
    
    def get_effective_execution_time(self, model_name):
        """
        Calculate the effective execution time for a model, considering:
        - The model's own execution time
        - The backbone's execution time if it hasn't been executed yet
        """
        model = self.get_model(model_name)
        if not model:
            return 0
            
        # Base execution time is the model's own time
        time = model.execution_time
        
        # If it's a classifier with a backbone that hasn't been executed yet
        if (isinstance(model, ClassifierModel) and 
            model.backbone and 
            not self._execution_status[model.backbone.name]):
            time += model.backbone.execution_time
            
        return time
    
    def mark_executed(self, model_name):
        """Mark a model as executed"""
        if model_name not in self._execution_status:
            return
            
        self._execution_status[model_name] = True
        
        # If it's a backbone, update the dependency graph for all its children
        model = self.get_model(model_name)
        if isinstance(model, BackboneModel):
            for child in model.child_models:
                child_name = child.name
                if model_name in self._dependency_graph.get(child_name, []):
                    self._dependency_graph[child_name].remove(model_name)
    
    def get_executable_models(self):
        """
        Get all models that can be executed now (all dependencies satisfied)
        """
        executable = []
        for name, model in self._models.items():
            if not self._execution_status[name] and not self._dependency_graph.get(name, []):
                executable.append(name)
        return executable
    
    def register_filter_model_mapping(self, filter_id, model_name):
        """Register a mapping from filter ID to model name"""
        self._filter_to_model[filter_id] = model_name
        
    def get_model_by_filter_id(self, filter_id):
        """Get a model by filter ID"""
        model_name = self._filter_to_model.get(filter_id)
        if model_name:
            return self.get_model(model_name)
        return None
    
    def get_filter_id_from_model_name(self, model_name):
        """Get filter ID from model name"""
        # Extract filter ID from model name (e.g., "F1_Water_Extent" -> "F1")
        for filter_id, mapped_name in self._filter_to_model.items():
            if mapped_name == model_name:
                return filter_id
        return None
    
    def copy(self):
        """Create a deep copy of the registry"""
        new_registry = ModelRegistry()
        new_registry._models = self._models # can stay the same
        new_registry._dependency_graph = deepcopy(self._dependency_graph)
        new_registry._execution_status = deepcopy(self._execution_status)
        new_registry._filter_to_model = self._filter_to_model # can stay the same
        return new_registry

def find_highest_satisfied_priority(formula):
    """
    Find the highest priority of any satisfied term in the formula.
    A term is satisfied if it has no literals left (empty term).
    """
    highest_priority = 0
    for term, priority in formula:
        if not term:  # Empty term means it's satisfied
            highest_priority = max(highest_priority, priority)
    return highest_priority

def find_highest_possible_priority(formula):
    """
    Find the highest priority of any term that could still be satisfied.
    """
    return max([priority for _, priority in formula], default=0)

def evaluate_formula_dnf_multitask(formula, model_registry : ModelRegistry, lower_threshold=0.0, upper_threshold=1.0, simulated_assignment=None, debug=False):
    """
    Evaluates a DNF formula using the multitask learning approach.
    Continuously propagates formula and reevaluates scores after each filter execution.
    Only discards formula when proven false.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    total_time = 0.0
    assignment = {}
    
    # Initialize with the original formula
    current_formula = formula.copy()
    
    while current_formula:
        # Identify all filters in the current formula - handle the correct structure
        unique_filters = set()
        for term_tuple in current_formula:
            term = term_tuple[0]  # Get the term (list of filter conditions)
            for filter_condition in term:
                fid = filter_condition[0]  # Extract filter ID
                unique_filters.add(fid)
        
        # If we've already evaluated all filters, check the result
        if all(fid in assignment for fid in unique_filters):
            break
            
        # Calculate scores for remaining filters
        scores = {}
        for fid in unique_filters:
            if fid in assignment:
                continue  # Skip already evaluated filters
                
            model : ClassifierModel = model_registry.get_model_by_filter_id(fid)
            if model:
                p = model.pass_probability
                t = model.execution_time if model.execution_time > 0 else 0.01
                scores[fid] = (1 - p) / t
            else:
                f_obj : Filter = Filter.get_filter(fid)
                p = f_obj.pass_probs["pass"]
                t = f_obj.time if f_obj.time > 0 else 0.01
                scores[fid] = (1 - p) / t
        
        if not scores:
            break  # No more filters to evaluate
            
        # Select the highest scoring filter
        best_filter = max(scores.items(), key=lambda x: x[1])[0]
        
        # Execute the selected filter
        model = model_registry.get_model_by_filter_id(best_filter)
        
        # Run the backbone first if needed
        if model and model.backbone and not model_registry._execution_status.get(model.backbone.name, False):
            backbone : Model = model.backbone
            total_time += backbone.execution_time
            model_registry.mark_executed(backbone.name)
        
        # Determine execution time
        exec_time = model.execution_time if model else Filter.get_filter(best_filter).time
        
        # Simulate evaluation
        result = simulated_assignment.get(best_filter) if simulated_assignment else (random.random() < 0.5)
        total_time += exec_time
        assignment[best_filter] = result
        
        if debug:
            print(f"Evaluated {best_filter} = {result}, time so far: {total_time}")
        
        # Propagate the formula based on this result
        current_formula = propagate_formula(current_formula, best_filter, result)
        
        # Check if we can determine the result early
        satisfied_term = find_satisfied_term(current_formula, assignment)
        
        if satisfied_term is not None:
            # We found a satisfied term, return its priority
            return assignment, total_time, 1.0, satisfied_term[1]
        
        if not current_formula:
            # No satisfiable terms remain
            return assignment, total_time, 1.0, 0
    
    # After evaluating all necessary filters, find the highest priority of any satisfied term
    result_priority = find_highest_satisfied_priority(formula, assignment)
    return assignment, total_time, 1.0, result_priority

def propagate_formula(formula, filter_id, result):
    """
    Propagates a DNF formula after evaluating a filter.
    Returns the updated formula with simplified terms.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    updated_formula = []
    
    for term_tuple in formula:
        term = term_tuple[0]  # Extract the term (list of filter conditions)
        priority = term_tuple[1]  # Extract the priority
        
        new_term = []
        term_falsified = False
        
        for filter_condition in term:
            fid = filter_condition[0]  # Extract filter ID
            polarity = filter_condition[1]  # Extract polarity (True/False)
            
            if fid == filter_id:
                # If this filter doesn't match the expected value, the term is falsified
                if polarity != result:
                    term_falsified = True
                    break
                # If it does match, we don't need to include it in the new term
                continue
            else:
                # Keep other filters in the term
                new_term.append((fid, polarity))
        
        if not term_falsified:
            # If term is empty, it's satisfied; otherwise add the simplified term
            if not new_term:
                # Empty term means it's satisfied - return just this term
                return [([], priority)]
            else:
                updated_formula.append((new_term, priority))
    
    return updated_formula

def find_satisfied_term(formula, assignment):
    """
    Find a term that is already satisfied by the current assignment.
    Returns the term if found, None otherwise.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    for term_tuple in formula:
        term = term_tuple[0]  # Extract the term (list of filter conditions)
        priority = term_tuple[1]  # Extract the priority
        
        satisfied = True
        
        for filter_condition in term:
            fid = filter_condition[0]  # Extract filter ID
            polarity = filter_condition[1]  # Extract polarity (True/False)
            
            if fid not in assignment or assignment[fid] != polarity:
                satisfied = False
                break
        
        if satisfied:
            return term_tuple
    
    return None

def find_highest_satisfied_priority(formula, assignment):
    """
    Find the highest priority of any satisfied term in the original formula.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    highest_priority = 0
    
    for term_tuple in formula:
        term = term_tuple[0]  # Extract the term (list of filter conditions)
        priority = term_tuple[1]  # Extract the priority
        
        satisfied = True
        
        for filter_condition in term:
            fid = filter_condition[0]  # Extract filter ID
            polarity = filter_condition[1]  # Extract polarity (True/False)
            
            if fid not in assignment or assignment[fid] != polarity:
                satisfied = False
                break
        
        if satisfied and priority > highest_priority:
            highest_priority = priority
    
    return highest_priority

def create_model_registry_from_filters(all_filters : list[Filter]):
    """
    Convert the existing Filter objects to the new Model framework
    and register them in a ModelRegistry.
    
    Returns:
    - ModelRegistry populated with models based on our filters
    - Dictionary mapping filter_id to model_name
    """
    registry = ModelRegistry()
    filter_to_model_map = {}
    
    # Group filters by their domain/category
    filter_categories = {
        "flood": [f for f in all_filters if f.filter_id.startswith("F")],
        "wildfire": [f for f in all_filters if f.filter_id.startswith("W")],
        "earthquake": [f for f in all_filters if f.filter_id.startswith("E")],
        "ship": [f for f in all_filters if f.filter_id.startswith("S") and not f.filter_id.startswith("SC")],
        "ship_class": [f for f in all_filters if f.filter_id.startswith("SC")],
        "aircraft": [f for f in all_filters if f.filter_id.startswith("A")],
        "general": [f for f in all_filters if f.filter_id.startswith("G")],
        "infrastructure": [f for f in all_filters if f.filter_id.startswith("I")],
        "security": [f for f in all_filters if f.filter_id.startswith("B")]
    }
    
    # Create backbone models for each category
    backbones = {}
    for category, filters in filter_categories.items():
        if filters:  # Only create backbone if there are filters in this category
            backbone_name = f"{category}_backbone"
            # Use average cost from the filters as backbone execution time
            # median cost
            median_cost = sorted(f.time for f in filters)[len(filters) // 2]
            backbone = BackboneModel(backbone_name, median_cost * 1.2)  # Backbone slightly slower than classifiers
            registry.register_model(backbone)
            backbones[category] = backbone
    
    # Create classifier models for each filter and attach to appropriate backbone
    for f in all_filters:
        model_name = f"{f.filter_id}_{f.filter_name.replace(' ', '_')}"
        # Determine which category this filter belongs to
        category = None
        for cat_name, filters in filter_categories.items():
            if f in filters:
                category = cat_name
                break
        
        if category:
            # Create classifier model
            classifier = ClassifierModel(
                model_name, 
                f.time * 0.2,  # Use the filter's cost as execution time
                f.pass_probs["pass"]  # Use the pass probability
            )
            
            # Connect to backbone
            if category in backbones:
                backbones[category].add_child(classifier)
            
            # Register model
            registry.register_model(classifier)
            
            # Map filter_id to model_name
            filter_to_model_map[f.filter_id] = model_name

    
    registry._filter_to_model = filter_to_model_map
    # Create dependencies between related models
    
    return registry, filter_to_model_map