import math
import random
from copy import deepcopy
from src.filter import Filter

def select_most_eliminating_variable(formula, assignment, variables):
    max_score = -1
    best_var = None

    for var in variables:
        if var in assignment:
            continue

        elimination_count = 0
        for term, _ in formula:
            for v, polarity in term:
                if v == var:
                    elimination_count += 1
                    break

        p_false = 1 - Filter.get_filter(var).pass_probs['pass']
        weighted_score = elimination_count * p_false
        # print(var, weighted_score, elimination_count, p_false)

        if weighted_score > max_score:
            max_score = weighted_score
            best_var = var

    return best_var

def threshold_adjuster(p_current, p_target, i_count, i_expected, current_threshold, epsilon=0.1, delta_t=0.05, delta_i=0.02):
    """
    Dynamic Threshold Adjustment
    """
    alpha_new = current_threshold
    
    # Calculate power deviation as a ratio compared to target
    power_ratio = p_current / p_target
    
    if power_ratio > (1 + epsilon):
        power_deviation = (power_ratio - (1 + epsilon)) / (1 + epsilon)
        adjustment = delta_t * power_deviation
        alpha_new = current_threshold + adjustment
    elif power_ratio < (1 - epsilon):
        power_deviation = ((1 - epsilon) - power_ratio) / (1 - epsilon) 
        adjustment = delta_t * power_deviation
        alpha_new = current_threshold - adjustment
    
    if abs(i_count - i_expected) > delta_i * i_expected:
        image_deviation = (i_count - i_expected) / i_expected  # Relative deviation
        sign_factor = 1 if i_count > i_expected else -1
        image_adjustment = sign_factor * delta_i * min(1.0, abs(image_deviation))
        alpha_new = alpha_new + image_adjustment
    
    alpha_new = max(0.0, min(1.0, alpha_new))
    
    return alpha_new


def term_probability(term, assignment):
    """
    Estimate the probability that a term (conjunction of literals) is true.
    For each literal:
      - If assigned false, the term is false.
      - If assigned true, ignore that literal.
      - For unassigned, use p if literal is positive, or (1-p) if negative.
    The probability for the term is the product over unassigned literals.
    """
    for var, is_positive in term:
        if var in assignment:
            # If any literal is already false, the term is false.
            val = assignment[var]
            if not ((val and is_positive) or (not val and not is_positive)):
                return 0.0
    # For unassigned literals, multiply the probability that they are true.
    prob = 1.0
    for var, is_positive in term:
        if var not in assignment:
            p = Filter.get_filter(var).pass_probs['pass']
            prob *= p if is_positive else (1 - p)
    return prob


def ground_truth_priority(formula, simulated_assignment):
    """
    Calculate the ground truth priority for the DNF formula.
    The priority is the maximum priority of terms that are satisfied by the assignment.
    A term is satisfied if all its literals are true.
    """
    max_priority = 0
    for term, pri in formula:
        # Check if the term is satisfied by the simulated assignment.
        if all((simulated_assignment.get(var) == is_positive) for var, is_positive in term):
            max_priority = max(max_priority, pri)
    return max_priority


def overall_confidence_dnf(formula, assignment):
    """
    Estimate the overall confidence that the DNF formula is true.
    A term is true if all its literals are true.
    The formula is true if at least one term is true.
    We compute:
      P(formula true) = 1 - Π (1 - P(term true))
    If a term has been satisfied (empty term), we consider its probability 1.
    """

    if not formula:
        return 0.0, 0  # No terms → formula is false

    prod = 1.0
    for term in formula:
        # If term is empty, it means all literals have been satisfied.
        p_term = 1.0 if not term[0] else term_probability(term[0], assignment)
        if p_term == 1:
            return 1, term[1] # confidence, and priority
        prod *= (1 - p_term)

    return 1 - prod, 0

# --- Propagation and update functions for DNF ---

def propagate_dnf(formula, var, value):
    """
    Propagate the assignment of var=value in the DNF formula.
    For each term:
      - If a literal corresponding to var is present:
          * If its evaluation is false, the whole term becomes false (remove it).
          * If true, remove that literal from the term.
    """
    new_formula = []
    for term, pri in formula:
        new_term = []
        term_eliminated = False
        for literal in term:
            v, is_positive = literal
            if v == var:
                # Check the literal evaluation.
                if (value and is_positive) or (not value and not is_positive):
                    # Literal is true; do not include it further.
                    continue
                else:
                    # Literal is false; the whole term fails.
                    term_eliminated = True
                    break
            else:
                new_term.append(literal)
        if not term_eliminated:
            new_formula.append((new_term, pri))
    return new_formula

def estimate_delta_confidence_dnf(formula, var, assignment):
    """
    Estimate the change in overall confidence for the DNF formula if var is evaluated.
    Simulate both outcomes (True and False) and return the absolute difference
    between the current confidence and the expected new confidence.
    """
    current_conf, _ = overall_confidence_dnf(formula, assignment)
    
    # Simulate var = True
    assignment_true = assignment.copy()
    assignment_true[var] = True
    formula_true = propagate_dnf(formula, var, True)
    conf_true, _ = overall_confidence_dnf(formula_true, assignment_true)
    
    # Simulate var = False
    assignment_false = assignment.copy()
    assignment_false[var] = False
    formula_false = propagate_dnf(formula, var, False)
    conf_false, _ = overall_confidence_dnf(formula_false, assignment_false)
    
    p = Filter.get_filter(var).pass_probs['pass']
    expected_conf = p * conf_true + (1 - p) * conf_false
    delta = abs(expected_conf - current_conf)
    return delta

# --- The main greedy evaluation function for DNF formulas with dual bounds ---

def evaluate_formula_dnf(formula, variables, lower_threshold=0.1, upper_threshold=0.9, simulated_assignment=None, verbose=False, mode=1):
    """
    Evaluate the DNF formula by sequentially checking variables.
    Stops when either:
    - The confidence that formula is true is >= upper_threshold
    - The confidence that formula is false is >= (1 - lower_threshold)
    - All variables have been evaluated
    
    Returns the variable assignment, total evaluation time, and final confidence.
    """
    total_time = 0
    assignment = {}
    pri = 0


    if not simulated_assignment:
        simulated_assignment = {}
        for var in variables:
            simulated_assignment[var] = (random.random() < Filter.get_filter(var).pass_probs['pass'])

    current_formula = deepcopy(formula)
    
    current_confidence, _ = overall_confidence_dnf(current_formula, assignment)
    

    while (lower_threshold < current_confidence < upper_threshold) and set(variables) != set(assignment.keys()):
        if mode == 1:
            next_var = select_max_entropy_variable(current_formula, assignment, variables)
        else:
            next_var = select_most_eliminating_variable(current_formula, assignment, variables)
        
        # Simulate evaluation: decide truth value based on probability.
        result = simulated_assignment[next_var]
        total_time += Filter.get_filter(next_var).time
        assignment[next_var] = result
        
        # Propagate the assignment in the DNF formula.
        current_formula = propagate_dnf(current_formula, next_var, result)
        current_confidence, pri = overall_confidence_dnf(current_formula, assignment)
        
        # Print outcome message with explanation of bounds
        decision = ""
        if current_confidence >= upper_threshold:
            decision = " (STOPPING: Confidence above upper threshold)"
            pris = [pri for term, pri in current_formula if term]
            pri = max(max(pris), pri) if pris else pri


        elif current_confidence <= lower_threshold:
            decision = " (STOPPING: Confidence below lower threshold)"
            
        if verbose:
            print("Evaluated {} as {} (time += {:.2f}); new confidence = {:.3f}{}".format(next_var, result, Filter.get_filter(next_var).time, current_confidence, decision))
    
    return assignment, total_time, current_confidence, pri


def select_max_entropy_variable(formula, assignment, variables):
    scores = {}
    for var in variables:
        if var in assignment:
            continue
        
        # Calculate information gain using entropy and maximum confidence change
        p = Filter.get_filter(var).pass_probs['pass']
        eps = 1e-10  # Avoid log(0)
        entropy = - (p * math.log(p + eps) + (1 - p) * math.log(1 - p + eps))
        
        # Use max_confidence_change instead of estimate_delta_confidence_dnf
        delta_conf = max_confidence_change(formula, var, assignment)
        
        # Only consider variables that can actually change the outcome
        if delta_conf > 0:
            filter_time = Filter.get_filter(var).time
            scores[var] = (entropy * delta_conf) / filter_time
    
    if not scores:
        return None

    return max(scores, key=scores.get)


def max_confidence_change(formula, var, assignment):
    """
    Calculate the maximum possible change in confidence by evaluating var.
    This approach looks at the maximum difference between the two outcomes (True/False)
    rather than the expected change.
    
    Parameters:
    - formula: DNF formula as a list of terms
    - var: Variable to evaluate
    - assignment: Current variable assignment
    
    Returns:
    - Maximum possible change in confidence from evaluating this variable
    """
    # Simulate var = True
    assignment_true = assignment.copy()
    assignment_true[var] = True
    formula_true = propagate_dnf(formula, var, True)
    conf_true, _ = overall_confidence_dnf(formula_true, assignment_true)
    
    # Simulate var = False
    assignment_false = assignment.copy()    
    assignment_false[var] = False
    formula_false = propagate_dnf(formula, var, False)
    conf_false, _ = overall_confidence_dnf(formula_false, assignment_false)
    
    # Return the maximum possible change in confidence
    return abs(conf_true - conf_false)

