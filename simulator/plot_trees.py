import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from src.filter import Filter
from src.formula import overall_confidence_dnf, propagate_dnf, max_confidence_change, evaluate_formula_dnf

def build_evaluation_tree(formula, variables, lower_threshold=0.1, upper_threshold=0.9, max_depth=4):
    """
    Build a decision tree for the DNF formula evaluation process.
    
    Parameters:
    - formula: DNF formula as a list of terms, where each term is a list of (var, is_positive) tuples
    - variables: list of variable names
    - lower_threshold: confidence below this means "formula is false"
    - upper_threshold: confidence above this means "formula is true"
    - max_depth: maximum depth of the tree to prevent exponential growth
    
    Returns:
    - tree: dictionary representing the evaluation tree
    """
    # Build the tree recursively
    def expand_node(current_formula, assignment, depth, path):
        current_confidence = overall_confidence_dnf(current_formula, assignment)
        
        node = {
            "id": path,
            "confidence": current_confidence,
            "assignment": assignment.copy(),
            "depth": depth,
            "children": [],
            "status": "internal",
            "formula": deepcopy(current_formula)
        }
        
        # Check termination conditions
        if current_confidence >= upper_threshold:
            node["status"] = "true"
            node["label"] = f"FORMULA TRUE\n({current_confidence:.3f} ≥ {upper_threshold})"
            return node
            
        if current_confidence <= lower_threshold:
            node["status"] = "false"
            node["label"] = f"FORMULA FALSE\n({current_confidence:.3f} ≤ {lower_threshold})"
            return node
            
        if set(variables) == set(assignment.keys()):
            node["status"] = "complete"
            node["label"] = f"All vars evaluated\nConf: {current_confidence:.3f}"
            return node
            
        if depth >= max_depth:
            node["status"] = "max_depth"
            node["label"] = f"Max depth reached\nConf: {current_confidence:.3f}"
            return node
        
        # Print formula state
        print(f"Node {path}: Current formula = {current_formula}")
        print(f"Node {path}: Assignment = {assignment}")
        print(f"Node {path}: Confidence = {current_confidence:.4f}")
        
        # Find best next variable to evaluate
        scores = {}
        print(f"Node {path}: Calculating variable scores:")
        for var in variables:
            if var in assignment:
                continue
                
            # Calculate scores using information gain and cost metrics
            p = Filter.get_filter(var).pass_probs['pass']
            
            # Calculate entropy (information content)
            eps = 1e-10  # Avoid log(0)
            entropy = - (p * math.log(p + eps) + (1 - p) * math.log(1 - p + eps))
            
            # Calculate maximum possible confidence change (NOT expected change)
            conf_delta = max_confidence_change(current_formula, var, assignment)
            
            # Get filter evaluation time cost
            filter_time = Filter.get_filter(var).time
            
            # Calculate score: (entropy * confidence_delta) / time
            # Higher score = better variable to evaluate next
            if conf_delta > 0:
                scores[var] = (entropy * conf_delta) / filter_time
                print(f"  {var}: entropy={entropy:.4f}, max_delta={conf_delta:.4f}, time={filter_time:.4f}, score={scores[var]:.6f}")
            else:
                print(f"  {var}: max_delta={conf_delta:.4f}, not relevant")
        
        if not scores:  # No variables left to evaluate or none that matter
            node["status"] = "complete"
            node["label"] = f"No relevant vars left\nConf: {current_confidence:.3f}"
            return node
            
        next_var = max(scores, key=scores.get)
        print(f"Node {path}: Selected {next_var} as best next variable (score={scores[next_var]:.6f})")
        
        node["next_var"] = next_var
        node["label"] = f"Evaluate: {next_var}\nConf: {current_confidence:.3f}"
        
        # Create child nodes for True and False outcomes
        true_assignment = assignment.copy()
        true_assignment[next_var] = True
        true_formula = propagate_dnf(current_formula, next_var, True)
        
        false_assignment = assignment.copy()
        false_assignment[next_var] = False
        false_formula = propagate_dnf(current_formula, next_var, False)
        
        # Recursively expand children
        true_child = expand_node(true_formula, true_assignment, depth + 1, f"{path}_T")
        false_child = expand_node(false_formula, false_assignment, depth + 1, f"{path}_F")
        
        # Add children to current node
        node["children"].append(true_child)
        node["children"].append(false_child)
        
        return node
    
    # Start the recursive expansion from the root
    print("\nBuilding evaluation tree:")
    root = expand_node(formula, {}, 0, "root")
    root["label"] = f"Start\nConf: {root['confidence']:.3f}"
    
    return root

def calculate_tree_positions(tree):
    """
    Calculate positions for all nodes in the tree.
    
    Uses a post-order traversal to calculate widths and positions.
    """
    # First, calculate node widths for subtrees
    def calculate_widths(node, min_width=1.0):
        if not node["children"]:
            node["width"] = min_width
            return min_width
        
        total_child_width = 0
        for child in node["children"]:
            total_child_width += calculate_widths(child, min_width)
        
        node["width"] = max(min_width, total_child_width)
        return node["width"]
    
    # Calculate horizontal positions using widths
    def calculate_x_positions(node, left_edge=0):
        if not node["children"]:
            node["x"] = left_edge + node["width"] / 2
            return
        
        current_left = left_edge
        for child in node["children"]:
            calculate_x_positions(child, current_left)
            current_left += child["width"]
        
        # Position this node centered above its children
        first_child = node["children"][0]
        last_child = node["children"][-1]
        node["x"] = (first_child["x"] + last_child["x"]) / 2
    
    # Assign y positions based on depth
    def calculate_y_positions(node, depth=0, y_spacing=1.0):
        node["y"] = -depth * y_spacing
        for child in node["children"]:
            calculate_y_positions(child, depth + 1, y_spacing)
    
    # Execute the calculations
    calculate_widths(tree)
    calculate_x_positions(tree)
    calculate_y_positions(tree)
    
    return tree

def plot_tree_improved(tree, title="DNF Formula Evaluation Tree"):
    """
    Visualize the evaluation tree using matplotlib with improved layout and labels.
    
    Parameters:
    - tree: tree structure from build_evaluation_tree with positions calculated
    - title: title of the plot
    """
    # Calculate positions for tree nodes
    tree = calculate_tree_positions(tree)
    
    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Define colors for different node statuses
    colors = {
        "true": "lightgreen",
        "false": "salmon",
        "internal": "lightblue",
        "complete": "lightgray",
        "max_depth": "wheat"
    }
    
    # Collect all nodes in a flat structure for easier plotting
    nodes = []
    def collect_nodes(node):
        nodes.append(node)
        for child in node["children"]:
            collect_nodes(child)
    
    collect_nodes(tree)
    
    # Calculate plot boundaries
    min_x = min(node["x"] for node in nodes) - 0.5
    max_x = max(node["x"] for node in nodes) + 0.5
    min_y = min(node["y"] for node in nodes) - 0.5
    max_y = 0.5  # Root node is at y=0
    
    # Draw edges first (so they appear behind nodes)
    for node in nodes:
        for child in node.get("children", []):
            # Draw edge
            ax.plot([node["x"], child["x"]], [node["y"], child["y"]], "k-", lw=1.5, zorder=1)
            
            # Add edge label (True/False)
            mid_x = (node["x"] + child["x"]) / 2
            mid_y = (node["y"] + child["y"]) / 2
            
            # Determine if this is a True or False branch
            child_id = child["id"]
            is_true_branch = child_id.endswith("_T")
            
            # Add evaluated variable result to the edge label
            if "next_var" in node:
                var_name = node["next_var"]
                result = "True" if is_true_branch else "False"
                label_text = f"{var_name} = {result}"
            else:
                label_text = "True" if is_true_branch else "False"
            
            # Add small offset for the label
            dx = child["x"] - node["x"]
            dy = child["y"] - node["y"]
            angle = np.arctan2(dy, dx)
            offset_x = 0.1 * np.sin(angle)
            offset_y = 0.1 * np.cos(angle)
            
            # Draw edge label with clearer background
            ax.text(mid_x + offset_x, mid_y + offset_y, label_text, fontsize=9, ha="center", va="center",
                   bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9, boxstyle="round,pad=0.2"),
                   zorder=3)
    
    # Draw nodes
    for node in nodes:
        x, y = node["x"], node["y"]
        color = colors.get(node["status"], "white")
        
        # Draw node circle
        circle = plt.Circle((x, y), 0.2, color=color, ec="black", lw=1.5, zorder=2)
        ax.add_patch(circle)
        
        # Prepare label text with assignment information
        label = node.get("label", "")
        
        # Add assignment information to the label
        if node["assignment"]:
            assigned_vars = []
            for var, value in node["assignment"].items():
                assigned_vars.append(f"{var}={value}")
            if assigned_vars:
                assignment_text = ", ".join(assigned_vars)
                label += f"\nAssigned: {assignment_text}"
        
        # Add formula state to debug
        terms_left = len(node.get("formula", []))
        if "formula" in node:
            label += f"\nTerms: {terms_left}"
        
        # Add label with multi-line support
        lines = label.split("\n")
        y_offset = 0.25  # Start below the node
        for i, line in enumerate(lines):
            ax.text(x, y - y_offset - i*0.15, line, ha="center", va="center", fontsize=9, 
                   bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.1"),
                   zorder=4)
    
    # Set up the plot
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker="o", color="w", label=status.capitalize(),
                               markerfacecolor=color, markersize=10) 
                    for status, color in colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Set limits with some padding
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    plt.tight_layout()
    return plt

def formula_tree_example():
    """
    Generate and visualize the evaluation tree for the sample DNF formula.
    """
    # Define filters
    Filter.add_filter("cf", "cloud free", 0.04, {"pass": 0.75, "fail": 0.15})
    Filter.add_filter("fl", "flooded", 0.04, {"pass": 0.25, "fail": 0.15})
    Filter.add_filter("id", "infrastructure damage", 0.04, {"pass": 0.2, "fail": 0.15})
    Filter.add_filter("cd", "vehicle detected", 0.04, {"pass": 0.4, "fail": 0.15})

    # Define formula: (cf AND fl AND id) OR (cf AND cd)
    formula = [
        [("cf", True), ("fl", True), ("id", True)],
        [("cf", True), ("cd", True)]
    ]

    # Set thresholds
    upper_threshold = 0.85
    lower_threshold = 0.02

    # Print formula structure
    print("\nFormula: (cf AND fl AND id) OR (cf AND cd)")
    print("Formula structure:", formula)
    print("Lower threshold:", lower_threshold)
    print("Upper threshold:", upper_threshold)

    # Build the evaluation tree
    tree = build_evaluation_tree(
        formula, 
        variables=["cf", "fl", "id", "cd"], 
        lower_threshold=lower_threshold, 
        upper_threshold=upper_threshold,
        max_depth=4  # Allow deeper tree to see more evaluations
    )

    # Visualize the tree with improved layout
    plt = plot_tree_improved(
        tree, 
        title=f"DNF Formula Evaluation Tree (α={lower_threshold}, β={upper_threshold})"
    )
    
    # Save the figure
    plt.savefig("dnf_evaluation_tree.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTree visualization complete. Saved as 'dnf_evaluation_tree.png'")

# Run the example
if __name__ == "__main__":

    Filter.add_filter("cf", "cloud free", 0.04, {"pass": 0.75, "fail": 0.15})
    Filter.add_filter("fl", "flooded", 0.04, {"pass": 0.25, "fail": 0.15})
    Filter.add_filter("id", "infrastructure damage", 0.04, {"pass": 0.2, "fail": 0.15})
    Filter.add_filter("cd", "vehicle detected", 0.04, {"pass": 0.4, "fail": 0.15})

    # Define formula: (cf AND fl AND id) OR (cf AND cd)
    formula = [
        [("cf", True), ("fl", True), ("id", True)],
        [("cf", True), ("cd", True)]
    ]

    variables = ["cf", "fl", "id", "cd"]

    # Set thresholds
    upper_threshold = 0.85
    lower_threshold = 0.02
    
    assignment, total_time, final_confidence = evaluate_formula_dnf(formula, variables)
    print(f"Final assignment: {assignment}, Total time: {total_time:.2f}, Final confidence: {final_confidence:.3f}")

    # formula_tree_example()