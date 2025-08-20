import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    # Get the target column (last column)
    target_column = data[:, -1]
    
    # Get unique classes and their counts
    unique_classes, counts = np.unique(target_column, return_counts=True)
    
    # Calculate total number of samples
    total_samples = len(target_column)
    
    # Calculate entropy
    entropy = 0.0
    for count in counts:
        if count > 0:  # Avoid log2(0)
            probability = count / total_samples
            entropy -= probability * np.log2(probability)
    
    return entropy

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    """
    # Get the attribute column
    attribute_column = data[:, attribute]
    total_samples = len(data)
    
    # Get unique values in the attribute
    unique_values = np.unique(attribute_column)
    
    avg_info = 0.0
    
    # For each unique value in the attribute
    for value in unique_values:
        # Create subset of data with this attribute value
        subset_mask = attribute_column == value
        subset = data[subset_mask]
        
        # Calculate weight (proportion of samples with this value)
        weight = len(subset) / total_samples
        
        # Calculate entropy of this subset
        subset_entropy = get_entropy_of_dataset(subset)
        
        # Add weighted entropy to average information
        avg_info += weight * subset_entropy
    
    return avg_info

def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    """
    # Calculate dataset entropy
    dataset_entropy = get_entropy_of_dataset(data)
    
    # Calculate average information of the attribute
    avg_info = get_avg_info_of_attribute(data, attribute)
    
    # Calculate information gain
    information_gain = dataset_entropy - avg_info
    
    # Round to 4 decimal places
    return round(information_gain, 4)

def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    """
    # Get number of attributes (excluding target variable)
    num_attributes = data.shape[1] - 1
    
    # Dictionary to store information gains
    information_gains = {}
    
    # Calculate information gain for each attribute
    for attribute_index in range(num_attributes):
        gain = get_information_gain(data, attribute_index)
        information_gains[attribute_index] = gain
    
    # Find attribute with maximum information gain
    selected_attribute = max(information_gains, key=information_gains.get)
    
    return (information_gains, selected_attribute)

