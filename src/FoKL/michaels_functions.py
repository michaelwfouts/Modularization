import numpy as np
import itertools

def perms(x):
    """
    Generates all unique permutations of the elements in a 1D array.

    This function is a Python equivalent of MATLAB's 'perms' function,
    but it also ensures that only unique permutations are returned.

    Args:
        x (np.ndarray): A 1D NumPy array whose elements will be permuted.

    Returns:
        np.ndarray: A 2D NumPy array where each row is a unique permutation
                    of the input array 'x'. The rows are reversed compared
                    to the default itertools.permutations order.
    """
    # Generate all permutations as tuples
    all_permutations = list(itertools.permutations(x))

    # Stack them vertically into a NumPy array
    # np.unique with axis=0 removes duplicate rows
    # [::-1] reverses the order of the rows
    unique_permutations = np.unique(np.vstack(all_permutations), axis=0)[::-1]
    return unique_permutations

def _get_integer_partitions_fixed_parts(n, k):
    """
    Recursively generates all unique partitions of integer n into k non-negative parts.
    The order of parts within a partition does not matter (e.g., [1,2] is same as [2,1]).
    Returns partitions as lists of integers, sorted in descending order to ensure uniqueness
    of the base partition.

    Args:
        n (int): The integer to partition.
        k (int): The fixed number of parts.

    Returns:
        list of list of int: A list where each element is a unique partition
                              of n into k parts, with parts sorted descending.
    """
    if k == 0:
        return [[]] if n == 0 else []
    if k == 1:
        return [[n]]

    partitions = set()
    # Iterate through possible values for the first part
    for i in range(n + 1):
        # Recursively find partitions for the remaining sum (n-i) into (k-1) parts
        for p in _get_integer_partitions_fixed_parts(n - i, k - 1):
            # Form the full partition, sort it descending to ensure uniqueness of the set
            # (e.g., (1,2,0) and (2,1,0) both become (2,1,0) before adding to set)
            partitions.add(tuple(sorted([i] + p, reverse=True)))
    return [list(p) for p in partitions]


def generate_interaction_matrix(m, ind, relats):
    """
    Generates and filters combinations of integer exponents for polynomial
    regression interaction terms, forming an interaction matrix.

    This function generates all possible unique combinations of 'm' non-negative
    integer exponents that sum up to 'ind', and then filters them based on
    predefined relationships (relats).

    Args:
        m (int): The number of input variables (e.g., features in a model).
                 This determines the length of the exponent vectors.
        ind (int): The target sum for the exponents in each combination.
                   This represents the 'degree' or 'order' of interaction
                   being considered (e.g., ind=2 for quadratic terms or
                   two-way interactions).
        relats (np.ndarray): A 2D NumPy array containing 'relation' patterns.
                             Each row is a binary vector (1s and 0s) that
                             represents a forbidden or redundant pattern of
                             active variables. If an interaction term's
                             active variable pattern matches a row in 'relats',
                             that term is filtered out. If 'relats' is empty,
                             no filtering is performed.

    Returns:
        np.ndarray: A 2D NumPy array (the 'interaction matrix' for this
                    particular 'ind' value). Each row represents a unique
                    and allowed combination of exponents for the 'm' input
                    variables that sum up to 'ind'. Returns an empty array
                    if no valid combinations are found after filtering.
    """
    # 1. Generate all unique base partitions of 'ind' into 'm' parts.
    # Each partition is a unique set of exponents that sum to 'ind'.
    # Example: for ind=4, m=3, this might return [[4,0,0], [3,1,0], [2,2,0], [2,1,1]]
    base_partitions = _get_integer_partitions_fixed_parts(ind, m)

    all_permutations_combined = []
    # 2. For each base partition, generate all its unique permutations.
    # These permutations represent all possible ways to assign the exponents
    # from that partition to the 'm' input variables.
    for partition_vec_list in base_partitions:
        partition_vec_np = np.array(partition_vec_list, dtype=int)
        # Get unique permutations for this specific partition
        current_permutations = perms(partition_vec_np)
        all_permutations_combined.append(current_permutations)

    if not all_permutations_combined:
        # If no partitions were generated, return an empty array of the correct shape
        return np.array([], dtype=int).reshape(0, m)

    # Combine all permutations and ensure overall uniqueness across all partitions
    vecs = np.unique(np.vstack(all_permutations_combined), axis=0)

    # Determine dimensions of vecs (still needed for subsequent loops)
    mvec, nvec = np.shape(vecs) # nvec will always be 'm' here

    kill_indices = [] # List to store indices of vectors to be removed

    # 3. Filter permutations based on 'relats'
    # This step removes terms that match predefined 'forbidden' patterns.
    num_relations = relats.shape[0] # Inferred number of relations
    if num_relations != 0: # Only filter if there are relation patterns
        for j in range(mvec): # Iterate through each potential interaction term
            # Create a binary 'testvec': 1 where original element is non-zero, 0 otherwise
            # This identifies which input variables are 'active' in this term.
            testvec = np.divide(vecs[j, :], vecs[j, :], where=vecs[j, :] != 0, out=np.zeros_like(vecs[j, :], dtype=float))
            testvec[np.isnan(testvec)] = 0 # Replace NaN (from 0/0) with 0
            testvec = testvec.astype(int) # Ensure it's integer type

            for k in range(num_relations): # Iterate through each relation pattern
                # Check if the current term's active variable pattern matches a relation
                if np.sum(testvec == relats[k, :]) == m:
                    kill_indices.append(j) # Mark this term for removal
                    break # No need to check other relations for this term

        # Create a new array 'nuvecs' containing only the allowed terms
        # This handles the case where no vectors are killed or all are killed.
        if len(kill_indices) == mvec: # If all vectors are killed
            nuvecs = np.array([], dtype=int).reshape(0, m) # Return empty array with correct shape
        else:
            # Get a boolean mask for rows to keep
            keep_mask = np.ones(mvec, dtype=bool)
            keep_mask[kill_indices] = False
            nuvecs = vecs[keep_mask]

        vecs = nuvecs # Update 'vecs' to contain only the filtered terms

    # 4. Return the processed interaction terms
    return vecs.astype(int)