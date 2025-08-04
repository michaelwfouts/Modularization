import numpy as np
import itertools
import math
from numpy import linalg as LA
from scipy.linalg import eigh

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

import numpy as np
import itertools
import sys
from scipy.stats import entropy

def calculate_entropy(partition):
    """
    Calculate the entropy of a partition to measure how evenly distributed
    the exponents are across variables.
    
    Args:
        partition (list or np.ndarray): A partition/exponent vector
        
    Returns:
        float: Entropy value (higher = more evenly distributed)
    """
    # Convert to numpy array and filter out zeros for entropy calculation
    partition = np.array(partition)
    non_zero_values = partition[partition > 0]
    
    if len(non_zero_values) == 0:
        return 0.0
    
    # Normalize to create a probability distribution
    probabilities = non_zero_values / np.sum(non_zero_values)
    
    # Calculate Shannon entropy
    return entropy(probabilities, base=2)

def generate_interaction_matrix_by_entropy(m, ind, relats, entropy_level=None):
    """
    Generates interaction matrix with entropy-based prioritization.
    
    Args:
        m (int): Number of input variables
        ind (int): Target sum for exponents (degree/order)
        relats (np.ndarray): Relation patterns for filtering
        entropy_level (int, optional): Specific entropy level to return.
                                     If None, returns all levels sorted by entropy.
    
    Returns:
        If entropy_level is None:
            dict: {entropy_level: interaction_matrix} sorted by descending entropy
        If entropy_level is specified:
            np.ndarray: Interaction matrix for that specific entropy level
    """
    # Generate base partitions as before
    base_partitions = _get_integer_partitions_fixed_parts(ind, m)
    
    if not base_partitions:
        if entropy_level is None:
            return {}
        else:
            return np.array([], dtype=int).reshape(0, m)
    
    # Group partitions by entropy level
    entropy_groups = {}
    
    for partition in base_partitions:
        ent = calculate_entropy(partition)
        # Round entropy to avoid floating point precision issues
        ent_rounded = round(ent, 6)
        
        if ent_rounded not in entropy_groups:
            entropy_groups[ent_rounded] = []
        entropy_groups[ent_rounded].append(partition)
    
    # Sort entropy levels in descending order (highest entropy first)
    sorted_entropy_levels = sorted(entropy_groups.keys(), reverse=True)
    
    # If specific entropy level requested
    if entropy_level is not None:
        if entropy_level >= len(sorted_entropy_levels):
            return np.array([], dtype=int).reshape(0, m)
        
        target_entropy = sorted_entropy_levels[entropy_level]
        partitions_to_process = entropy_groups[target_entropy]
    else:
        # Process all partitions, but we'll organize by entropy level
        partitions_to_process = base_partitions
    
    # Generate permutations for the selected partitions
    all_permutations_combined = []
    for partition_vec_list in partitions_to_process:
        partition_vec_np = np.array(partition_vec_list, dtype=int)
        current_permutations = perms(partition_vec_np)
        all_permutations_combined.append(current_permutations)
    
    if not all_permutations_combined:
        if entropy_level is None:
            return {}
        else:
            return np.array([], dtype=int).reshape(0, m)
    
    # Combine and get unique permutations
    vecs = np.unique(np.vstack(all_permutations_combined), axis=0)
    
    # Sort by number of active variables (as before)
    num_active_variables = np.count_nonzero(vecs, axis=1)
    sort_indices = np.argsort(num_active_variables)
    vecs = vecs[sort_indices]
    
    # Apply filtering based on relats (same as your original code)
    mvec, nvec = np.shape(vecs)
    kill_indices = []
    
    num_relations = relats.shape[0]
    if num_relations != 0:
        for j in range(mvec):
            testvec = np.divide(vecs[j, :], vecs[j, :], where=vecs[j, :] != 0, 
                              out=np.zeros_like(vecs[j, :], dtype=float))
            testvec[np.isnan(testvec)] = 0
            testvec = testvec.astype(int)
            
            for k in range(num_relations):
                if np.sum(testvec == relats[k, :]) == m:
                    kill_indices.append(j)
                    break
        
        if len(kill_indices) == mvec:
            filtered_vecs = np.array([], dtype=int).reshape(0, m)
        else:
            keep_mask = np.ones(mvec, dtype=bool)
            keep_mask[kill_indices] = False
            filtered_vecs = vecs[keep_mask]
    else:
        filtered_vecs = vecs
    
    if entropy_level is None:
        # Return all entropy levels organized in a dictionary
        result = {}
        for i, ent_val in enumerate(sorted_entropy_levels):
            # Process each entropy level separately
            level_result = generate_interaction_matrix_by_entropy(m, ind, relats, i)
            if level_result.size > 0:
                result[i] = level_result
        return result
    else:
        return filtered_vecs.astype(int)

# Convenience function to get all entropy levels for a given degree
def get_entropy_levels(m, ind):
    """
    Get information about available entropy levels for given parameters.
    
    Args:
        m (int): Number of input variables
        ind (int): Target sum for exponents
        
    Returns:
        list: List of tuples (entropy_level, entropy_value, example_partitions)
    """
    base_partitions = _get_integer_partitions_fixed_parts(ind, m)
    
    entropy_info = {}
    for partition in base_partitions:
        ent = round(calculate_entropy(partition), 6)
        if ent not in entropy_info:
            entropy_info[ent] = []
        entropy_info[ent].append(partition)
    
    sorted_entropies = sorted(entropy_info.keys(), reverse=True)
    
    result = []
    for i, ent_val in enumerate(sorted_entropies):
        result.append((i, ent_val, entropy_info[ent_val]))
    
    return result

import numpy as np
from scipy.linalg import eigh
import scipy.linalg as LA
import math

def evaluate_basis_cubic_splines(coeffs, x):
    """Evaluate cubic spline basis function"""
    return coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3

def evaluate_basis_bernoulli(coeffs, x):
    """Evaluate Bernoulli polynomial basis function"""
    # Add your Bernoulli polynomial evaluation logic here
    # This is a placeholder - replace with your actual implementation
    return sum(c * (x**i) for i, c in enumerate(coeffs))

def gibbs(inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws, phind, xsm, sigsqd, tausqd, dtd, kernel='Cubic Splines'):
    """
    'inputs' is the set of normalized inputs -- both parameters and model
    inputs -- with columns corresponding to inputs and rows the different
    experimental designs. (numpy array)

    'data' are the experimental results: column vector, with entries
    corresponding to rows of 'inputs'

    'phis' are a data structure with the coefficients for the basis
    functions

    'discmtx' is the interaction matrix for the bss-anova function -- rows
    are terms in the function and columns are inputs (cols should line up
    with cols in 'inputs'

    'a' and 'b' are the parameters of the ig distribution for the
    observation error variance of the data

    'atau' and 'btau' are the parameters of the ig distribution for the 'tau
    squared' parameter: the variance of the beta priors

    'draws' is the total number of draws

    Additional Constants (to avoid repeat calculations found in later development):
        - phind
        - xsm
        - sigsqd
        - tausqd
        - dtd
    """
    
    # Define available kernels
    kernels = ['Cubic Splines', 'Bernoulli Polynomials']
    
    # Define basis function evaluation
    def evaluate_basis(coeffs, x):
        if kernel == kernels[0]:  # 'Cubic Splines'
            return evaluate_basis_cubic_splines(coeffs, x)
        elif kernel == kernels[1]:  # 'Bernoulli Polynomials'
            return evaluate_basis_bernoulli(coeffs, x)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    # Rest of your function code remains the same, just remove all `self.` references
    minp, ninp = np.shape(inputs)
    phi_vec = []
    if np.shape(discmtx) == ():
        mmtx = 1
    else:
        mmtx, null = np.shape(discmtx)

    if np.size(Xin) == 0:
        Xin = np.ones((minp, 1))
        mxin, nxin = np.shape(Xin)
    else:
        mxin, nxin = np.shape(Xin)
    if mmtx - nxin < 0:
        X = Xin
    else:
        X = np.append(Xin, np.zeros((minp, mmtx - nxin)), axis=1)

    for i in range(minp):
        for j in range(nxin, mmtx + 1):
            null, nxin2 = np.shape(X)
            if j == nxin2:
                X = np.append(X, np.zeros((minp, 1)), axis=1)

            phi = 1

            for k in range(ninp):
                if np.shape(discmtx) == ():
                    num = discmtx
                else:
                    num = discmtx[j - 1][k]

                if num != 0:
                    nid = int(num - 1)

                    if kernel == kernels[0]:  # 'Cubic Splines'
                        coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]
                    elif kernel == kernels[1]:  # 'Bernoulli Polynomials'
                        coeffs = phis[nid]
                    
                    phi = phi * evaluate_basis(coeffs, xsm[i, k])

            X[i][j] = phi

    # ... rest of your existing code (XtX calculation onwards) remains unchanged
    XtX = np.transpose(X).dot(X)
    Xty = np.transpose(X).dot(data)
    
    Lamb, Q = eigh(XtX)
    Lamb_inv = np.diag(1 / Lamb)
    
    betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
    squerr = LA.norm(data - X.dot(betahat)) ** 2
    
    n = len(data)
    astar = a + 1 + n / 2 + (mmtx + 1) / 2
    atau_star = atau + mmtx / 2
    
    betas = np.zeros((draws, mmtx + 1))
    sigs = np.zeros((draws, 1))
    taus = np.zeros((draws, 1))
    lik = np.zeros((draws, 1))
    
    for k in range(draws):
        Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
        Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))
        
        mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
        S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))
        
        vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))
        betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))
        
        vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))
        
        bstar = b + 0.5 * (betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) +
                           dtd + betas[k][:].dot(np.transpose([betas[k][:]])) / tausqd)
        
        if bstar < 0:
            sigsqd = math.nan
        else:
            sigsqd = 1 / np.random.gamma(astar, 1 / bstar)
        
        sigs[k] = sigsqd
        
        btau_star = (1/(2*sigsqd)) * (betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau
        
        tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
        taus[k] = tausqd
    
    siglik = np.var(data - np.matmul(X, betahat))
    lik = -(n / 2) * np.log(siglik) - (n - 1) / 2
    ev = 3*(mmtx + 1) * np.log(n) - 2 * np.max(lik)
    
    X = X[:, 0:mmtx + 1]
    
    return betas, sigs, taus, betahat, X, ev

def build_design_matrix(inputs, phis, Xin, discmtx, phind, xsm, kernel='Cubic Splines'):
    """
    Build the design matrix X for the regression model.
    
    Returns:
    --------
    X : numpy array
        Design matrix
    mmtx : int
        Number of basis functions
    """
    kernels = ['Cubic Splines', 'Bernoulli Polynomials']
    
    def evaluate_basis(coeffs, x):
        if kernel == kernels[0]:
            return evaluate_basis_cubic_splines(coeffs, x)
        elif kernel == kernels[1]:
            return evaluate_basis_bernoulli(coeffs, x)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    minp, ninp = np.shape(inputs)
    
    if np.shape(discmtx) == ():
        mmtx = 1
    else:
        mmtx, null = np.shape(discmtx)

    if np.size(Xin) == 0:
        Xin = np.ones((minp, 1))
        mxin, nxin = np.shape(Xin)
    else:
        mxin, nxin = np.shape(Xin)
    
    if mmtx - nxin < 0:
        X = Xin
    else:
        X = np.append(Xin, np.zeros((minp, mmtx - nxin)), axis=1)

    for i in range(minp):
        for j in range(nxin, mmtx + 1):
            null, nxin2 = np.shape(X)
            if j == nxin2:
                X = np.append(X, np.zeros((minp, 1)), axis=1)

            phi = 1

            for k in range(ninp):
                if np.shape(discmtx) == ():
                    num = discmtx
                else:
                    num = discmtx[j - 1][k]

                if num != 0:
                    nid = int(num - 1)

                    if kernel == kernels[0]:
                        coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]
                    elif kernel == kernels[1]:
                        coeffs = phis[nid]
                    
                    phi = phi * evaluate_basis(coeffs, xsm[i, k])

            X[i][j] = phi

    X = X[:, 0:mmtx + 1]
    return X, mmtx

def calculate_bic(inputs, data, phis, Xin, discmtx, phind, xsm, kernel='Cubic Splines'):
    """
    Calculate the BIC (Bayesian Information Criterion) for model selection.
    
    Parameters:
    -----------
    inputs : numpy array
        Normalized inputs (parameters and model inputs)
    data : numpy array
        Experimental results
    phis : data structure
        Coefficients for basis functions
    Xin : numpy array
        Initial design matrix columns
    discmtx : numpy array
        Interaction matrix for bss-anova function
    phind : numpy array
        Phase indices
    xsm : numpy array
        Scaled input values
    kernel : str
        Type of basis function ('Cubic Splines' or 'Bernoulli Polynomials')
    
    Returns:
    --------
    ev : float
        BIC value (evidence)
    betahat : numpy array
        Maximum likelihood estimate of coefficients
    X : numpy array
        Design matrix
    """
    # Build design matrix
    X, mmtx = build_design_matrix(inputs, phis, Xin, discmtx, phind, xsm, kernel)
    
    # Calculate coefficients using least squares
    XtX = np.transpose(X).dot(X)
    Xty = np.transpose(X).dot(data)
    
    Lamb, Q = eigh(XtX)
    Lamb_inv = np.diag(1 / Lamb)
    
    betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
    
    # Calculate BIC
    n = len(data)
    siglik = np.var(data - np.matmul(X, betahat))
    lik = -(n / 2) * np.log(siglik) - (n - 1) / 2
    ev = 3*(mmtx + 1) * np.log(n) - 2 * np.max(lik)
    
    return ev, betahat, X

def gibbs_sampling(X, data, a, b, atau, btau, draws, sigsqd, tausqd, dtd):
    """
    Perform Gibbs sampling for Bayesian regression.
    
    Parameters:
    -----------
    X : numpy array
        Design matrix (from calculate_bic or build_design_matrix)
    data : numpy array
        Experimental results
    betahat : numpy array
        Initial coefficient estimates (from calculate_bic)
    a, b : float
        Parameters of inverse gamma distribution for observation error variance
    atau, btau : float
        Parameters of inverse gamma distribution for tau squared
    draws : int
        Number of Gibbs sampling draws
    sigsqd : float
        Initial sigma squared value
    tausqd : float
        Initial tau squared value
    dtd : float
        Data transpose times data
    
    Returns:
    --------
    betas : numpy array
        Sampled regression coefficients
    sigs : numpy array
        Sampled sigma squared values
    taus : numpy array
        Sampled tau squared values
    """
    minp, mmtx_plus_1 = np.shape(X)
    mmtx = mmtx_plus_1 - 1
    
    # Pre-compute matrices
    XtX = np.transpose(X).dot(X)
    Xty = np.transpose(X).dot(data)
    
    Lamb, Q = eigh(XtX)
    
    n = len(data)
    astar = a + 1 + n / 2 + (mmtx + 1) / 2
    atau_star = atau + mmtx / 2

    # Initialize storage for samples
    betas = np.zeros((draws, mmtx + 1))
    sigs = np.zeros((draws, 1))
    taus = np.zeros((draws, 1))

    for k in range(draws):
        # Sample beta
        Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
        Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

        mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
        S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

        vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))
        betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

        # Sample sigma squared
        bstar = b + 0.5 * (betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) +
                           dtd + betas[k][:].dot(np.transpose([betas[k][:]])) / tausqd)

        if bstar < 0:
            sigsqd = math.nan
        else:
            sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

        sigs[k] = sigsqd

        # Sample tau squared
        btau_star = (1/(2*sigsqd)) * (betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau
        tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
        taus[k] = tausqd

    return betas, sigs, taus