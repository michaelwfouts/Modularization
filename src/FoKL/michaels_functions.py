import numpy as np
import itertools
import math
from numpy import linalg as LA
from scipy.linalg import eigh
import numpy as np
import itertools
import sys
from scipy.stats import entropy

"""
INTERACTION MATRIX GENERATION FOR BASIS FUNCTION SELECTION

This module generates interaction matrices that specify which basis functions to include
in a linear combination for the regression modeling applications.

OVERVIEW:
The interaction matrix serves as a selection mechanism for basis functions in expansions
of the form: 
f(x1, x2, ..., xm) = Σ Σ c_i * φ_i(x_k) * φ_j(x_l)

Each row of the interaction matrix represents a specific basis function selection pattern,
where the values indicate which basis functions or interaction terms to activate.

WORKFLOW:
1. Generate integer partitions of 'ind' into 'm' parts (selection patterns)
2. Calculate entropy for each partition to measure complexity distribution
3. Group partitions by entropy level for systematic exploration
4. Generate all permutations of each partition pattern
5. Apply relation-based filtering to remove unwanted selection patterns
6. Sort and organize the final interaction matrix

The resulting interaction matrix provides a structured approach to basis function
selection, with entropy levels offering a natural progression for model complexity
and systematic exploration of different interaction patterns.
"""

def _get_integer_partitions_fixed_parts(n, k):
    """
    Recursively generates all unique partitions of an integer n into k non-negative parts.

    This is a core function for generating the "building blocks" of the exponent vectors.
    It finds all the different ways to break down a total degree 'n' into 'k' parts.
    For example, for n=3 and k=3, it finds the unique partitions [3, 0, 0], [2, 1, 0], and [1, 1, 1].
    The order of parts does not matter at this stage (e.g., [1, 2, 0] is considered the
    same as [2, 1, 0]), which is why the parts are sorted to ensure that only
    unique base partitions are returned.

    Args:
        n (int): The integer to partition (the total degree of the polynomial term).
        k (int): The fixed number of parts (the number of input variables).

    Returns:
        list of list of int: A list where each element is a unique partition
                             of n into k parts. The parts within each partition
                             are sorted in descending order.
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

def perms(x):
    """
    Generates all unique permutations of the elements in a 1D array.

    This function is a Python equivalent of MATLAB's 'perms' function,
    but it also ensures that only unique permutations are returned. It's
    used to create all possible vectors for a given base partition.
    For example, if the base partition is [2, 1, 0], this function will
    generate all unique orderings like [2, 1, 0], [2, 0, 1], [1, 2, 0], etc.
    Each of these permutations represents a unique polynomial term.

    Args:
        x (np.ndarray): A 1D NumPy array (a partition) whose elements will be permuted.

    Returns:
        np.ndarray: A 2D NumPy array where each row is a unique permutation
                    of the input array 'x'. The rows are sorted in a specific
                    way that is reversed compared to the default itertools.permutations order.
    """
    # Generate all permutations as tuples. If x is [1, 1, 0], it will generate
    # (1, 1, 0), (1, 0, 1), (1, 1, 0), (1, 0, 1), etc.
    all_permutations = list(itertools.permutations(x))

    # Stack them vertically into a NumPy array and use np.unique with axis=0
    # to find and remove any duplicate rows. The [::-1] then reverses the
    # order of the unique rows.
    unique_permutations = np.unique(np.vstack(all_permutations), axis=0)[::-1]

    return unique_permutations

def calculate_entropy(partition):
    """
    Calculates the Shannon entropy of a partition to measure how evenly
    distributed the exponents are across variables.

    Entropy is a concept from information theory. In this context, it
    serves as a measure of "balance" or "interactivity."
    - A partition like [1, 1, 1, 1] would have high entropy because the total
      degree is spread equally among all variables. This represents a complex,
      multi-variable interaction.
    - A partition like [4, 0, 0, 0] would have low entropy because the entire
      degree is concentrated on a single variable, representing a simple,
      non-interactive term.

    Args:
        partition (list or np.ndarray): A partition/exponent vector (e.g., [2, 1, 0]).

    Returns:
        float: The entropy value (higher = more evenly distributed exponents).
    """
    # Convert to a numpy array and filter out any zero values, as they
    # don't contribute to the "information content" of the term's interaction.

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
        ind (int): Target sum for polynomial degree
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
    Provides information about the different entropy levels available for a
    given number of variables `m` and total degree `ind`.

    This is a helper function designed to help the user understand the
    structure of the generated terms. It lists the distinct entropy values
    that exist for the given parameters and provides an example partition
    for each level. This allows the user to choose which entropy level to
    generate an interaction matrix for, effectively controlling the complexity
    of the terms they are interested in.

    Args:
        m (int): The number of input variables.
        ind (int): The target sum for exponents (the total degree).

    Returns:
        list: A list of tuples. Each tuple contains:
              (entropy_level_index, entropy_value, example_partitions)
    """
    # First, get all the base partitions for the given parameters.
    base_partitions = _get_integer_partitions_fixed_parts(ind, m)

    entropy_info = {}
    for partition in base_partitions:
        # Calculate and round the entropy to group similar floating-point values.
        ent = round(calculate_entropy(partition), 6)
        if ent not in entropy_info:
            entropy_info[ent] = []
        entropy_info[ent].append(partition)

    # Sort the entropy values from highest to lowest.
    sorted_entropies = sorted(entropy_info.keys(), reverse=True)

    result = []
    # Create the final output list with the level index, value, and partitions.
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

def build_design_matrix(inputs, phis, Xin, discmtx, phind, xsm, kernel='Cubic Splines'):
    """
    Build the design matrix X for the regression model using a specified kernel.

    This function constructs the design matrix, where each column corresponds to a basis function
    from a specified kernel (e.g., Cubic Splines or Bernoulli Polynomials). The values in the
    matrix are the evaluations of these basis functions at the given input points. The function
    handles both single and multi-dimensional interactions as defined by the `discmtx`
    (interaction matrix).

    Parameters:
    -----------
    inputs : numpy array
        Normalized inputs (parameters and model inputs) with shape `(minp, ninp)`.
    phis : list or numpy array
        A data structure containing the coefficients for the basis functions. The structure
        depends on the chosen `kernel`. For 'Cubic Splines', it's a nested list of coefficients
        for each segment. For 'Bernoulli Polynomials', it's a list of coefficients.
    Xin : numpy array
        A pre-defined design matrix.  If a portion of the model is already solved for, to prevent
        the calculations from being repeated, this gives the evaluation values so only 
        the new basis functions need to be evaluated. If `Xin` is empty, everything will be 
        calculated.
    discmtx : numpy array
        The interaction matrix that defines which basis functions (from `phis`) are combined
        to form the columns of the design matrix. Each row of `discmtx` corresponds to a
        basis function in the design matrix, and the values in the row indicate which
        single-variable basis functions are to be multiplied together. A value of 0 indicates
        no interaction for that input variable.
    phind : numpy array
        Phase indices that map each input point to a specific segment of the basis function.
        Used primarily for 'Cubic Splines'.
    xsm : numpy array
        Scaled input values, which are used to evaluate the basis functions.
    kernel : str, optional
        The type of basis function to use. Supported options are 'Cubic Splines' and
        'Bernoulli Polynomials'. Defaults to 'Cubic Splines'.

    Returns:
    --------
    X : numpy array
        The constructed design matrix with shape `(minp, mmtx + 1)`.
    mmtx : int
        The number of basis functions (columns) generated from the interaction matrix.
        Note that the final design matrix `X` will have `mmtx + 1` columns if `Xin`
        is an empty array, accounting for the intercept.
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

    # The loop constructs the columns of X corresponding to the basis functions defined by discmtx.
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
        Normalized inputs
    data : numpy array
        Experimental results (y values)
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
    # Build design matrix based on the current interaction matrix.
    X, mmtx = build_design_matrix(inputs, phis, Xin, discmtx, phind, xsm, kernel)
    
    # Perform least squares regression to find coefficients.
    # The normal equations are X.T * X * beta_hat = X.T * y.
    XtX = np.transpose(X).dot(X)
    Xty = np.transpose(X).dot(data)
    
    # Solve for beta_hat using a stable method involving eigendecomposition.
    # (X.T*X)^-1 = Q * Lambda^-1 * Q.T
    Lamb, Q = eigh(XtX)
    Lamb_inv = np.diag(1 / Lamb)
    betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
    
    # Calculate BIC
    n = len(data)
    # Estimate the variance of the residuals (sigma^2_lik).
    siglik = np.var(data - np.matmul(X, betahat))
    # Calculate the log-likelihood (part of the BIC formula).
    lik = -(n / 2) * np.log(siglik) - (n - 1) / 2
    # The BIC formula is BIC = k * ln(n) - 2 * ln(L_hat).
    # Here, k is the number of parameters, which is (mmtx + 1).
    ev = (mmtx + 1) * np.log(n) - 2 * np.max(lik)
    
    return ev, betahat, X

def gibbs_sampling(X, data, a, b, atau, btau, draws, sigsqd, tausqd, dtd):
    """
    Perform Gibbs sampling for Bayesian linear regression with hierarchical priors.

    This function implements a Gibbs sampler to draw samples from the posterior distribution
    of the regression coefficients ($\beta$), the observation error variance ($\sigma^2$), and
    the prior variance for the coefficients ($\tau^2$). The model assumes a prior
    distribution on $\beta$ that is Gaussian with variance $\tau^2 I$, and inverse-gamma
    distributions for $\sigma^2$ and $\tau^2$. This is a standard hierarchical Bayesian
    setup often used for variable selection or regularization.

    Parameters:
    -----------
    X : numpy array
        The design matrix for the regression.
    data : numpy array
        The observed data or response variable.
    a, b : float
        Parameters of the inverse-gamma prior for the observation error variance $\sigma^2$,
        i.e., $\sigma^2 \sim \text{InvGamma}(a, b)$.
    atau, btau : float
        Parameters of the inverse-gamma prior for the prior variance of coefficients $\tau^2$,
        i.e., $\tau^2 \sim \text{InvGamma}(a_{tau}, b_{tau})$.
    draws : int
        The number of iterations (samples) to draw from the posterior.
    sigsqd : float
        Initial value for the observation error variance $\sigma^2$.
    tausqd : float
        Initial value for the prior variance of the coefficients $\tau^2$.
    dtd : float
        The dot product of the data vector with itself, i.e., $data^T data$. This is pre-calculated
        to save computation inside the loop.

    Returns:
    --------
    betas : numpy array
        An array of shape `(draws, mmtx + 1)` containing the sampled regression coefficients
        for each iteration.
    sigs : numpy array
        An array of shape `(draws, 1)` containing the sampled observation error variances $\sigma^2$.
    taus : numpy array
        An array of shape `(draws, 1)` containing the sampled prior variances $\tau^2$.
    """
    minp, mmtx_plus_1 = np.shape(X)
    mmtx = mmtx_plus_1 - 1
    
    # Pre-compute matrices that are constant throughout the sampling.
    XtX = np.transpose(X).dot(X)
    Xty = np.transpose(X).dot(data)
    Lamb, Q = eigh(XtX)
    n = len(data)

    # Pre-calculate the updated shape parameters for the inverse-gamma posteriors.
    # The posterior for sigsqd is InvGamma(astar, bstar).
    # The posterior for tausqd is InvGamma(atau_star, btau_star).
    astar = a + 1 + n / 2 + (mmtx + 1) / 2
    atau_star = atau + mmtx / 2

    # Initialize storage for the sampled values.
    betas = np.zeros((draws, mmtx + 1))
    sigs = np.zeros((draws, 1))
    taus = np.zeros((draws, 1))

    for k in range(draws):
        # 1. Sample beta from its posterior distribution (a multivariate normal).
        # The posterior mean is given by mu_n = (X.T*X + (1/tausqd)*I)^-1 * X.T*y.
        # The posterior covariance is given by Sigma_n = sigsqd * (X.T*X + (1/tausqd)*I)^-1.
        # The code uses eigendecomposition for stable and efficient computation.
        Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
        Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))
        mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)

        # S is the Cholesky-like decomposition of the posterior covariance matrix.
        # Sigma_n = sigsqd * Q * Lamb_tausqd_inv * Q.T = sigsqd * S * S.T.
        S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

        # Sample beta using the reparameterization trick: beta = mu_n + sqrt(sigsqd) * S * Z,
        # where Z is a vector of standard normal random variables.
        vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))
        betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

        # 2. Sample sigma squared from its posterior (an inverse-gamma distribution).
        # The updated scale parameter `bstar` depends on the most recently sampled beta.
        bstar = b + 0.5 * (betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) +
                           dtd + betas[k][:].dot(np.transpose([betas[k][:]])) / tausqd)

        if bstar < 0:
            sigsqd = math.nan
        else:
            sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

        sigs[k] = sigsqd

        # 3. Sample tau squared from its posterior (an inverse-gamma distribution).
        # The updated scale parameter `btau_star` depends on the most recently sampled beta
        # and sigma squared.
        btau_star = (1/(2*sigsqd)) * (betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau
        tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
        taus[k] = tausqd

    return betas, sigs, taus