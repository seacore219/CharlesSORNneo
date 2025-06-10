print("[INFO] Starting import of libraries and function definitions...")
import os
import re
import tables
print("[INFO] numpy imported successfully.")
import numpy as np
import math
import mrestimator as mre
import matplotlib
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
import os.path as op
from decimal import Decimal, ROUND_HALF_UP
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import welch
from scipy.interpolate import interp1d
from copy import deepcopy as cdc
import time
import criticality as cr
import psutil
import gc
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
gc.collect()
tables.file._open_files.close_all()
print("[INFO] all libraries imported successfully.")
##############################################################
# Define the necessary functions here
##############################################################
print("[INFO] Searching for directories with simulation results...")
## Need to find all the directories where there are valid sims
base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\GitHub\\SORN\\mu=0.06_sigma=0.05_500K+3.5M_plastic_raster\\test_single'
#pattern = r"SPmu=(\d+\.\d+)_sigma=(\d+\.\d+base_).*_raster"
#pattern = r"SPmu=(\d+\.\d+)_sigma=(\d+\.\d+)_(\d+)K.*_sigma_(\d+\.\d+)_.*raster"
#->pattern = r"SPmu=(0.08)_sigma=(0.05)_(\d+)K.*_sigma_(0.05)_.*raster"
pattern = r"202"
print("-- base_dir dir: ", base_dir)
##############################################################

# Firing activity susceptibility (variance)
def susceptibility(raster):
    num_rows, num_columns = raster.shape
    term1 = np.mean(np.mean(raster,axis=0)**2)
    term2 = np.mean(np.mean(raster,axis=0))**2
    #susceptability = num_rows * (term1 - term2)
    susceptibility = (num_rows/(num_rows-1)) * (term1 - term2)
    return susceptibility #, suscept2

##############################################################

# Firing activity per bin size
def rho(raster):
    print("Computing rho")
    rho=np.mean(np.mean(raster, axis=0))
    return rho

##############################################################

# Firing activity coefficient of variation 
def cv(raster):
    var = susceptibility(raster)
    rho=np.mean(np.mean(raster, axis=0))
    cv=math.sqrt(var)/rho
    return cv
##############################################################

# Priesemann's BR method
def calc_BR(A_t, k_max,ava_binsz,pltname):

	dt = ava_binsz*1000
	kmax = k_max

	src = mre. input_handler(A_t)
	rks = mre. coefficients(src , steps =(1, kmax) , dt=dt, dtunit='ms', method='trialseparated')

	fit1 = mre.fit(rks , fitfunc='complex')
	fit2 = mre.fit(rks , fitfunc='exp_offset')

	fig,ax1 = plt.subplots()
	ax1.plot(rks.steps, rks.coefficients, '.k', alpha = 0.2, label=r'Data')

	ax1.plot(rks.steps, mre.f_complex(rks.steps*dt, *fit1.popt), label='complex m={:.5f}'.format(fit1.mre))
	ax1.plot(rks.steps, mre.f_exponential_offset(rks.steps*dt, *fit2.popt), label='exp + offset m={:.5f}'.format(fit2.mre))

	ax1.set_xlabel(r'Time lag $\delta t$')
	ax1.set_ylabel(r'Autocorrelation $r_{\delta t}$')
	ax1.legend()
	plt.savefig(pltname)
	plt.close()

	fit_acc1 = sc.stats.pearsonr(rks.coefficients, mre.f_complex(rks.steps*dt, *fit1.popt))[0]
	fit_acc2 = sc.stats.pearsonr(rks.coefficients, mre.f_exponential_offset(rks.steps*dt, *fit2.popt))[0]

	return fit1.mre #, fit2.mre, fit_acc1, fit_acc2


##############################################################

def myround(n):
    return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))

def branchparam(TIMERASTER, lverbose=0):
    # myround = np.vectorize(lambda x: round(x))
    # Number of rows and columns
    r, c = TIMERASTER.shape
    print("Raster shape: ", TIMERASTER.shape)
    if lverbose:
        print(f'Number of rows and columns are: {r} and {c}, respectively')    # Initialize arrays
    descendants = np.zeros(r + 1)
    if lverbose:
        print(f'descendants {descendants}')
    prob = np.zeros(r)
    if lverbose:
        print(f'prob {prob}')    # Convert non-zero elements to 1
    TIMERASTER[TIMERASTER != 0] = 1
    # print(f’nonzero TIMERASTER {np.nonzero(TIMERASTER)}‘)
    # print(f’nonzero {np.where(TIMERASTER != 0)[0]}‘)    # Find frames with at least one active site and
    # no sites active in the previous frame
    sums = np.sum(TIMERASTER, axis=0)
    if lverbose:
        print(f'sums {sums}')
    if lverbose:
        print(f'sums: {sums.shape}')    
    actives = np.nonzero(sums)[0]
    if lverbose:
        print(f'actives {actives}')
    if lverbose:
        print(f'len actives {len(actives)}') 
    for i in range(1, len(actives) - 1):
        ancestors = 0
        if lverbose:
            print(f'i {i} actives[i] {actives[i]} '
                f'actives[i]-1 {actives[i]-1} '
                f'sums(actives[i]-1) {sums[actives[i]-1]}')
        if sums[actives[i] - 1] == 0:
            ancestors = sums[actives[i]]
            if lverbose:
                print(f'i {i} ancestors {ancestors}')
            num = sums[actives[i] + 1]
            if lverbose:
                print(f'i {i} num {num}')
            # num = round(num / ancestors)
            num = myround(num / ancestors)
            # num = int(np.ceil(num / ancestors))
            if lverbose:
                print(f'i {i} num {num}')
            # descendants[num + 1] += ancestors
            descendants[num] = descendants[num] + ancestors
                # descendants[num] += ancestors
            if lverbose:
                print(f'i {i} descendants {descendants[num ]}')
        # print(f’i {i} ancestors {ancestors} num {num} descendants(num+1)
        # {descendants[num+1]}‘) 
    if lverbose:
        print(f'sum ancestors: {np.sum(ancestors)}')
        print(f'sum descendants: {np.sum(descendants)}')
        print(f'descendants: {descendants}')
        print(f'num: {num}')    
    # Calculate the probability of each number of descendants
    sumd = np.sum(descendants)
    prob = descendants / sumd if sumd != 0 else np.zeros(r)  
    if lverbose:
        print(f'descendants: {descendants}')
        print(f'prob: {prob}')    
    # Calculate the expected value
    # sig = np.sum((np.arange(r + 1) - 1) * prob)
    sig = 0.0
    if lverbose:
        print(f'sig: {sig}')
    # for i in range(r + 1):
    # CHARLES -> for i in range(r):
    for i in range(r+1):
        sig = sig + ((i)*prob[i])
        if lverbose:
            print(f'i{i} prob(i){prob[i]} '
                  f'(i)*probi{(i)*prob[i]} sig{sig}')
    if lverbose:
        print(f'sig: {sig}')    
    return sig

##############################################################
# Function to find avalanches from raster data

def find_avalanches(array):
    activity_array = np.sum(array, axis=0)
    avalanches = []
    current_avalanche = []
    for activity in activity_array:
        if activity > 0:
            current_avalanche.append(activity)
        elif current_avalanche:
            avalanches.append(current_avalanche)
            current_avalanche = []
    if current_avalanche:  # Add the last avalanche if it exists
        avalanches.append(current_avalanche)
    return avalanches
#############################################################


#####################################################################

def rebinner(raster, bin_size):
    channels, timesteps = raster.shape

    # Calculating the number of new bins based on bin_size
    new_timesteps = int(np.ceil(timesteps / bin_size))

    # Preallocate the new raster matrix
    new_raster = csr_matrix((channels, new_timesteps), dtype=int)

    # Loop through each channel
    for i_channel in range(channels):
        # Extract the spike times for this channel
        spike_times = np.where(raster[i_channel, :])[0]

        # Rebin spike times
        new_spike_times = np.unique(np.ceil((spike_times + 1) / bin_size).astype(int) - 1)

        # Update the new raster matrix
        new_raster[i_channel, new_spike_times] = 1

    return new_raster


from copy import deepcopy as cdc
import time


##############################################################

def avgshapes(shapes, durations, method=None, args=()):
    # Determine sampling method
    target_indices = np.ones(len(durations), dtype=bool)
    
    if method is not None:
        if method == 'limits':
            lower_lim, upper_lim = args
            target_indices = (durations >= lower_lim) & (durations <= upper_lim)
        elif method == 'order':
            magnitude = args[0]
            if np.isscalar(magnitude):
                lower_lim = 10 ** magnitude
                upper_lim = 10 ** (magnitude + 1)
            else:
                lower_lim = 10 ** np.min(magnitude)
                upper_lim = 10 ** np.max(magnitude)
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (freqs >= lower_lim) & (freqs < upper_lim)
        elif method == 'linspace':
            lower_lim, upper_lim, n = args
            target_durations = np.round(np.linspace(lower_lim, upper_lim, n))
            target_indices = np.isin(durations, target_durations)
        elif method == 'logspace':
            x, lower_lim, upper_lim = args
            target_durations = x ** np.arange(lower_lim, upper_lim + 1)
            target_indices = np.isin(durations, target_durations)
        elif method == 'durations':
            target_durations = args[0]
            target_indices = np.isin(durations, target_durations)
        elif method == 'cutoffs':
            lower_lim, threshold = args
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (np.unique(durations) >= lower_lim) & (freqs >= threshold)

    # Compute average shapes
    sampled_shapes = [shape for shape, index in zip(shapes, target_indices) if index]
    sampled_durations = durations[target_indices]

    unique_durations = np.unique(sampled_durations)
    avg_profiles = []
    for dur in unique_durations:
        these_shapes = [shape for shape, d in zip(sampled_shapes, sampled_durations) if d == dur]
        avg_profiles.append(np.mean(these_shapes, axis=0))

    return avg_profiles

##############################################################

##############################################################
# Avalanche shape collapse

def avshapecollapse(shapes, durations, method=None, args=(),plot_flag=True, save_flag=False, filename=base_dir):
    
    if not shapes:
        print("[WARNING] No shapes provided to avshapecollapse")
        return None
    
    # Determine sampling method
    target_indices = np.ones(len(durations), dtype=bool)
    
    if method is not None:
        #uses avalanche shapes whose durations are inclusively bound by specified limits (scalar doubles)
        if method == 'limits': 
            lower_lim, upper_lim = args
            target_indices = (durations >= lower_lim) & (durations <= upper_lim)
        # uses avalanche shapes whose durations occur with frequency on the same order of magnitude
        # if magnitude is scalar or within the bounds of decades 10^(min(magnitude)) and 10^(max(magnitude)) if magnitude is a vector.
        elif method == 'order': 
            magnitude = args[0]
            if np.isscalar(magnitude):
                lower_lim = 10 ** magnitude
                upper_lim = 10 ** (magnitude + 1)
            else:
                lower_lim = 10 ** np.min(magnitude)
                upper_lim = 10 ** np.max(magnitude)
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (freqs >= lower_lim) & (freqs < upper_lim)
        # uses avalanche shapes of n different durations, linearly spaced between specified limits (scalar double)
        elif method == 'linspace':
            lower_lim, upper_lim, n = args
            target_durations = np.round(np.linspace(lower_lim, upper_lim, n))
            target_indices = np.isin(durations, target_durations)
        # uses avalanche shapes whose durations are logarithmically spaced between x^(lowerLim) and x^(upperLim) (scalar doubles)
        elif method == 'logspace':
            x, lower_lim, upper_lim = args
            target_durations = x ** np.arange(lower_lim, upper_lim + 1)
            target_indices = np.isin(durations, target_durations)
        # uses avalanche shapes of specific durations, durs (vector double)
        elif method == 'durations':
            target_durations = args[0]
            target_indices = np.isin(durations, target_durations)
        # uses avalanche shapes bounded below by both an absolute minimum duration (>= minDur) and a 
        # threshold for the frequency of occurrence (>= threshold) (scalar  doubles)
        elif method == 'cutoffs':
            lower_lim, threshold = args
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (np.unique(durations) >= lower_lim) & (freqs >= threshold)

    # Compute average shapes
    sampled_shapes = [shape for shape, index in zip(shapes, target_indices) if index]
    sampled_durations = durations[target_indices]

    if not sampled_shapes:
        print("[WARNING] No shapes remain after filtering in avshapecollapse")
        return {
            'exponent': None,
            'secondDer': None,
            'range': [],
            'errors': [],
            'coefficients': None
        }
    
    unique_durations = np.unique(sampled_durations)
    avg_shapes = []
    for dur in unique_durations:
        these_shapes = [shape for shape, d in zip(sampled_shapes, sampled_durations) if d == dur]
        avg_shapes.append(np.mean(these_shapes, axis=0))
    
    if not avg_shapes:
        print("[WARNING] No average shapes could be computed in avshapecollapse")
        return {
            'exponent': None,
            'secondDer': None,
            'range': [],
            'errors': [],
            'coefficients': None
        }
    
    ##############################
    precision=1e-3
    n_interp_points=1000
    bounds=(0, 3)
    n_avs = len(avg_shapes)
    max_duration = max([len(shape) for shape in avg_shapes])
    
    # Scale durations by duration length (t/T)
    scaled_durs = [(np.arange(1, len(shape) + 1) / len(shape)) for shape in avg_shapes]

    # Continually refine exponent value range to find optimal 1/(sigma nu z)
    n_iterations = int(-np.log10(precision))
    errors = []
    ranges = []
    
    for i in range(n_iterations):
        exponent_range = np.arange(bounds[0], bounds[1], 10 ** (-i - 1))
        errors_iteration = []
        for exponent in exponent_range:
            # Scale shapes by T^{1 - 1/(sigma nu z)}
            scaled_shapes = [shape * (len(shape) ** (1 - exponent)) for shape in avg_shapes]
            # Interpolate shapes to match maximum duration length
            interp_shapes = [np.interp(np.linspace(0, 1, n_interp_points), scaled_dur, scaled_shape) for scaled_shape, scaled_dur in zip(scaled_shapes, scaled_durs)]
            # Compute error of all shape collapses
            error = np.mean(np.var(interp_shapes, axis=0)) / ((np.max(interp_shapes) - np.min(interp_shapes)) ** 2)
            errors_iteration.append(error)
        errors.append(errors_iteration)
        ranges.append(exponent_range)
        # Find exponent value that minimizes error
        best_index = np.argmin(errors_iteration)
        sigma_nu_z_inv = exponent_range[best_index]
        # Generate new range of exponents to finer precision
        if i < n_iterations - 1:
            bounds = (sigma_nu_z_inv - 10 ** (-i - 1), sigma_nu_z_inv + 10 ** (-i - 1))
    
    # Fit to 2nd degree polynomial and find second derivative
    best_shapes = [shape * (len(shape) ** (1 - sigma_nu_z_inv)) for shape in avg_shapes]
    interp_best_shapes = [np.interp(np.linspace(0, 1, n_interp_points), scaled_dur, scaled_shape) for scaled_shape, scaled_dur in zip(best_shapes, scaled_durs)]
    avg_scaled_shape = np.mean(interp_best_shapes, axis=0)
    coeffs = np.polyfit(np.linspace(0, 1, n_interp_points), avg_scaled_shape, 2)
    second_drv = 2 * coeffs[0]
    
    # Plot and save
    if plot_flag or save_flag:
        plt.figure()
        # Plot all shapes
        for shape in interp_best_shapes:
            plt.plot(np.linspace(0, 1, n_interp_points), shape, alpha=0.5)
        # Overlay polynomial fit
        exponent_label = f'Exponent fit = {sigma_nu_z_inv:.2f}' if sigma_nu_z_inv is not None else 'Exponent not computed'
        plt.plot(np.linspace(0, 1, n_interp_points), np.polyval(coeffs, np.linspace(0, 1, n_interp_points)), '-r', linewidth=3, label=exponent_label)
    
        # Label axes
        plt.xlabel('Scaled Avalanche Duration (t/T)', fontsize=14)
        plt.ylabel('Scaled Avalanche Shapes', fontsize=14)
        # Set title and legend
        plt.title('Avalanche Shape Collapse', fontsize=14)
        #plt.legend(['Scaled shape', 'Polynomial fit'])
        plt.legend()
        # Save figure if required
        if save_flag:
            fname = filename + '_shapes'
            fig_name = f'{fname}.pdf'
            plt.savefig(fig_name)
            if not plot_flag:
                plt.close()
        else:
            fig_name = None
        plt.show()
    else:
        fig_name = None
        
    print('exponent 1/(sigma nu z):', sigma_nu_z_inv)
    
    Result = {
        'exponent': sigma_nu_z_inv,
        'secondDer': second_drv,
        'range': ranges,
        'errors': errors,
        'coefficients': coeffs
    }
    
    return Result

##############################################################

##############################################################
# Function to compute branching ratios, the naive way

def branching_ratios(shapes):
    branching_ratios = []

    for shape in shapes:
        # Skip avalanches with a duration of 1
        if len(shape) <= 1:
            continue
        # Calculate the branching ratio for each step in the avalanche,
        # skipping steps where the ancestor count is zero to avoid division by zero
        ratios = [shape[i] / shape[i - 1] for i in range(1, len(shape)) if shape[i - 1] != 0]
        # Calculate the average branching ratio for the avalanche
        if ratios:
            avg_ratio = np.mean(ratios)
            branching_ratios.append(avg_ratio)

    # Check if we have any valid branching ratios
    if len(branching_ratios) == 0:
        print("[WARNING] No valid branching ratios calculated - no avalanches with duration > 1")
        total_avg_branching_ratios = np.nan
        std_error = np.nan
    else:
        # Calculate the total average branching ratio and its standard error
        total_avg_branching_ratios = np.mean(branching_ratios)
        std_error = np.std(branching_ratios) / np.sqrt(len(branching_ratios))
    
    Result = {
    'BR': branching_ratios,
    'avgBR': total_avg_branching_ratios,
    'BRstd': std_error
    }
    return  Result
    
##############################################################

##############################################################
# Function for logarithmic binning avalanche data

def bin_data(data, logb, xmin, xmax):
    dist = np.sort(data)
    Nav = len(dist)
    jmax = int(np.round(np.log(max(dist)) / np.log(logb)))

    bincell = []

    for i in range(jmax):
        bincell.append(dist[(dist < logb ** (i + 1)) & (dist >= logb ** i)])

    Ps = []
    dist1 = []

    for j in range(len(bincell)):
        if len(bincell[j]) > 0:
            Ps.append(len(bincell[j]) / (Nav * (max(bincell[j]) - min(bincell[j]) + 1)))
            dist1.append(np.sqrt(max(bincell[j]) * min(bincell[j])))
        else:
            Ps.append(0)
            dist1.append(0)

    Ps = np.array(Ps)[Ps != 0]
    dist1 = np.array(dist1)[dist1 != 0]

    while xmax > (len(Ps) - 3):
        xmax = xmax - 1

    x = np.log10(dist1[xmin:xmax])
    y = np.log10(Ps[xmin:xmax])

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    m = abs(slope)
    merr = std_err

    x1 = np.linspace(0, 4.5)
    y1 = -m * x1 + intercept
    
    Result = {
    'Ps': Ps,
    'dist': dist1,
    'exponent': m,
    'expstd': merr,
    'fitdistx': x,
    'fitdisty': y,
    'fitx': x1,
    'fity': y1
    }

    return Result

##############################################################

##############################################################
# Function to compute Pearson correlations with lags
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def correlations(raster, lag=1, plot_eigenvalues=False, plot_weights=True, save_plot=True, saveloc=base_dir
, pltname='analysis', state = ''):
    """
    Compute the Pearson correlation matrix for neuron activities at a given lag, optionally plot eigenvalue spectrum, 
    plot the distribution of weights, and fit an exponential decay.

    Parameters:
        raster (np.array): The raster data matrix with shape (n_neurons, t_length).
        lag (int): The lag for computing correlations.
        plot_eigenvalues (bool): If True, plot the eigenvalue spectrum in the complex plane.
        plot_weights (bool): If True, plot the distribution of weights and fit an exponential decay.
        save_plot (bool): If True, save the plots instead of displaying them.
        saveloc (str): The directory to save the plot.
        pltname (str): The name of the saved plot file.
        
    Returns:
        correlation_matrix (np.array): The computed correlation matrix.
        max_eigenvalue (float): The maximum absolute eigenvalue of the correlation matrix.
        exponential_params (tuple): Parameters of the fitted exponential decay function.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag  # adjust length for lag
    correlation_matrix = np.zeros((n_neurons, n_neurons))

    for i in range(n_neurons):
        for j in range(n_neurons):
            series_i = raster[i, :(t_length)]
            series_j = raster[j, lag:(lag + t_length)]
            if np.any(series_i) and np.any(series_j):  # check if there's data in the slice
                correlation_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
            else:
                correlation_matrix[i, j] = 0

    # Compute eigenvalues for the original correlation matrix
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Plot eigenvalue spectrum if requested
    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum in the Complex Plane for {state}')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        else:
            plt.show()
        plt.close()

    # Plot distribution of weights and fit an exponential decay if requested
    exponential_params = None
    if plot_weights:
        # Get the absolute values of the correlation matrix and sort them for each neuron
        sorted_weights = np.sort(np.abs(correlation_matrix), axis=1)[:, ::-1]  # Sort in descending order
        
        # Aggregate the sorted weights across all neurons
        average_sorted_weights = np.mean(sorted_weights, axis=0)
        
        # Plot the distribution of weights
        plt.figure(figsize=(8, 6))
        plt.plot(average_sorted_weights, 'o-', label='Average sorted weights')
        
        # Fit an exponential decay to the distribution
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_data = np.arange(len(average_sorted_weights))
        
        # Initial parameter guess and bounds
        p0 = [max(average_sorted_weights), 0.1, min(average_sorted_weights)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        popt, _ = curve_fit(exp_decay, x_data, average_sorted_weights, p0=p0, bounds=bounds, maxfev=10000)
        exponential_params = popt
        
        # Plot the fitted exponential decay
        plt.plot(x_data, exp_decay(x_data, *popt), 'r--', 
                label=f'Exponential fit: a*exp(-b*x) + c\n(a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f})')
        
        plt.xlabel('Index Node')
        plt.ylabel('Weight')
        plt.title(f'Distribution of {state} Weights and Exponential Fit')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_weights.pdf')
        else:
            plt.show()
        plt.close()

    #return correlation_matrix, max_eigenvalue, exponential_params
    return max_eigenvalue, exponential_params

##############################################################


# ---- with shuffling
##############################################################
def correlations_with_shuffling(raster, lag=1, plot_eigenvalues=False, save_plot=True, saveloc=base_dir, pltname='eigenvalue_spectrum', n_shuffles=100):
    """
    Compute the Pearson correlation matrix for neuron activities at a given lag and optionally plot eigenvalue spectrum.
    Includes shuffling for noise estimation.

    Parameters:
        raster (np.array): The raster data matrix with shape (n_neurons, t_length).
        lag (int): The lag for computing correlations.
        plot_eigenvalues (bool): If True, plot the eigenvalue spectrum in the complex plane.
        save_plot (bool): If True, save the plot instead of displaying it.
        save_location (str): The directory to save the plot.
        plot_name (str): The name of the saved plot file.
        n_shuffles (int): Number of shuffles to generate surrogate data for noise estimation.
        
    Returns:
        correlation_matrix (np.array): The computed correlation matrix.
        max_eigenvalue_original (float): The maximum eigenvalue from the original data.
        max_eigenvalue_shuffled_mean (float): The mean of the maximum eigenvalues from shuffled data.
        max_eigenvalue_shuffled_std (float): The standard deviation of the maximum eigenvalues from shuffled data.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag  # adjust length for lag
    correlation_matrix = np.zeros((n_neurons, n_neurons))

    for i in range(n_neurons):
        for j in range(n_neurons):
            series_i = raster[i, :(t_length)]
            series_j = raster[j, lag:(lag + t_length)]
            if np.any(series_i) and np.any(series_j):  # check if there's data in the slice
                correlation_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
            else:
                correlation_matrix[i, j] = 0

    # Compute eigenvalues for the original correlation matrix
    eigenvalues_original = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue_original = np.max(np.abs(eigenvalues_original))

    # Generate surrogate data by shuffling and compute eigenvalues
    max_eigenvalues_shuffled = []
    for _ in range(n_shuffles):
        shuffled_raster = np.copy(raster)
        for i in range(n_neurons):
            np.random.shuffle(shuffled_raster[i, :])  # shuffle spike times within each neuron
        shuffled_correlation_matrix = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                series_i = shuffled_raster[i, :(t_length)]
                series_j = shuffled_raster[j, lag:(lag + t_length)]
                if np.any(series_i) and np.any(series_j):  # check if there's data in the slice
                    shuffled_correlation_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
                else:
                    shuffled_correlation_matrix[i, j] = 0
        eigenvalues_shuffled = np.linalg.eigvals(shuffled_correlation_matrix)
        max_eigenvalues_shuffled.append(np.max(np.abs(eigenvalues_shuffled)))

    max_eigenvalue_shuffled_mean = np.mean(max_eigenvalues_shuffled)
    max_eigenvalue_shuffled_std = np.std(max_eigenvalues_shuffled)

    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues_original.real, eigenvalues_original.imag, color='red', label='Original Data')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Spectrum in the Complex Plane')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        else:
            plt.show()
        plt.close()

    #return correlation_matrix, max_eigenvalue_original, max_eigenvalue_shuffled_mean, max_eigenvalue_shuffled_std
    return max_eigenvalue_original, max_eigenvalue_shuffled_mean, max_eigenvalue_shuffled_std

##############################################################

##############################################################
# Function to compute power spectrum
def welch_psd(raster_data, fs=1000, nperseg=10000, x1=0, x2=1, plot=False, save_plot=True, filename=os.path.join(base_dir, 'psd_plot.pdf')):
    """
    Compute the Power Spectral Density (PSD) using Welch's method.
    
    :param raster_data: 2D numpy array where each row is a time series for a given neuron.
    :param fs: Sampling frequency in Hz.
    :param nperseg: Length of each segment for Welch's method (buiilt in function, maybe we can try others later).
    :param x1, x2: Lower and upper bounds of the frequency region to fit.
    :param plot: Boolean, whether to display the plot.
    :param save_plot: Boolean, whether to save the plot as a PDF.
    :param filename: String, name of the file to save the plot.
    :return: Frequencies, PSD, and slope of the fit in the specified region.
    """
    # Collapse the raster data into one time series
    collapsed_series = np.sum(raster_data, axis=0)
    
    # Compute PSD using Welch's method (you can change the overlap window, but by default it does the nperseg/2 which is fine)
    f, psd = welch(collapsed_series, fs, nperseg=nperseg, noverlap=None)

    # Convert data to logarithmic scale
    log_f = np.log10(f[f > 0])  # avoid log(0) by ensuring frequencies are > 0
    log_psd = np.log10(psd[f > 0])

    # Define the region for fitting
    index1 = (log_f > x1) & (log_f < x2)  # Example region 1

    # Fit linear model to the region
    slope, intercept = np.polyfit(log_f[index1], log_psd[index1], 1)

    # Plotting stuff
    if plot or save_plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(f, psd, label='Original PSD')
        plt.loglog(10**log_f[index1], 10**(log_f[index1]*slope + intercept), 'r-', label=f'Fit: slope={slope:.2f}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.title('Power Spectral Density')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        if save_plot:
            plt.savefig(filename)
        if plot:
            plt.show()
        plt.close()

    return f, psd, slope

#################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.stats.stattools import durbin_watson

# Define the triple exponential model
def triple_exponential(x, A1, A2, A3, lambda1, lambda2, lambda3):
    return A1 * np.exp(-lambda1 * x) + A2 * np.exp(-lambda2 * x) + A3 * np.exp(-lambda3 * x)

# Define the double exponential model
def double_exponential(x, A1, lambda1, lambda2):
    return A1 * np.exp(-lambda1 * x) + (1 - A1) * np.exp(-lambda2 * x)

##############################################################
# Function to compute autocorrelation and fit the model
def compute_autocorrelation(time_series, name, min_lag=0, max_lag=None, model_type='triple'):
    """
    Compute the autocorrelation of a given time series and fit an exponential decay model (double or triple).
    
    :param time_series: 1D numpy array of time series data
    :param name: Name for saving the plot
    :param min_lag: Minimum lag to consider for the fit
    :param max_lag: Maximum lag to consider for the fit
    :param model_type: Choose between 'double' or 'triple' exponential fit
    :return: Fitted parameters based on the chosen model type
    """
    # Preprocess the time series
    time_series = time_series - np.mean(time_series)
    variance = np.var(time_series)

    # Compute the autocorrelation
    n = len(time_series)
    result = np.correlate(time_series, time_series, mode='full')
    autocorr = result[result.size // 2:] / (variance * np.arange(n, 0, -1))

    if max_lag is None:
        max_lag = len(autocorr)

    lags = np.arange(len(autocorr))
    fit_lags = lags[min_lag:max_lag]
    fit_autocorr = autocorr[min_lag:max_lag]

    # Fit the chosen model type
    try:
        if model_type == 'triple':
            # Initial parameters for triple exponential: A1, A2, A3, lambda1, lambda2, lambda3
            initial_params = (0.3, 0.3, 0.4, 0.01, 0.001, 0.0001)
            popt, _ = curve_fit(triple_exponential, fit_lags, fit_autocorr, p0=initial_params, maxfev=5000)
            
            # Extract the parameters A1, A2, A3, lambda1, lambda2, lambda3
            A1, A2, A3, lambda1, lambda2, lambda3 = popt
            # Sort parameters based on coefficients (A1, A2, A3) in descending order
            coeffs_and_lambdas = sorted(zip([A1, A2, A3], [lambda1, lambda2, lambda3]), reverse=True)
            (A1, lambda1), (A2, lambda2), (A3, lambda3) = coeffs_and_lambdas

            fitted_values = triple_exponential(fit_lags, A1, A2, A3, lambda1, lambda2, lambda3)
            fitted_params = (A1, A2, A3, lambda1, lambda2, lambda3)

            # Format the parameters for the legend
            legend_text = f"A1={A1:.2f}, A2={A2:.2f}, A3={A3:.2f},\n" \
                          f"tau1={1/lambda1:.4f}, tau2={1/lambda2:.4f}, tau3={1/lambda3:.4f}"

        elif model_type == 'double':
            # Initial parameters for double exponential: A1, lambda1, lambda2
            initial_params = (0.5, 0.01, 0.001)
            popt, _ = curve_fit(double_exponential, fit_lags, fit_autocorr, p0=initial_params, maxfev=5000)

            # Extract the parameters A1, lambda1, lambda2
            A1, lambda1, lambda2 = popt

            fitted_values = double_exponential(fit_lags, A1, lambda1, lambda2)
            fitted_params = (A1, lambda1, lambda2)

            # Format the parameters for the legend
            legend_text = f"A1={A1:.2f}, tau1={1/lambda1:.4f}, tau2={1/lambda2:.4f}"

        else:
            raise ValueError("Invalid model type. Choose 'double' or 'triple'.")

        # Compute residuals
        residuals = fit_autocorr - fitted_values
        dw_stat = durbin_watson(residuals)  # Apply Durbin-Watson test for autocorrelation in residuals
        print(f'Durbin-Watson statistic for {name}: {dw_stat}')

    except RuntimeError:
        print(f"Failed to fit the {model_type} exponential model for {name}.")
        if model_type == 'triple':
            fitted_params = [np.nan] * 6
            legend_text = "Fit failed"
        elif model_type == 'double':
            fitted_params = [np.nan] * 3
            legend_text = "Fit failed"
        fitted_values = np.zeros_like(fit_autocorr)

    # Plot results
    plt.figure(figsize=(12, 6))

    # Linear scale plot
    plt.subplot(1, 2, 1)
    plt.plot(fit_lags, fit_autocorr, label='Autocorrelation')
    plt.plot(fit_lags, fitted_values, 'r-', label=f'{model_type.capitalize()} Exponential Fit')
    plt.legend(loc='upper right', title=legend_text)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation with {model_type.capitalize()} Exponential Fit (Linear Scale)')
    plt.grid(False)

    # Logarithmic scale plot
    plt.subplot(1, 2, 2)
    plt.plot(fit_lags, fit_autocorr, label='Autocorrelation')
    plt.plot(fit_lags, fitted_values, 'r-', label=f'{model_type.capitalize()} Exponential Fit')
    plt.yscale('log')
    plt.legend(loc='upper right', title=legend_text)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation (log scale)')
    plt.title(f'Autocorrelation with {model_type.capitalize()} Exponential Fit (Log Scale)')
    plt.grid(False)

    # Save the figure as a PDF
    plt.savefig(f'ACF_{name}_{model_type}_exp_fit.pdf')
    plt.close()

    return fitted_params

# Example usage:
# compute_autocorrelation(time_series_data, 'example', model_type='double')  # For double exponential fit
# compute_autocorrelation(time_series_data, 'example', model_type='triple')  # For triple exponential fit


'''
def double_exponential(x, A1, lambda1, A2, lambda2):
    return A1 * np.exp(-lambda1 * x) + A2 * np.exp(-lambda2 * x)

def compute_autocorrelation(time_series,name):
    """
    Compute the autocorrelation of a given time series and fit a double exponential decay.

    :param time_series: 1D numpy array of time series data
    :return: autocorrelation values
    """
    n = len(time_series)
    time_series = time_series - np.mean(time_series)
    result = np.correlate(time_series, time_series, mode='full')
    autocorr = result[result.size // 2:] / max(result)
    
    # Fit the double exponential decay
    lags = np.arange(len(autocorr))
    try:
        popt, _ = curve_fit(double_exponential, lags, autocorr, p0=(1, 0.01, 1, 0.001), maxfev=5000, bounds=(0, np.inf))
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan, np.nan]  # if fitting fails, return NaNs


    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Linear y-axis plot
    plt.subplot(1, 2, 1)
    plt.plot(lags, autocorr, label='Autocorrelation')
    plt.plot(lags, double_exponential(lags, *popt), 'r-', label=f'Fit: A1={popt[0]:.2f}, lambda1={popt[1]:.4f}, A2={popt[2]:.2f}, lambda2={popt[3]:.4f}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation with Double Exponential Fit (Linear Scale)')
    plt.legend()
    plt.grid(False)
    
    # Logarithmic y-axis plot
    plt.subplot(1, 2, 2)
    plt.plot(lags, autocorr, label='Autocorrelation')
    plt.plot(lags, double_exponential(lags, *popt), 'r-', label=f'Fit: A1={popt[0]:.2f}, lambda1={popt[1]:.4f}, A2={popt[2]:.2f}, lambda2={popt[3]:.4f}')
    plt.yscale('log')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation (log scale)')
    plt.title('Autocorrelation with Double Exponential Fit (Log Scale)')
    plt.legend()
    plt.grid(False)
    
    # Save the figure as a PDF
    #plt.savefig(f'autocorrelation_{name}.pdf')
    plt.show
    #plt.close()

    return autocorr, popt
'''

# Function to compute eigenvalues and identify state-dense windows
def compute_measures_and_state_dense(data, sleep_states, window_size_bins, overlap_bins, binsz, condition):
    n_windows = (data.shape[1] - window_size_bins) // overlap_bins + 1
    print("# windows:", n_windows)
    results = []

    for i in range(n_windows):
        print("----------- New window ---------------")
        print("--------------------------------------")
        start_bin = i * overlap_bins
        end_bin = start_bin + window_size_bins
        temp = data[:, start_bin:end_bin] # this is the piece of data
        print("start bin", start_bin)
        print("end bin", end_bin)
        print("----------- sleep states -------------")

        # Compute correlation matrix
        correlation_matrix = compute_correlations(temp, lag=0)
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        max_evalue0 = np.max(np.abs(eigenvalues))
        correlation_matrix = compute_correlations(temp, lag=1)
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        max_evalue1 = np.max(np.abs(eigenvalues))
        correlation_matrix = compute_correlations(temp, lag=2)
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        max_evalue2 = np.max(np.abs(eigenvalues))

        # compute autocorrelations
        
        _, fits = compute_autocorrelation(np.sum(temp, axis=0), condition + '_' + str(i))
        # Determine if the window is sleep-dense or wake-dense
                
        # convert time bins into 4 second bins for the sleep_states
        startss = int(start_bin*binsz/4)
        endss = int(end_bin*binsz/4)
        print("startss", startss)
        print("endss", endss)
        
        sleep_duration = np.sum(sleep_states[startss:endss] == 2) + np.sum(sleep_states[startss:endss] == 3)
        print("sleep duration:", sleep_duration)
        wake_duration = int(window_size_bins*binsz/4) - sleep_duration
        print("wake duration:", wake_duration)
        if sleep_duration >= 0.6 * int(window_size_bins*binsz/4):
            state = 'sleep'
            print("sleep dense")
        elif wake_duration >= 0.6 * int(window_size_bins*binsz/4):
            state = f'wake/{condition}'
            print("wake dense")
        else:
            state = 'none'

        results.append({
            'window': i+1, 
            'behavior_condition': state, 
            'max_evalue lag 0': max_evalue0,
            'max_evalue lag 1': max_evalue1,
            'max_evalue lag 2': max_evalue2, 
            'A1': fits[0],'lambda1': 1/fits[1],
            'A2': fits[2],'lambda2': 1/fits[3],})
        

    return results


def get_h5_files(backup_path):
    """Get paths to all result.h5 files from a specified backup directory."""
    # Get all timestamped folders
    date_folders = [f for f in os.listdir(backup_path) if f.startswith('202')]
    date_folders.sort()  # Sort chronologically

    h5_files = []
    for folder in date_folders:
        h5_path = os.path.join(backup_path, folder, 'common', 'result.h5')
        if os.path.exists(h5_path):
            h5_files.append(h5_path)
            print(f"Found H5 file in: {folder}")

    return h5_files

def process_h5_file(file_path, starting_time_point, end_time_point):
    """
    Process a single .h5 file to extract raster data starting from a given time point.
    """
    try:
        h5 = tables.open_file(file_path, 'r')
        data = h5.root

        if data.__contains__('Spikes'):
            print(f"Looking at raster in file: {file_path}")
            last_spikes = data.c.stats.only_last_spikes[0]
            tmp_p_raster = data.Spikes[0, :, -last_spikes:]
            raster = tmp_p_raster[:, starting_time_point:end_time_point] if starting_time_point is not None else tmp_p_raster
            print(f"Raster shape for {file_path}: {raster.shape}")
            return raster
        else:
            print(f"No 'Spikes' data found in {file_path}.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None
    finally:
        try:
            h5.close()
        except:
            pass

def get_and_process_rasters(backup_path, starting_time_point):
    """
    Retrieve and process raster data from all .h5 files in a given backup path.
    """
    # Get list of .h5 files
    h5_files = get_h5_files(backup_path)
    print(f"\nFound {len(h5_files)} H5 files to analyze")

    all_rasters = []

    # Process each file
    for file_path in h5_files:
        print(f"\nProcessing: {file_path}")
        raster = process_h5_file(file_path, starting_time_point)
        if raster is not None:
            all_rasters.append(raster)

    # Combine all rasters into a single structure if needed
    if all_rasters:
        combined_raster = np.hstack(all_rasters)
        print(f"\nCombined raster shape: {combined_raster.shape}")
        return combined_raster
    else:
        print("No valid rasters processed.")
        return None
    
def process_raster(data_raster, start_time=None, end_time=None):
    # Convert from sparse to dense matrix if necessary
    if hasattr(data_raster, 'toarray'):
        tmp_p_raster = data_raster.toarray()
    else:
        tmp_p_raster = data_raster
    
    # Print shape (equivalent to size in MATLAB)
    print(tmp_p_raster.shape)
    
    tmp_raster = tmp_p_raster
    
    # Calculate average activity
    avgActivity = np.mean(np.sum(tmp_raster > 0, axis=0))
    print(f"Average activity: {avgActivity}")
    
    # Create binary raster
    binaryRaster = tmp_raster > 0
    
    # Calculate activity per time point
    activityPerTimePoint = np.sum(binaryRaster, axis=0)
    
    # Create mask for above average activity
    aboveAverageMask = activityPerTimePoint >= avgActivity/2
    
    # Create full mask
    fullMask = np.broadcast_to(aboveAverageMask, tmp_raster.shape)
    
    # Apply mask to raster
    raster = tmp_raster * fullMask
    
    # Handle time range subsetting
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = raster.shape[1]
        
    # Ensure indices are within bounds
    start_time = max(0, start_time)
    end_time = min(raster.shape[1], end_time)
    
    # Return the time-ranged subset of the raster
    return raster[:, start_time:end_time]

def correlationsOptimized(raster, lag=1, plot_eigenvalues=False, plot_weights=True, save_plot=True, 
                saveloc=base_dir,
                pltname='analysis', state=''):
    """
    Fixed version of the optimized correlations function that matches original behavior.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag
    correlation_matrix = np.zeros((n_neurons, n_neurons))

    # Pre-compute the time series for all neurons
    series_early = raster[:, :t_length]  # All neurons, early timepoints
    series_late = raster[:, lag:lag + t_length]  # All neurons, later timepoints
    
    # Check for non-zero series
    any_early = np.any(series_early, axis=1)
    any_late = np.any(series_late, axis=1)
    valid_pairs = np.outer(any_early, any_late)
    
    # Where both series have data, compute correlation using np.corrcoef
    for i in range(n_neurons):
        for j in range(n_neurons):
            if valid_pairs[i, j]:
                correlation_matrix[i, j] = np.corrcoef(
                    series_early[i], 
                    series_late[j]
                )[0, 1]

    # Compute eigenvalues using faster method for symmetric matrices
    # eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Plot eigenvalue spectrum if requested
    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum in the Complex Plane for {state}')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        else:
            plt.show()
        plt.close()

    # Plot distribution of weights and fit exponential decay if requested
    exponential_params = None
    if plot_weights:
        # Vectorized computation of sorted weights
        sorted_weights = -np.sort(-np.abs(correlation_matrix), axis=1)
        average_sorted_weights = np.mean(sorted_weights, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(average_sorted_weights, 'o-', label='Average sorted weights')
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_data = np.arange(len(average_sorted_weights))
        p0 = [max(average_sorted_weights), 0.1, min(average_sorted_weights)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            popt, _ = curve_fit(exp_decay, x_data, average_sorted_weights, 
                              p0=p0, bounds=bounds, maxfev=10000)
            exponential_params = popt
            
            plt.plot(x_data, exp_decay(x_data, *popt), 'r--', 
                    label=f'Exponential fit: a*exp(-b*x) + c\n(a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f})')
        except:
            print("Warning: Could not fit exponential decay")
        
        plt.xlabel('Index Node')
        plt.ylabel('Weight')
        plt.title(f'Distribution of {state} Weights and Exponential Fit')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_weights.pdf')
        else:
            plt.show()
        plt.close()

    return max_eigenvalue, exponential_params

def correlationsMoreOptimized(raster, lag=1, plot_eigenvalues=False, plot_weights=True, save_plot=True, 
                saveloc=base_dir,
                pltname='analysis', state=''):
    """
    Vectorized version of the correlations function using numpy operations.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag

    # Pre-compute the time series
    series_early = raster[:, :t_length]
    series_late = raster[:, lag:lag + t_length]
    
    # Compute means and standard deviations for all neurons at once
    means_early = np.mean(series_early, axis=1, keepdims=True)
    means_late = np.mean(series_late, axis=1, keepdims=True)
    
    stds_early = np.std(series_early, axis=1, keepdims=True)
    stds_late = np.std(series_late, axis=1, keepdims=True)
    
    # Normalize the data in a vectorized way
    # Handle zero standard deviations to avoid division by zero
    valid_early = stds_early > 0
    valid_late = stds_late > 0
    
    # Initialize normalized arrays
    norm_early = np.zeros_like(series_early)
    norm_late = np.zeros_like(series_late)
    
    # Normalize only where std > 0
    norm_early[valid_early.ravel()] = ((series_early - means_early) / stds_early)[valid_early.ravel()]
    norm_late[valid_late.ravel()] = ((series_late - means_late) / stds_late)[valid_late.ravel()]
    
    # Compute correlation matrix using matrix multiplication
    correlation_matrix = np.zeros((n_neurons, n_neurons))
    valid_pairs = np.outer(valid_early.ravel(), valid_late.ravel())
    
    # Only compute correlations for valid pairs
    if np.any(valid_pairs):
        # Use matrix multiplication for vectorized correlation computation
        correlation_matrix[valid_pairs] = (
            np.dot(norm_early, norm_late.T)[valid_pairs] / t_length
        )

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Plot eigenvalue spectrum if requested
    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum in the Complex Plane for {state}')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        plt.close()

    # Plot distribution of weights and fit exponential decay if requested
    exponential_params = None
    if plot_weights:
        # Vectorized computation of sorted weights
        sorted_weights = -np.sort(-np.abs(correlation_matrix), axis=1)
        average_sorted_weights = np.mean(sorted_weights, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(average_sorted_weights, 'o-', label='Average sorted weights')
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_data = np.arange(len(average_sorted_weights))
        p0 = [max(average_sorted_weights), 0.1, min(average_sorted_weights)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            popt, _ = curve_fit(exp_decay, x_data, average_sorted_weights, 
                              p0=p0, bounds=bounds, maxfev=10000)
            exponential_params = popt
            
            plt.plot(x_data, exp_decay(x_data, *popt), 'r--', 
                    label=f'Exponential fit: a*exp(-b*x) + c\n(a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f})')
        except:
            print("Warning: Could not fit exponential decay")
        
        plt.xlabel('Index Node')
        plt.ylabel('Weight')
        plt.title(f'Distribution of {state} Weights and Exponential Fit')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_weights.pdf')
        plt.close()

    return max_eigenvalue, exponential_params

""" 
#
#
# This is the main driving part of this code
#
#
"""

matching_dirs = []
for dirname in os.listdir(base_dir):
   full_path = os.path.join(base_dir, dirname)
   print("-- Full path:", full_path)
#    if os.path.isdir(full_path):
#        match = re.match(pattern, dirname)
#        if match:
#            mu_value = float(match.group(1))  # Extract mu value
#            sigma_value = float(match.group(2))  # Extract sigma value
#            sp_steps = float(match.group(3)) 
#            sigma_value_no_sp = float(match.group(4)) 
#            print("-- Dir:", full_path)
#            print("-- mu: ", mu_value)
#            print("-- sigma: ", sigma_value)
#            print("-- sp_steps: ", sp_steps)
#            print("-- sigma_value_no_sp: ", sigma_value_no_sp)
#            matching_dirs.append((full_path, mu_value, sigma_value))
overall_susc=[]
overall_rho_values=[]
overall_cv_values=[]
overall_br_method_1=[]
overall_br_method_2=[]
overall_br_priesman=[]
mu_values = []
overall_pearson_kappa=[]

av_collapse_exponent=[]
av_collapse_secondDer=[]
av_collapse_range=[]
av_collapse_errors=[]
av_collapse_coefficients=[]
av_collapse_min_error=[]

#for directory, mu, sigma in matching_dirs:
if True : 

    # # # Call to retrieve and combine raster data
    # backup_path = r"D:\Users\seaco\SORN\backup\test_single\'"
    # combined_raster = get_and_process_rasters(backup_path, starting_time_point)

    # # # Check if data was successfully retrieved
    # if combined_raster is not None:
    #     print("Raster data successfully retrieved and combined.")
    # else:
    #     print("No raster data retrieved. Exiting.")
    #     exit()

    # print("Debug susceptibility: " , susceptibility(combined_raster))
    # print("Debug rho: " , rho(combined_raster))

    starting_time_point = 1000000
    end_time_point = 3500000  # Set to None to use the full length of the raster

    ## Need to loop over all the folders associated to the current param
    # single_param_backup_path = directory + '\\test_single'
    # single_param_backup_path = directory
    single_param_backup_path = base_dir
    mu = 0.05
    sigma = 0.05

    # Initialize arrays to store processed data
    all_rasters = []

    susc, rho_values, cv_values, br_method_1, br_method_2, br_priesman = [], [], [], [], [], []
    pearson_kappa = []

    # Initialize arrays to store all avalanches
    all_burst = np.array([])
    all_T = np.array([], dtype=int)
    all_shapes = []

    # Get list of H5 files
    h5_files = get_h5_files(single_param_backup_path)
    print(f"\nFound {len(h5_files)} H5 files to analyze for mu={mu}")
    print(f"[INFO] Located {len(h5_files)} .h5 simulation files to process.")
    if(len(h5_files) > 0):

        # Process each H5 file
        for file_path in h5_files:
            print(f"\nProcessing: {file_path}")

            try:
                # Load data
                h5 = tables.open_file(file_path, 'r')
                data = h5.root

                # Debug prints to understand the structure
                print("File structure:")
                h5.list_nodes('/')

                # Try different methods to read the data
                pickle_dir = None
                for node in h5.iter_nodes(h5.root.c, 'Array'):
                    if node.name == 'logfilepath':
                        pickle_dir = str(node[0])
                        break

                if data.__contains__('Spikes'):

                    ######################
                    # Raster plot (last_n_spikes)
                    ######################
                    print("Looking at raster in file: ", file_path)
                    last_spikes = data.c.stats.only_last_spikes[0]
                    print("Last Spike (from stats): ", last_spikes)

                    # Check actual data dimensions
                    actual_data_length = data.Spikes.shape[2]
                    print("Actual data length: ", actual_data_length)

                    # Define the actual end point using real data dimensions
                    actual_end = min(end_time_point, actual_data_length) if end_time_point else actual_data_length

                    print(f"Time window: {starting_time_point} to {actual_end}")
                    print(f"Requested end point: {end_time_point}")
                    print(f"Available data ends at: {actual_data_length}")

                    # Load ONLY the time window directly from HDF5
                    raster = data.Spikes[0, :, starting_time_point:actual_end]
                    print(f"Raster shape for {file_path}: {raster.shape}")
                    print(f"Time window: {starting_time_point} to {actual_end}")
                    print(f"Requested end point: {end_time_point}")
                    print(f"Available data ends at: {last_spikes}")
                    print(f"Actually analyzing {raster.shape[1]} time steps")
                    print(f"Percentage of requested window analyzed: {(actual_end - starting_time_point) / (end_time_point - starting_time_point) * 100:.1f}%")
                    #raster = process_raster(tmp_p_raster, start_time=starting_time_point)
                    print(f"Raster shape for {file_path}: {raster.shape}")

                    ### Concatenating rasters - not memory efficient
                    # all_rasters.append(raster)
                    
                    ######################
                    # Per raster stats
                    # Will eventually need to average stats over different rasters 
                    ######################
                    single_susc = susceptibility(raster)
                    susc.append(single_susc)
                    print("-- Susceptibility: ", single_susc)

                    single_rho = rho(raster)
                    rho_values.append(single_rho)
                    print("-- Rho: ", single_rho)

                    single_cv = cv(raster)
                    cv_values.append(single_cv)
                    print("-- CV: ", single_cv)
                    
                    # single_br_method_1 = 0
                    single_br_method_1 = branchparam(raster, lverbose=0)
                    br_method_1.append(single_br_method_1)
                    print("-- Branching param [method 1]: ", single_br_method_1)

                    max_eig, exp_params = correlationsMoreOptimized(raster, lag=1, plot_eigenvalues=False, plot_weights=False, 
                                                   save_plot=False, saveloc='', pltname='analysis', state = '')
                    print(f"Maximum eigenvalue: {max_eig}")
                    pearson_kappa.append(max_eig)

                    single_results = cr.get_avalanches(raster, perc=0.25, ncells=-1, const_threshold=None)
                    
                    if 'S' in single_results and len(single_results['S']) > 0:
                        print(f"[INFO] Found {len(single_results['S'])} avalanches")
                        
                        # Concatenating avalanche statistics - this is more memory efficient
                        all_burst = np.concatenate((all_burst, single_results['S']))
                        all_T = np.concatenate((all_T, single_results['T']))
                        
                        # Since cr.get_avalanches doesn't return shapes, use our own function
                        print("[INFO] Computing avalanche shapes using find_avalanches...")
                        manual_shapes = find_avalanches(raster)
                        
                        if len(manual_shapes) > 0:
                            all_shapes.extend(manual_shapes)
                            print(f"[INFO] Found {len(manual_shapes)} avalanche shapes")
                            
                            # Calculate branching ratios using the shapes
                            try:
                                single_results_br = branching_ratios(manual_shapes)
                                single_br_method_2 = single_results_br['avgBR']
                                print(f"-- Branching param [method 2]: {single_br_method_2}")
                            except Exception as e:
                                print(f"[WARNING] Error calculating branching ratios: {e}")
                                single_br_method_2 = np.nan
                        else:
                            print("[WARNING] No avalanche shapes found")
                            single_br_method_2 = np.nan
                    else:
                        print("[INFO] No avalanches found")
                        single_br_method_2 = 0

                    br_method_2.append(single_br_method_2)

                    activity_array = np.sum(raster, axis=0) - rho(raster)
                    # single_priesman_br = 0
                    single_priesman_br = calc_BR(activity_array, 100, 0.001, os.path.join(base_dir, f'priesman_br_mu_{mu}.pdf'))
                    br_priesman.append(single_priesman_br)
                    print("-- Branching param [method Priesman]: ", single_priesman_br)

                    # #print(correlations_with_shuffling(raster))
                    # #print(correlations(raster))
                    # welch_psd(raster)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
            finally:
                try:
                    h5.close()
                except:
                    pass
        
        # Save individual metrics to a `.txt` file 
        text_output_path = os.path.join(base_dir, f'Stats_Mu_{mu:.2f}.txt')
        with open(text_output_path, 'w') as f:
            f.write(f"Metrics Summary for Mu = {mu:.2f}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Individual Susceptibility Values: {susc}\n")
            f.write(f"Mean Susceptibility: {np.mean(susc):.4f}\n")
            f.write(f"Individual Rho Values: {rho_values}\n")
            f.write(f"Mean Rho: {np.mean(rho_values):.4f}\n")
            f.write(f"Individual CV Values: {cv_values}\n")
            f.write(f"Mean CV: {np.mean(cv_values):.4f}\n")
            f.write(f"Individual Branching Ratio (Method 1 - branchparam) Values: {br_method_1}\n")
            f.write(f"Mean Branching Ratio (Method 1 - branchparam): {np.mean(br_method_1):.4f}\n")
            f.write(f"Individual Branching Ratio (Method 2 - get_avalanches) Values: {br_method_2}\n")
            f.write(f"Mean Branching Ratio (Method 2 - get_avalanches): {np.mean(br_method_2):.4f}\n")
            f.write(f"Individual Branching Ratio (Method 3 - Priesman) Values: {br_priesman}\n")
            f.write(f"Mean Branching Ratio (Method 3 - Priesman): {np.mean(br_priesman):.4f}\n")
            f.write(f"Mean Pearson Kappa coefficient: {np.mean(pearson_kappa):.4f}\n")
            f.write("-" * 60 + "\n")
        print(f"Metrics for Mu = {mu:.2f} saved to: {text_output_path}")

        overall_susc.append(np.mean(susc))
        overall_rho_values.append(np.mean(rho_values))
        overall_cv_values.append(np.mean(cv_values))
        overall_br_method_1.append(np.mean(br_method_1))
        overall_br_method_2.append(np.mean(br_method_2))
        overall_br_priesman.append(np.mean(br_priesman))
        mu_values.append(mu)
        overall_pearson_kappa.append(np.mean(pearson_kappa))


        # Combine all rasters into a single structure if needed
        #if all_rasters:
        if True:
            #combined_raster = np.hstack(all_rasters)
            #print(f"\nCombined raster shape: {combined_raster.shape}")

            # Aggregate avalanches 
            #results = cr.get_avalanches(combined_raster, perc=0.25, ncells=-1, const_threshold=None)

            # compute some stats on avalanche shapes
            avg_results = None
            #avg_results = avshapecollapse(results['shapes'], results['T'], method=None, args=(),plot_flag=True, save_flag=True, filename=f'D:\\Users\\seaco\\SORN\\backup\\test_single\\sanity_check\\AvShapeCollapse_mu={mu}')
            # -> avg_results = avshapecollapse(all_shapes, all_T, method=None, args=(),plot_flag=True, save_flag=True, filename=f'D:\\Users\\seaco\\SORN\\backup\\backup\\test_single\\sanity_check\\AvShapeCollapse_mu={mu}')

            # Handle the results with error checking
            if avg_results is not None:
                av_collapse_exponent.append(avg_results['exponent'])
                av_collapse_secondDer.append(avg_results['secondDer'])
                av_collapse_range.append(avg_results['range'])
                av_collapse_errors.append(avg_results['errors'])
                av_collapse_coefficients.append(avg_results['coefficients'])
                if avg_results['errors'] and avg_results['errors'][-1]:  # Check if errors list is not empty
                    min_error = np.min(avg_results['errors'][-1])
                else:
                    min_error = np.nan  # or some other default value
                av_collapse_min_error.append(min_error)
            else:
                # Append default values when no results are available
                av_collapse_exponent.append(np.nan)  # or None, depending on what you prefer
                av_collapse_secondDer.append(np.nan)
                av_collapse_range.append([])
                av_collapse_errors.append([])
                av_collapse_coefficients.append(None)
                av_collapse_min_error.append(np.nan)
                print(f"[WARNING] No avalanche shape collapse results available for mu={mu}")

            # Plot log-log avalanche stats and crackling noise graph
            # scaling_plots(results['S'], results['T'], pltname=f'scaling_plot_mu={mu}',saveloc=base_dir, show_plot=False, plot_type='pdf')
            #-> scaling_plots(all_burst, all_T, pltname=f'scaling_plot_mu={mu}', saveloc=base_dir, show_plot=False, plot_type='pdf')
            
            # Compute branchig ratio using all aggregate avalanches
            # results_br = branching_ratios(results['shapes'])
            #-> results_br = branching_ratios(all_shapes)
            #-> print("-- Combined Branching param [method 2]: ", results_br['avgBR'])

        else:
            print("No valid rasters processed.")

    # AV_Result = cr.AV_analysis(burst=all_burst,
    #                      T=all_T,
    #                      flag=1,          # Enable p-value testing
    #                      plot=True,       # Generate plots
    #                      pltname='avalanche_analysis_',  # Prefix for saved plots
    #                      saveloc=base_dir  # Save in same directory as other plots
    #                      )
    # # Print key results
    # print("\nAnalysis Results:")
    # print(f"Alpha (size exponent): {AV_Result['alpha']:.3f}")
    # print(f"Beta (duration exponent): {AV_Result['beta']:.3f}")
    # print(f"Size range: {AV_Result['xmin']:.0f} to {AV_Result['xmax']:.0f}")
    # print(f"Duration range: {AV_Result['tmin']:.0f} to {AV_Result['tmax']:.0f}")
    # if AV_Result['P_burst'] is not None:
    #     print(f"Size distribution p-value: {AV_Result['P_burst']:.3f}")
    # if AV_Result['P_t'] is not None:
    #     print(f"Duration distribution p-value: {AV_Result['P_t']:.3f}")
    # print(f"Scaling relation difference: {AV_Result['df']:.3f}")

import matplotlib.pyplot as plt
import pandas as pd


# Prepare data for saving
results_df = pd.DataFrame({
    'Mu': mu_values,
    'Overall_Susceptibility': overall_susc,
    'Overall_Rho': overall_rho_values,
    'Overall_CV': overall_cv_values,
    'Branching_Ratio_Method_1': overall_br_method_1,
    'Branching_Ratio_Method_2': overall_br_method_2,
    'Branching_Ratio_Priesman': overall_br_priesman,
    'Pearson_Kappa': overall_pearson_kappa,
    'Av_Collapse_exponent': av_collapse_exponent,
    'Av_Collapse_error': av_collapse_min_error
})

# Loop over each mu and save individual metrics to separate text files
for i in range(len(mu_values)):
    # Create a unique file name for each mu
    text_output_path = os.path.join(base_dir, f'Stats_Mu_{mu_values[i]:.2f}.txt')

    with open(text_output_path, 'w') as f:
        f.write(f"Metrics Summary for Mu = {mu_values[i]:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean Susceptibility: {overall_susc[i]:.4f}\n")
        f.write(f"Mean Rho: {overall_rho_values[i]:.4f}\n")
        f.write(f"Mean CV: {overall_cv_values[i]:.4f}\n")
        f.write(f"Mean Branching Ratio (Method 1 - branchparam): {overall_br_method_1[i]:.4f}\n")
        f.write(f"Mean Branching Ratio (Method 2 - get_avalanches): {overall_br_method_2[i]:.4f}\n")
        f.write(f"Mean Branching Ratio (Method 3 - Priesman): {overall_br_priesman[i]:.4f}\n")
        f.write(f"Mean Pearson Kappa: {overall_pearson_kappa[i]:.4f}\n")
        f.write(f"Av_Collapse_exponent: {av_collapse_exponent[i]:.4f}\n")
        f.write(f"Av_Collapse_error: {av_collapse_min_error[i]:.4f}\n")
        f.write("-" * 60 + "\n")

    print(f"Metrics for Mu = {mu_values[i]:.2f} saved to: {text_output_path}")


# Save to CSV
csv_output_path = os.path.join(base_dir, 'Overall_Stats.csv')
results_df.to_csv(csv_output_path, index=False)
print(f"Aggregated results saved to: {csv_output_path}")

# Load data from CSV
results_df = pd.read_csv(csv_output_path)

# Extract columns for plotting
mu_values = results_df['Mu']
overall_susc = results_df['Overall_Susceptibility']
overall_rho_values = results_df['Overall_Rho']
overall_cv_values = results_df['Overall_CV']
overall_br_method_1 = results_df['Branching_Ratio_Method_1']
overall_br_method_2 = results_df['Branching_Ratio_Method_2']
overall_br_priesman = results_df['Branching_Ratio_Priesman']
overall_pearson_kappa = results_df['Pearson_Kappa']
av_collapse_exponent = results_df['Av_Collapse_exponent']
av_collapse_min_error = results_df['Av_Collapse_error']

# Set up the plotting style
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

width = 6  # Fixed width for individual subplots
height = width * 1.5  # Taller height for individual subplots (height > width)

# Create a figure with a grid of subplots (5 rows, 2 columns)
fig, axes = plt.subplots(5, 2, figsize=(width * 2, height * 5))  # Adjust figure size to accommodate all subplots
axes = axes.flatten()

# Common plotting parameters
plot_params = dict(
    marker='x',
    linestyle=':',  # Dotted line for style consistency
    markersize=6,
    linewidth=1.5
)

# Plot each dataset
plotting_data = [
    (overall_susc, 'Susceptibility ($\\chi$)'),
    (overall_rho_values, 'Rho ($\\rho$)'),
    (overall_cv_values, 'CV'),
    (overall_br_method_1, 'Branching Ratio (Method 1 - branchparam)'),
    (overall_br_method_2, 'Branching Ratio (Method 2 - get_avalanches)'),
    (overall_br_priesman, 'Branching Ratio (Method 3 - Priesman)'),
    (overall_pearson_kappa, 'Pearson $\\kappa$'),
    (av_collapse_exponent, 'Av_Collapse_exponent 1/($\\sigma$$\\nu$z)'),
    (av_collapse_min_error, 'Av_Collapse_error')
]

# Preprocessing: Remove invalid data points (e.g., NaN)
for idx, (data, title) in enumerate(plotting_data):
    if idx < len(axes):
        ax = axes[idx]
        
        # Convert to NumPy arrays for filtering
        mu_clean = np.array(mu_values)
        data_clean = np.array(data)
        
        # Remove NaN or None values
        valid_mask = ~np.isnan(data_clean) & ~np.isnan(mu_clean)
        mu_clean = mu_clean[valid_mask]
        data_clean = data_clean[valid_mask]
        
        # Plot cleaned data
        ax.plot(mu_clean, data_clean, label=title, **plot_params)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel(r'$\mu$', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        
        # Customize grid and spines
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(frameon=True, fontsize=8)

# Remove any unused subplot slots
for idx in range(len(plotting_data), len(axes)):
    fig.delaxes(axes[idx])

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.2, wspace=0.3)  # Reduced hspace for less vertical space between rows

# Save the entire figure as a PDF with high DPI
output_path = os.path.join(base_dir, 'All_Overall_Stats.pdf')
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')

# Show the plots
plt.show()

print(f"Aggregated plot saved as a PDF at {output_path}")

