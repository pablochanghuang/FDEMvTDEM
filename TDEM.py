import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh
import time

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem
from simpeg.utils import plot_1d_layer_model

#Set Solver
#from simpeg.utils.solver_utils import get_default_solver
#Solver = get_default_solver()

plt.rcParams.update({"font.size": 16})
write_output = False


def calculate_rms_difference(data_with_layer, data_without_layer, noise_percentage):
    """
    Calculate the RMS difference between two complex datasets (with and without a layer).
    
    Parameters:
    - data_with_layer: numpy array of complex values with the layer.
    - data_without_layer: numpy array of complex values without the layer.
    
    Returns:
    - rms_real: RMS difference for the real parts.
    - rms_imag: RMS difference for the imaginary parts.
    - rms_combined: Combined RMS difference.
    """
    # Calculate the differences
    delta = (data_with_layer - data_without_layer)/(np.abs(data_with_layer)*noise_percentage)

    # Calculate RMS for each part
    rms_real = np.sqrt(np.mean(delta**2))


    return rms_real



def create_TDEMmodel(conductivity = 1e-1, depth = 0 , thickness = 0, plot = False):

    # Source properties
    source_location = np.array([0.0, 0.0, 30.0])
    source_orientation = "z"  # "x", "y" or "z"
    source_current = 1.0  # maximum on-time current
    source_radius = 6.0  # source loop radius

    # Receiver properties
    receiver_location = np.array([0.0, 0.0, 30.0])
    receiver_orientation = "z"  # "x", "y" or "z"
    times = np.logspace(np.log10(2e-5), np.log10(8e-3), 20)# time channels (s)

    # Define receiver list. In our case, we have only a single receiver for each source.
    # When simulating the response for multiple component and/or field orientations,
    # the list consists of multiple receiver objects.
    receiver_list = []
    receiver_list.append(
        tdem.receivers.PointMagneticFluxDensity(
            receiver_location, times, orientation=receiver_orientation
        )
    )

    # Define the source waveform. Here we define a unit step-off. The definition of
    # other waveform types is covered in a separate tutorial.
    waveform = tdem.sources.StepOffWaveform()

    # Define source list. In our case, we have only a single source.
    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=receiver_list,
            location=source_location,
            waveform=waveform,
            current=source_current,
            radius=source_radius,
        )
    ]

    # Define the survey
    survey = tdem.Survey(source_list)


    # Physical properties: Electrical conductivity Siemens per meter (S/m)
    background_conductivity = 1e-1
    layer_conductivity = conductivity

    # Layer Location,Layer thicknesses
    thicknesses = np.array([depth, thickness])
    n_layer = len(thicknesses) + 1

    # physical property model (conductivity model)
    model = background_conductivity * np.ones(n_layer)
    model[1] = layer_conductivity

    # Define a mapping from model parameters to conductivities
    model_mapping = maps.IdentityMap(nP=n_layer)

    # Plot conductivity model, max_depth

    #Calculate max depth
    max_depth = 600 - (depth + thickness)

    thicknesses_for_plotting = np.r_[thicknesses, max_depth]
    mesh_for_plotting = TensorMesh([thicknesses_for_plotting])

    # Plot model if wanted
    if plot:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
        plot_1d_layer_model(thicknesses_for_plotting, model, ax=ax, show_layers=False)
        plt.show()


    return times, survey, thicknesses, model_mapping, model


def forward_TDEM(times, survey, thicknesses, model_mapping, model, noise_percentage, plot = False):
    # Define the simulation
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=model_mapping,
    )

    # Predict sounding data
    dpred = simulation.dpred(model)

    # Add gausian random noise
    noise = np.random.normal(0, noise_percentage * np.abs(dpred), size=dpred.shape)
    dpred = dpred + noise


    # Plot sounding data
    if plot: 
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.2, 0.15, 0.75, 0.78])
        ax.loglog(times, dpred, "k-o", lw=2)
        ax.set_xlabel("Times (s)")
        ax.set_ylabel("|B| (T)")
        ax.set_title("Magnetic Flux")

    return dpred


def process_confounders_TDEM(confounders, noise_percentage=0.05, plot = False):
    """
    Processes the confounders by creating models, running forward FDEM, 
    and calculating RMS errors.
    
    Parameters:
    - confounders: A 2D array (n, 3) where each row contains conductivity, depth, and thickness.
    - noise_percentage: The noise percentage to be applied in the forward FDEM function (default is 0.05).
    
    Returns:
    - results: A 2D array (n, 15) containing the results with dpred_real, dpred_imag, and RMS values.
    """
    
    # Create model with no anomaly
    times_0, survey_0, thicknesses_0, model_mapping_0, model_0 = create_TDEMmodel(depth=50, thickness=20, plot=plot)
    dpred_0 = forward_TDEM(times_0, survey_0, thicknesses_0, model_mapping_0, model_0, noise_percentage=noise_percentage)

    # Initialize the results array
    results = np.zeros((confounders.shape[0], 21))

    # Start time recording
    start = time.time()

    # Loop through the confounders
    for i, (conductivity, depth, thickness) in enumerate(confounders):
        # Create model with the given conductivity, depth, and thickness
        frequencies, survey, thicknesses, model_mapping, model = create_TDEMmodel(conductivity=conductivity, depth=depth, thickness=thickness)
        
        # Run the forward model
        dpred_n = forward_TDEM(frequencies, survey, thicknesses, model_mapping, model, noise_percentage=noise_percentage)

        # Calculate RMS error
        rms_n = calculate_rms_difference(dpred_n, dpred_0, noise_percentage)

        # Save the results
        results[0, :20] = dpred_n
        results[i, 20] = rms_n

        # Print elapsed time every 10,000 iterations
        if (i + 1) % 10000 == 0:
            elapsed_total = time.time() - start
            print(f"Iteration {i + 1}: Elapsed time = {elapsed_total:.2f} seconds")

    return results




