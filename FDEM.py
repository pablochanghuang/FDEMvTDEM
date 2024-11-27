import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh
import time

from simpeg import maps
from simpeg.electromagnetics import frequency_domain as fdem
from simpeg.utils import plot_1d_layer_model

#Set Solver
#from simpeg.utils.solver_utils import get_default_solver
#Solver = get_default_solver()

plt.rcParams.update({"font.size": 16})
write_output = False


def calculate_rms_difference(real_with_layer, imag_with_layer, real_without_layer, imag_without_layer, noise_percentage):
    """
    Calculate the RMS difference between two datasets (with and without a layer).
    
    Parameters:
    - real_with_layer: numpy array of real parts with the layer.
    - imag_with_layer: numpy array of imaginary parts with the layer.
    - real_without_layer: numpy array of real parts without the layer.
    - imag_without_layer: numpy array of imaginary parts without the layer.
    
    Returns:
    - rms_real: RMS difference for the real parts.
    - rms_imag: RMS difference for the imaginary parts.
    - rms_combined: Combined RMS difference.
    """
    #Divide by the standard deviation

    #Calculate differences
    delta_real = (real_with_layer - real_without_layer)/(np.abs(real_with_layer)*noise_percentage)
    delta_imag = (imag_with_layer - imag_without_layer)/(np.abs(imag_with_layer)*noise_percentage)

    #Calculate RMS for each
    rms_real = np.sqrt(np.mean(delta_real**2))
    rms_imag = np.sqrt(np.mean(delta_imag**2))

    # Step 3: Combine Real and Imaginary RMS
    rms_combined = np.sqrt(rms_real**2 + rms_imag**2)

    return rms_real, rms_imag, rms_combined



def create_FDEMmodel(conductivity = 1e-1, depth = 0 , thickness = 0, plot = False):

    # Frequencies being observed in Hz
    frequencies = np.array([400, 1800, 3300, 8200, 40000, 140000], dtype=float)

    # Define a list of receivers
    receiver_location = np.array([7.86, 0.0, 30.0])
    receiver_orientation = "z"  # "x", "y" or "z"
    data_type = "ppm"  # "secondary", "total" or "ppm"

    receiver_list = []
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location,
            orientation=receiver_orientation,
            data_type=data_type,
            component="real",
        )
    )
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location,
            orientation=receiver_orientation,
            data_type=data_type,
            component="imag",
        )
    )

    # Define the source list. 
    source_location = np.array([0.0, 0.0, 30.0])
    source_orientation = "z"  # "x", "y" or "z"
    moment = 1.0  # dipole moment

    source_list = []
    for freq in frequencies:
        source_list.append(
            fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=freq,
                location=source_location,
                orientation=source_orientation,
                moment=moment,
            )
        )

    # Define a 1D FDEM survey
    survey = fdem.survey.Survey(source_list)

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


    if plot:

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
        plot_1d_layer_model(thicknesses_for_plotting, model, ax=ax, show_layers=False)
        plt.show()


    return frequencies, survey, thicknesses, model_mapping, model


def forward_FDEM(frequencies, survey, thicknesses, model_mapping, model, noise_percentage, plot = False):
    # Define the simulation
    simulation = fdem.Simulation1DLayered(
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
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
        ax.semilogx(frequencies, np.abs(dpred[0::2]), "k-o", lw=3, ms=10)
        ax.semilogx(frequencies, np.abs(dpred[1::2]), "k:o", lw=3, ms=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|Hs/Hp| (ppm)")
        ax.set_title("Secondary Magnetic Field as ppm")
        ax.legend(["Real", "Imaginary"])
        plt.show()

    return dpred[0::2], dpred[1::2]


def process_confounders_FDEM(confounders, noise_percentage=0.05, plot = False):
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
    frequencies_0, survey_0, thicknesses_0, model_mapping_0, model_0 = create_FDEMmodel(depth=50, thickness=20, plot=plot)
    dpred_real_0, dpred_imag_0 = forward_FDEM(frequencies_0, survey_0, thicknesses_0, model_mapping_0, model_0, noise_percentage=noise_percentage)

    # Initialize the results array
    results = np.zeros((confounders.shape[0], 15))

    # Start time recording
    start = time.time()

    # Loop through the confounders
    for i, (conductivity, depth, thickness) in enumerate(confounders):
        # Create model with the given conductivity, depth, and thickness
        frequencies, survey, thicknesses, model_mapping, model = create_FDEMmodel(conductivity=conductivity, depth=depth, thickness=thickness)
        
        # Run the forward model
        dpred_real_n, dpred_imag_m = forward_FDEM(frequencies, survey, thicknesses, model_mapping, model, noise_percentage=noise_percentage)

        # Calculate RMS error
        rms_real_n, rms_imag_n, rms_combined_n = calculate_rms_difference(dpred_real_n, dpred_imag_m, dpred_real_0, dpred_imag_0, noise_percentage)

        # Save the results
        results[0, :6] = dpred_real_n
        results[0, 6:12] = dpred_imag_m
        results[i, 12] = rms_real_n
        results[i, 13] = rms_imag_n
        results[i, 14] = rms_combined_n

        # Print elapsed time every 10,000 iterations
        if (i + 1) % 10000 == 0:
            elapsed_total = time.time() - start
            print(f"Iteration {i + 1}: Elapsed time = {elapsed_total:.2f} seconds")

    return results





#Call function to create the model

#Start time recording
#start = time.time()
#frequencies, survey, thicknesses, model_mapping, model = create_FDEMmodel(depth = 50, thickness = 20)
#end = time.time()
#print("Create model: ", end - start)

#Call function to run the forward model
#start = time.time()
#dpred_real, dpred_imag = forward_FDEM(frequencies, survey, thicknesses, model_mapping, model, noise_percentage = 0.05)
#end = time.time()
#print("Run model: ", end - start)



