# -*- coding: utf-8 -*-
"""
pysteps.blending.steps_mpi
======================

Implementation of the STEPS stochastic blending method as described in
:cite:`BPS2004`, :cite:`BPS2006` and :cite:`SPN2013`, with additional changes for MPI-parallellism
The STEPS blending method consists of the following main steps:

    #. Set the radar rainfall fields in a Lagrangian space.
    #. Initialize the noise method.
    #. Perform the cascade decomposition for the input radar rainfall fields.
       The method assumes that the cascade decomposition of the NWP model fields is
       already done prior to calling the function, as the NWP model fields are
       generally not updated with the same frequency (which is more efficient). A
       method to decompose and store the NWP model fields whenever a new NWP model
       field is present, is present in pysteps.blending.utils.decompose_NWP.
    #. Estimate AR parameters for the extrapolation nowcast and noise cascade.
    #. Initialize all the random generators.
    #. Calculate the initial skill of the NWP model forecasts at t=0.
    #. Start the forecasting loop:
        #. Determine which NWP models will be combined with which nowcast ensemble
           member. The number of output ensemble members equals the maximum number
           of (ensemble) members in the input, which can be either the defined
           number of (nowcast) ensemble members or the number of NWP models/members.
        #. Determine the skill and weights of the forecasting components
           (extrapolation, NWP and noise) for that lead time.
        #. Regress the extrapolation and noise cascades separately to the subsequent
           time step.
        #. Extrapolate the extrapolation and noise cascades to the current time step.
        #. Blend the cascades.
        #. Recompose the cascade to a rainfall field.
        #. Post-processing steps (masking and probability matching, which are
           different from the original blended STEPS implementation).

.. autosummary::
    :toctree: ../generated/

    forecast
    calculate_ratios
    calculate_weights_bps
    calculate_weights_spn
    blend_means_sigmas
"""

import time

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure


from pysteps import cascade
from pysteps import extrapolation
from pysteps import noise
from pysteps import utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps import blending

from pysteps.blending.steps import _transform_to_lagrangian
from pysteps.blending.steps import _init_noise
from pysteps.blending.steps import _compute_cascade_decomposition_radar
from pysteps.blending.steps import _compute_cascade_decomposition_nwp
from pysteps.blending.steps import _estimate_ar_parameters_radar
from pysteps.blending.steps import _init_random_generators
from pysteps.blending.steps import _prepare_forecast_loop
from pysteps.blending.steps import calculate_weights_bps
from pysteps.blending.steps import blend_means_sigmas
from pysteps.blending.steps import _compute_incremental_mask
from pysteps.blending.steps import _find_nwp_combination
from pysteps.blending.steps import _compute_initial_nwp_skill
from pysteps.blending.steps import _fill_nans_infs_nwp_cascade
from pysteps.blending.steps import _determine_max_nr_rainy_cells_nwp
from pysteps.blending.steps import _init_noise_cascade

def forecast(
    comm,
    root,
    precip,
    precip_models,
    velocity,
    velocity_models,
    precip_shape,
    timesteps,
    timestep,
    issuetime,
    n_ens_members,
    n_nwp_members,
    n_cascade_levels=8,
    blend_nwp_members=False,
    precip_thr=None,
    norain_thr=0.0,
    kmperpixel=None,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    noise_stddev_adj=None,
    ar_order=2,
    vel_pert_method="bps",
    weights_method="bps",
    conditional=False,
    probmatching_method="cdf",
    mask_method="incremental",
    callback=None,
    return_output=True,
    seed=None,
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    outdir_path_skill="./tmp/",
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    vel_pert_kwargs=None,
    clim_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
):
    """
    Generate a blended nowcast ensemble by using the Short-Term Ensemble
    Prediction System (STEPS) method.

    Parameters
    ----------
    comm: mpi4py.MPI.COMM_WOLD attribute
      The MPI-communicator
    root: int
      root process, which will handle the gathering and scattering of the data
    precip: array-like
      Array of shape (ar_order+1,m,n) containing the input precipitation fields
      ordered by timestamp from oldest to newest. The time steps between the
      inputs are assumed to be regular.
    precip_models: array-like
      Array of shape (n_models,timesteps+1) containing, per timestep (t=0 to
      lead time here) and per (NWP) model or model ensemble member, a
      dictionary with a list of cascades obtained by calling a method
      implemented in :py:mod:`pysteps.cascade.decomposition`. In case of one
      (deterministic) model as input, add an extra dimension to make sure
      precip_models is five dimensional prior to calling this function.
      It is also possible to supply the original (NWP) model forecasts containing
      rainfall fields as an array of shape (n_models,timestep+1,m,n), which will
      then be decomposed in this function. Note that for an operational application
      or for testing with multiple model runs, it is recommended to decompose
      the model forecasts outside beforehand, as this reduces calculation times.
      This is possible with :py:func:`pysteps.blending.utils.decompose_NWP`,
      :py:func:`pysteps.blending.utils.compute_store_nwp_motion`, and
      :py:func:`pysteps.blending.utils.load_NWP`.
    velocity: array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection
      field. The velocities are assumed to represent one time step between the
      inputs. All values are required to be finite.
    velocity_models: array-like
      Array of shape (n_models,timestep,2,m,n) containing the x- and y-components
      of the advection field for the (NWP) model field per forecast lead time.
      All values are required to be finite.
    precip_shape = tuple
      tuple with two elements describing the shape of 1 radar field (number of x- and y-pixels)
    timesteps: int or list of floats
      Number of time steps to forecast or a list of time steps for which the
      forecasts are computed (relative to the input time step). The elements of
      the list are required to be in ascending order.
    timestep: float
      Time step of the motion vectors (minutes). Required if vel_pert_method is
      not None or mask_method is 'incremental'.
    issuetime: datetime
      Datetime object containing the date and time for which the forecast
      is issued.
    n_ens_members: int
      The number of ensemble members to generate. This number should always be
      equal to or larger than the number of NWP ensemble members / number of
      NWP models.
    n_nwp_members: int
      The number of NWP members, this can no longer be deduced since it is
      possible that different members live on different processess.
    n_cascade_levels: int, optional
      The number of cascade levels to use. Default set to 8 due to default
      climatological skill values on 8 levels.
    blend_nwp_members: bool
      Check if NWP models/members should be used individually, or if all of
      them are blended together per nowcast ensemble member. Standard set to
      false.
    precip_thr: float, optional
      Specifies the threshold value for minimum observable precipitation
      intensity. Required if mask_method is not None or conditional is True.
    kmperpixel: float, optional
      Spatial resolution of the input data (kilometers/pixel). Required if
      vel_pert_method is not None or mask_method is 'incremental'.
    extrap_method: str, optional
      Name of the extrapolation method to use. See the documentation of
      :py:mod:`pysteps.extrapolation.interface`.
    decomp_method: {'fft'}, optional
      Name of the cascade decomposition method to use. See the documentation
      of :py:mod:`pysteps.cascade.interface`.
    bandpass_filter_method: {'gaussian', 'uniform'}, optional
      Name of the bandpass filter method to use with the cascade decomposition.
      See the documentation of :py:mod:`pysteps.cascade.interface`.
    noise_method: {'parametric','nonparametric','ssft','nested',None}, optional
      Name of the noise generator to use for perturbating the precipitation
      field. See the documentation of :py:mod:`pysteps.noise.interface`. If set to None,
      no noise is generated.
    noise_stddev_adj: {'auto','fixed',None}, optional
      Optional adjustment for the standard deviations of the noise fields added
      to each cascade level. This is done to compensate incorrect std. dev.
      estimates of casace levels due to presence of no-rain areas. 'auto'=use
      the method implemented in :py:func:`pysteps.noise.utils.compute_noise_stddev_adjs`.
      'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
      noise std. dev adjustment.
    ar_order: int, optional
      The order of the autoregressive model to use. Must be >= 1.
    vel_pert_method: {'bps',None}, optional
      Name of the noise generator to use for perturbing the advection field. See
      the documentation of :py:mod:`pysteps.noise.interface`. If set to None, the advection
      field is not perturbed.
    weights_method: {'bps','spn'}, optional
      The calculation method of the blending weights. Options are the method
      by :cite:`BPS2006` and the covariance-based method by :cite:`SPN2013`.
      Defaults to bps.
    conditional: bool, optional
      If set to True, compute the statistics of the precipitation field
      conditionally by excluding pixels where the values are below the threshold
      precip_thr.
    mask_method: {'obs','incremental',None}, optional
      The method to use for masking no precipitation areas in the forecast field.
      The masked pixels are set to the minimum value of the observations.
      'obs' = apply precip_thr to the most recently observed precipitation intensity
      field, 'incremental' = iteratively buffer the mask with a certain rate
      (currently it is 1 km/min), None=no masking.
    probmatching_method: {'cdf','mean',None}, optional
      Method for matching the statistics of the forecast field with those of
      the most recently observed one. 'cdf'=map the forecast CDF to the observed
      one, 'mean'=adjust only the conditional mean value of the forecast field
      in precipitation areas, None=no matching applied. Using 'mean' requires
      that mask_method is not None.
    callback: function, optional
      Optional function that is called after computation of each time step of
      the nowcast. The function takes one argument: a three-dimensional array
      of shape (n_ens_members,h,w), where h and w are the height and width
      of the input field precip, respectively. This can be used, for instance,
      writing the outputs into files.
    return_output: bool, optional
      Set to False to disable returning the outputs as numpy arrays. This can
      save memory if the intermediate results are written to output files using
      the callback function.
    seed: int, optional
      Optional seed number for the random generators.
    num_workers: int, optional
      The number of workers to use for parallel computation. Applicable if dask
      is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
      is advisable to disable OpenMP by setting the environment variable
      OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
      threads.
    fft_method: str, optional
      A string defining the FFT method to use (see FFT methods in
      :py:func:`pysteps.utils.interface.get_method`).
      Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
      the recommended method is 'pyfftw'.
    domain: {"spatial", "spectral"}
      If "spatial", all computations are done in the spatial domain (the
      classical STEPS model). If "spectral", the AR(2) models and stochastic
      perturbations are applied directly in the spectral domain to reduce
      memory footprint and improve performance :cite:`PCH2019b`.
    outdir_path_skill: string, optional
      Path to folder where the historical skill are stored. Defaults to
      path_workdir from rcparams. If no path is given, './tmp' will be used.
    extrap_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the extrapolation
      method. See the documentation of :py:func:`pysteps.extrapolation.interface`.
    filter_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the filter method.
      See the documentation of :py:mod:`pysteps.cascade.bandpass_filters`.
    noise_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the initializer of
      the noise generator. See the documentation of :py:mod:`pysteps.noise.fftgenerators`.
    vel_pert_kwargs: dict, optional
      Optional dictionary containing keyword arguments 'p_par' and 'p_perp' for
      the initializer of the velocity perturbator. The choice of the optimal
      parameters depends on the domain and the used optical flow method.
      Default parameters from :cite:`BPS2006`:
      p_par  = [10.88, 0.23, -7.68]
      p_perp = [5.76, 0.31, -2.72]

      Parameters fitted to the data (optical flow/domain):

      darts/fmi:
      p_par  = [13.71259667, 0.15658963, -16.24368207]
      p_perp = [8.26550355, 0.17820458, -9.54107834]

      darts/mch:
      p_par  = [24.27562298, 0.11297186, -27.30087471]
      p_perp = [-7.80797846e+01, -3.38641048e-02, 7.56715304e+01]

      darts/fmi+mch:
      p_par  = [16.55447057, 0.14160448, -19.24613059]
      p_perp = [14.75343395, 0.11785398, -16.26151612]

      lucaskanade/fmi:
      p_par  = [2.20837526, 0.33887032, -2.48995355]
      p_perp = [2.21722634, 0.32359621, -2.57402761]

      lucaskanade/mch:
      p_par  = [2.56338484, 0.3330941, -2.99714349]
      p_perp = [1.31204508, 0.3578426, -1.02499891]

      lucaskanade/fmi+mch:
      p_par  = [2.31970635, 0.33734287, -2.64972861]
      p_perp = [1.90769947, 0.33446594, -2.06603662]

      vet/fmi:
      p_par  = [0.25337388, 0.67542291, 11.04895538]
      p_perp = [0.02432118, 0.99613295, 7.40146505]

      vet/mch:
      p_par  = [0.5075159, 0.53895212, 7.90331791]
      p_perp = [0.68025501, 0.41761289, 4.73793581]

      vet/fmi+mch:
      p_par  = [0.29495222, 0.62429207, 8.6804131 ]
      p_perp = [0.23127377, 0.59010281, 5.98180004]

      fmi=Finland, mch=Switzerland, fmi+mch=both pooled into the same data set

      The above parameters have been fitten by using run_vel_pert_analysis.py
      and fit_vel_pert_params.py located in the scripts directory.

      See :py:mod:`pysteps.noise.motion` for additional documentation.
    clim_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the climatological
      skill file. Arguments can consist of: 'outdir_path', 'n_models'
      (the number of NWP models) and 'window_length' (the minimum number of
      days the clim file should have, otherwise the default is used).
    mask_kwargs: dict
      Optional dictionary containing mask keyword arguments 'mask_f' and
      'mask_rim', the factor defining the the mask increment and the rim size,
      respectively.
      The mask increment is defined as mask_f*timestep/kmperpixel.
    measure_time: bool
      If set to True, measure, print and return the computation time.

    Returns
    -------
    out: ndarray
      If return_output is True, a four-dimensional array of shape
      (n_ens_members,num_timesteps,m,n) containing a time series of forecast
      precipitation fields for each ensemble member. Otherwise, a None value
      is returned. The time series starts from t0+timestep, where timestep is
      taken from the input precipitation fields precip. If measure_time is True, the
      return value is a three-element tuple containing the nowcast array, the
      initialization time of the nowcast generator and the time used in the
      main loop (seconds).

    See also
    --------
    :py:mod:`pysteps.extrapolation.interface`, :py:mod:`pysteps.cascade.interface`,
    :py:mod:`pysteps.noise.interface`, :py:func:`pysteps.noise.utils.compute_noise_stddev_adjs`

    References
    ----------
    :cite:`Seed2003`, :cite:`BPS2004`, :cite:`BPS2006`, :cite:`SPN2013`, :cite:`PCH2019b`

    Notes
    -----
    1. The blending currently does not blend the beta-parameters in the parametric
    noise method. It is recommended to use the non-parameteric noise method.

    2. If blend_nwp_members is True, the BPS2006 method for the weights is
    suboptimal. It is recommended to use the SPN2013 method instead.

    3. Not yet implemented (and neither in the steps nowcasting module): The regression
    of the lag-1 and lag-2 parameters to their climatological values. See also eq.
    12 - 19 in :cite: `BPS2004`. By doing so, the Phi parameters change over time,
    which enhances the AR process. This can become a future development if this
    turns out to be a warranted functionality.
    """
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    # 0.1 Start with some checks
    _check_inputs(precip, precip_models, velocity, velocity_models, timesteps, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()

    if filter_kwargs is None:
        filter_kwargs = dict()

    if noise_kwargs is None:
        noise_kwargs = dict()

    if vel_pert_kwargs is None:
        vel_pert_kwargs = dict()

    if clim_kwargs is None:
        # Make sure clim_kwargs at least contains the number of models
        clim_kwargs = dict({"n_models": n_nwp_members})

    if mask_kwargs is None:
        mask_kwargs = dict()

    if velocity is not None:
        if np.any(~np.isfinite(velocity)):
            raise ValueError("velocity contains non-finite values")

    if mask_method not in ["obs", "incremental", None]:
        raise ValueError(
            "unknown mask method %s: must be 'obs', 'incremental' or None" % mask_method
        )

    if conditional and precip_thr is None:
        raise ValueError("conditional=True but precip_thr is not set")

    if mask_method is not None and precip_thr is None:
        raise ValueError("mask_method!=None but precip_thr=None")

    if noise_stddev_adj not in ["auto", "fixed", None]:
        raise ValueError(
            "unknown noise_std_dev_adj method %s: must be 'auto', 'fixed', or None"
            % noise_stddev_adj
        )

    if kmperpixel is None:
        if vel_pert_method is not None:
            raise ValueError("vel_pert_method is set but kmperpixel=None")
        if mask_method == "incremental":
            raise ValueError("mask_method='incremental' but kmperpixel=None")

    if timestep is None:
        if vel_pert_method is not None:
            raise ValueError("vel_pert_method is set but timestep=None")
        if mask_method == "incremental":
            raise ValueError("mask_method='incremental' but timestep=None")

    # 0.2 Log some settings
    if rank == root:
        print("STEPS blending",flush=True)
        print("==============",flush=True)
        print("",flush=True)
    
        print("Inputs",flush=True)
        print("------",flush=True)
        print(f"forecast issue time:         {issuetime.isoformat()}",flush=True)
        print(f"input dimensions:            {precip.shape[1]}x{precip.shape[2]}",flush=True)
        if kmperpixel is not None:
            print(f"km/pixel:                    {kmperpixel}",flush=True)
        if timestep is not None:
            print(f"time step:                   {timestep} minutes",flush=True)
        print("")
    
        print("NWP and blending inputs",flush=True)
        print("-----------------------",flush=True)
        print(f"number of (NWP) models:      {n_nwp_members}",flush=True)
        print(f"blend (NWP) model members:   {blend_nwp_members}",flush=True)
##TODO: how to check already here if NWP needs decomposition
        print(f"decompose (NWP) models:      {'no'}",flush=True)
        print("",flush=True)
    
        print("Methods",flush=True)
        print("-------",flush=True)
        print(f"extrapolation:               {extrap_method}",flush=True)
        print(f"bandpass filter:             {bandpass_filter_method}",flush=True)
        print(f"decomposition:               {decomp_method}",flush=True)
        print(f"noise generator:             {noise_method}",flush=True)
        print(f"noise adjustment:            {'yes' if noise_stddev_adj else 'no'}",flush=True)
        print(f"velocity perturbator:        {vel_pert_method}",flush=True)
        print(f"blending weights method:     {weights_method}",flush=True)
        print(f"conditional statistics:      {'yes' if conditional else 'no'}",flush=True)
        print(f"precip. mask method:         {mask_method}",flush=True)
        print(f"probability matching:        {probmatching_method}",flush=True)
        print(f"FFT method:                  {fft_method}",flush=True)
        print(f"domain:                      {domain}",flush=True)
        print("",flush=True)
    
        print("Parameters",flush=True)
        print("----------",flush=True)
        if isinstance(timesteps, int):
            print(f"number of time steps:        {timesteps}",flush=True)
        else:
            print(f"time steps:                  {timesteps}",flush=True)
        print(f"ensemble size:               {n_ens_members}",flush=True)
        print(f"parallel threads:            {num_workers}",flush=True)
        print(f"number of cascade levels:    {n_cascade_levels}",flush=True)
        print(f"order of the AR(p) model:    {ar_order}",flush=True)
    if vel_pert_method == "bps":
        vp_par = vel_pert_kwargs.get("p_par", noise.motion.get_default_params_bps_par())
        vp_perp = vel_pert_kwargs.get(
            "p_perp", noise.motion.get_default_params_bps_perp()
        )
        if rank == root:
            print(f"vel. pert., parallel:        {vp_par[0]},{vp_par[1]},{vp_par[2]}",flush=True)
            print(f"vel. pert., perpendicular:   {vp_perp[0]},{vp_perp[1]},{vp_perp[2]}",flush=True)
    else:
        vp_par, vp_perp = None, None

    if rank == root:
        if conditional or mask_method is not None:
            print(f"precip. intensity threshold: {precip_thr}",flush=True)
        print("",flush=True)
        print("MPI parameters",flush=True)
        print("--------------",flush=True)
        print(f"number of processes:     {nproc}",flush=True)
        print(f"root process:            {rank}",flush=True)       
        print("",flush=True)


    # 0.3 Get the methods that will be used
    if measure_time:
        starttime_init = time.time()
    fft = utils.get_method(fft_method, shape=precip_shape, n_threads=1)
    
    # - initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    bp_filter = filter_method(precip_shape, n_cascade_levels, **filter_kwargs)

    # - get the decomposition method
    decompositor, recompositor = cascade.get_method(decomp_method)
    
    # - get the extrolation method
    extrapolator = extrapolation.get_method(extrap_method)

    # - we need all the nwp models on the root rank for the no rain cases
    precip_models, velocity_models = _send_receive_nwp(
        precip_models, velocity_models, timestep=-1, comm=comm,timesteps=timesteps,root=root)
    
    # - only needed for root rank
    if rank == root:
        # 1.1 prepare for move to Lagrangian space
        x_values, y_values = np.meshgrid(
            np.arange(precip.shape[2]), np.arange(precip.shape[1])
            )
        xy_coords = np.stack([x_values, y_values])

        domain_mask = np.logical_or.reduce(
            [~np.isfinite(precip[i,:,:]) for i in range(precip.shape[0])]
            )

        if conditional:
            MASK_thr = np.logical_and.reduce(
                [precip[i, :, :] >= precip_thr for i in range(precip.shape[0])]
                )
        else:
            MASK_thr = None

        # we need to know the zerovalue of precip to replace the mask when decomposing after extrapolation
        zerovalue = np.nanmin(precip)


        # 1.2 move to Lagrangian space
        precip = _transform_to_lagrangian(
            precip, velocity, ar_order, xy_coords, extrapolator, extrap_kwargs, 1
            )

        # 2. Perform the cascade decomposition for the input precip fields and
        # if necessary, for the (NWP) model fields
        
        # 2.1 Compute the cascade decompositions of the input precipitation fields
        (
            precip_cascade,
            mu_extrapolation,
            sigma_extrapolation,
        ) = _compute_cascade_decomposition_radar(
            precip,
            ar_order,
            n_cascade_levels,
            n_ens_members,
            MASK_thr,
            domain,
            bp_filter,
            decompositor,
            fft,
            )
            
        # 2.2 If necessary, decompose (NWP) model forecasts and stack cascades
        (
            precip_models_cascade,
            mu_models,
            sigma_models,
            precip_models_pm,
        ) = _compute_cascade_decomposition_nwp(
            precip_models, bp_filter, decompositor, recompositor, fft, domain
        )
        # 2.3 Check for zero input fields in the radar and NWP data.
        zero_precip_radar = blending.utils.check_norain(precip, precip_thr, norain_thr)
        # The norain fraction threshold used for nwp is the default value of 0.0,
        # since nwp does not suffer from clutter.
        zero_model_fields = blending.utils.check_norain(precip_models_pm, precip_thr)
    # All other ranks
    else:
        precip = None
        zerovalue = None
        zero_precip_radar = None 
        zero_model_fields = None
        
    # All ranks    
    precip = broadcast(precip,comm)
    zerovalue = broadcast(zerovalue,comm)
    zero_precip_radar = broadcast(zero_precip_radar,comm)
    zero_model_fields = broadcast(zero_model_fields,comm)
    
    #  calculate the number of members on the local rank
    n_local_ens_members = len(np.array_split(np.ones(n_ens_members),nproc)[rank])
    if zero_precip_radar and zero_model_fields:
        if rank == root:
            print(
                "No precipitation above the threshold found in both the radar and NWP fields"
                )
            print("The resulting forecast will contain only zeros")
        # Create the output list

        
        if isinstance(timesteps, int):
            timesteps = range(timesteps + 1)
            timestep_type = "int"
        else:
            original_timesteps = [0] + list(timesteps)
            timesteps = nowcast_utils.binned_timesteps(original_timesteps)
            timestep_type = "list"
            
        # Save per time step to ensure the array does not become too large if
        # no return_output is requested and callback is not None.
        for t, subtimestep_idx in enumerate(timesteps):
            # Create an empty np array with shape [n_local_ens_members, rows, cols]
            # and fill it with the minimum value from precip (corresponding to
            # zero precipitation)
            R_f_ = np.full(
                (n_local_ens_members, precip_shape[0], precip_shape[1]), np.nanmin(precip)
            )
            if callback is not None:
                if R_f_.shape[1] > 0:
                    callback(R_f_.squeeze())
#            if return_output:
#                for j in range(n_ens_members):
#                    R_f[j].append(R_f_[j])

            R_f_ = None

        if measure_time:
            zero_precip_time = time.time() - starttime_init
        return None
    
    else:
        if rank == root:
            # 2.3.3 Check if the NWP fields contain nans or infinite numbers. If so,
            # fill these with the minimum value present in precip (corresponding to
            # zero rainfall in the radar observations)
            (
                precip_models_cascade,
                precip_models_pm,
                mu_models,
                sigma_models,
            ) = _fill_nans_infs_nwp_cascade(
                precip_models_cascade,
                precip_models_pm,
                precip_cascade,
                precip,
                mu_models,
                sigma_models,
            )
                
            # 2.3.4 If zero_precip_radar is True, only use the velocity field of the NWP
            # forecast. I.e., velocity (radar) equals velocity_model at the first time
            # step.
            if zero_precip_radar:
                # Use the velocity from velocity_models at time step 0
                velocity = velocity_models[:, 0, :, :, :]
                # Take the average over the first axis, which corresponds to n_models
                # (hence, the model average)
                velocity = np.mean(velocity, axis=0)
                
            # 3. Initialize the noise method.
            # If zero_precip_radar is True, initialize noise based on the NWP field time
            # step where the fraction of rainy cells is highest (because other lead times
            # might be zero as well). Else, initialize the noise with the radar
            # rainfall data
            if zero_precip_radar:
                precip_noise_input = _determine_max_nr_rainy_cells_nwp(
                    precip_models_pm, precip_thr, precip_models_pm.shape[0], timesteps
                )
                # Make sure precip_noise_input is three dimensional
                precip_noise_input = precip_noise_input[np.newaxis, :, :]
            else:
                precip_noise_input = precip.copy()

            pp, generate_noise, noise_std_coeffs = _init_noise(
                precip_noise_input,
                precip_thr,
                n_cascade_levels,
                bp_filter,
                decompositor,
                fft,
                noise_method,
                noise_kwargs,
                noise_stddev_adj,
                measure_time,
                num_workers,
            )
            precip_noise_input = None
            
            # 4. Estimate AR parameters for the radar rainfall field
            PHI = _estimate_ar_parameters_radar(
                precip_cascade,
                ar_order,
                n_cascade_levels,
                MASK_thr,
                zero_precip_radar,
            )
                
            # 5. Initialize all the random generators and prepare for the forecast loop
            randgen_prec, vps, generate_vel_noise = _init_random_generators(
                velocity,
                noise_method,
                vel_pert_method,
                vp_par,
                vp_perp,
                seed,
                n_ens_members,
                kmperpixel,
                timestep,
            )
            

        # 6 if not root rank, initialize all variables that need to be broadcasted
        else:
            velocity = None
            domain_mask = None
            precip_cascade = None
            mu_extrapolation = None
            sigma_extrapolation = None
            pp =  None
            generate_noise = None
            noise_std_coeffs =  None
            PHI = None
            randgen_prec = None
            MASK_thr = None
        

        # 6.1 Broadcast and scatter the variables 
        # - broadcast
        velocity = broadcast(velocity,comm)
        domain_mask = broadcast(domain_mask,comm)
        precip_cascade = broadcast(precip_cascade,comm)
        mu_extrapolation = broadcast(mu_extrapolation,comm)
        sigma_extrapolation = broadcast(sigma_extrapolation,comm)
        pp = broadcast(pp,comm)
        generate_noise = broadcast(generate_noise,comm)
        noise_std_coeffs = broadcast(noise_std_coeffs,comm)
        PHI = broadcast(PHI,comm)
        MASK_thr = broadcast(MASK_thr,comm)
        # - scatter
        randgen_prec = scatter(randgen_prec,n_ens_members,comm)

        # 7. Repeat the precip_cascade for the amount of ensemble members needed on the rank
        # - discard all except the p-1 last cascades because they are not needed
        # for the AR(p) model
        precip_cascade = [precip_cascade[i][-ar_order:] for i in range(n_cascade_levels)]
        precip_cascade = [
            [precip_cascade[j].copy() for j in range(n_cascade_levels)]
            for i in range(n_local_ens_members)
            ]
        precip_cascade = np.stack(precip_cascade)

    
        # 8.1 Prepare forecast loop
        D, D_Yn, D_pb, R_f, R_m, mask_rim, struct, fft_objs = _prepare_forecast_loop(
            precip_cascade,
            noise_method,
            fft_method,
            n_cascade_levels,
            n_local_ens_members,
            mask_method,
            mask_kwargs,
            timestep,
            kmperpixel,
        )
        
        # 8.2 Also initialize the cascade of temporally correlated noise, which has the
        # same shape as precip_cascade, but starts random noise.
        noise_cascade, mu_noise, sigma_noise = _init_noise_cascade(
            shape=precip_cascade.shape,
            n_ens_members=n_local_ens_members,
            n_cascade_levels=n_cascade_levels,
            generate_noise=generate_noise,
            decompositor=decompositor,
            pp=pp,
            randgen_prec=randgen_prec,
            fft_objs=fft_objs,
            bp_filter=bp_filter,
            domain=domain,
            noise_method=noise_method,
            noise_std_coeffs=noise_std_coeffs,
            ar_order=ar_order,
        )

        ## TODO can we do this before the broadcast?
        precip = precip[-1,:,:]

        # 8.2 Initizalize the current and previous extrapolation forecast scale
        # for the nowcasting component
        rho_extr_prev = np.repeat(1.0, PHI.shape[0])
        rho_extr = PHI[:, 0] / (1.0 - PHI[:, 1])  # phi1 / (1 - phi2), see BPS2004
    
        #################################
        #   Start the forecasting loop  #
        #################################
        if rank == root:
            print("Starting blended nowcast computation.",flush=True)
            if measure_time:
                starttime_mainloop = time.time()
    
        if isinstance(timesteps, int):
            timesteps = range(timesteps + 1)
            timestep_type = "int"
        else:
            original_timesteps = [0] + list(timesteps)
            timesteps = nowcast_utils.binned_timesteps(original_timesteps)
            timestep_type = "list"
    
        extrap_kwargs["return_displacement"] = True
        forecast_prev = precip_cascade
        noise_prev = noise_cascade
        t_prev = [0.0 for j in range(precip_cascade.shape[0])]
        t_total = [0.0 for j in range(precip_cascade.shape[0])]
    
        for t, subtimestep_idx in enumerate(timesteps):
            if timestep_type == "list":
                subtimesteps = [original_timesteps[t_] for t_ in subtimestep_idx]
            else:
                subtimesteps = [t]
    
            if (timestep_type == "list" and subtimesteps) or (
                timestep_type == "int" and t > 0
            ):
                is_nowcast_time_step = True
            else:
                is_nowcast_time_step = False
    
            if is_nowcast_time_step and rank == 0:
                print(
                    f"Computing nowcast for time step {t}...",
                    end="",
                    flush=True,
                    )
    
            if measure_time and rank == 0:
                starttime = time.time()
    

            # With the way it is implemented at this moment: n_ens_members of the output equals
            # the maximum number of (ensemble) members in the input (either the nowcasts or NWP).
    
            # Send or receive the NWP precipitation/velocities for the current timestep
    
            if rank == root:
                (
                    precip_models_cascade_temp,
                    precip_models_pm_temp,
                    velocity_models_temp,
                    mu_models_temp,
                    sigma_models_temp,
                    n_model_indices,
                ) = _find_nwp_combination(
                    precip_models_cascade[:, t, :, :, :],
                    precip_models_pm[:, t, :, :],
                    velocity_models[:, t, :, :, :],
                    mu_models[:, t, :],
                    sigma_models[:, t, :],
                    n_ens_members,
                    ar_order,
                    n_cascade_levels,
                    blend_nwp_members,
                    )
    
                if t == 0:
                # Calculate the initial skill of the (NWP) model forecasts at t=0
                # Do it here before the members are scattered to the different ranks
                    rho_nwp_models = _compute_initial_nwp_skill(
                        precip_cascade,
                        precip_models_cascade_temp,
                        domain_mask,
                        issuetime,
                        outdir_path_skill,
                        clim_kwargs,
                        )
            else:
                # initialize the variables that need to be scattered
                precip_models_cascade_temp = None
                precip_models_pm_temp = None
                velocity_models_temp = None
                mu_models_temp = None
                sigma_models_temp = None
                n_model_indices =  None
                if t == 0:
                    rho_nwp_models = None
    
    
            # Scatter the arrays to all ranks
            precip_models_cascade_temp = scatter(
                precip_models_cascade_temp,
                n_ens_members,comm,
                root=0,
                tag=11
                )
            precip_models_pm_temp = scatter(
                precip_models_pm_temp,
                n_ens_members,
                comm,
                root=0,
                tag=12
                )
            velocity_models_temp = scatter(
                velocity_models_temp,
                n_ens_members,
                comm,
                root=0,
                tag=13
                )
            mu_models_temp = scatter(
                mu_models_temp,
                n_ens_members,
                comm,
                root=0,
                tag=14
                )
            sigma_models_temp = scatter(
                sigma_models_temp,
                n_ens_members,
                comm,
                root=0,
                tag=15
                )
            n_model_indices = scatter(
                n_model_indices,
                n_ens_members,
                comm,
                root=0,
                tag=17
                )
            if t==0:
                rho_nwp_models = scatter(
                    rho_nwp_models,
                    n_ens_members,
                    comm,
                    root=0,
                    tag=18
                    )
            if t>0:
                (
                    rho_extr,
                    rho_extr_prev,
                ) = blending.skill_scores.lt_dependent_cor_extrapolation(
                    PHI=PHI, correlations=rho_extr, correlations_prev=rho_extr_prev
                    )
    
    
            #############################################################
            #   Start the loop over the ensemble members in each rank   #
            #############################################################
            R_rank = []
            for j in range(n_local_ens_members):
                # Determine the skill of the nwp components for lead time (t0 + t)
                # Then for the model components
                if blend_nwp_members:
                    rho_nwp_fc = [
                        blending.skill_scores.lt_dependent_cor_nwp(
                        lt=(t * int(timestep)),
                        correlations=rho_nwp_models[n_model],
                        outdir_path=outdir_path_skill,
                        n_model=n_model,
                        skill_kwargs=clim_kwargs,
                            )
                        for n_model in range(rho_nwp_models.shape[0])
                        ]
                    rho_nwp_fc = np.stack(rho_nwp_fc)
                    # Concatenate rho_extr and rho_nwp
                    rho_fc = np.concatenate((rho_extr[None, :], rho_nwp_fc), axis=0)
                else:
                    rho_nwp_fc = blending.skill_scores.lt_dependent_cor_nwp(
                        lt=(t * int(timestep)),
                        correlations=rho_nwp_models[j],
                        outdir_path=outdir_path_skill,
                        n_model=n_model_indices[j],
                        skill_kwargs=clim_kwargs,
                        )
                    # Concatenate rho_extr and rho_nwp
                    rho_fc = np.concatenate(
                        (rho_extr[None, :], rho_nwp_fc[None, :]), axis=0
                        )
    
                # Determine the weights per component
                # Weights following the bps method. These are needed for the velocity
                # weights prior to the advection step. If weights method spn is
                # selected, weights will be overwritten with those weights prior to
                # blending step.
                # weight = [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                weights = calculate_weights_bps(rho_fc)
    
                # The model only weights
                if weights_method == "bps":
                    # Determine the weights of the components without the extrapolation
                    # cascade, in case this is no data or outside the mask.
                    weights_model_only = calculate_weights_bps(rho_fc[1:, :])
                elif weights_method == "spn":
                    # Only the weights of the components without the extrapolation
                    # cascade will be determined here. The full set of weights are
                    # determined after the extrapolation step in this method.
                    if blend_nwp_members and precip_models_cascade_temp.shape[0] > 1:
                        weights_model_only = np.zeros(
                            (precip_models_cascade_temp.shape[0] + 1, n_cascade_levels)
                            )
                        for i in range(n_cascade_levels):
                            # Determine the normalized covariance matrix (containing)
                            # the cross-correlations between the models
                            cov = np.corrcoef(
                                np.stack(
                                    [
                                        precip_models_cascade_temp[
                                            n_model, i, :, :
                                            ].flatten()
                                        for n_model in range(
                                            precip_models_cascade_temp.shape[0]
                                            )
                                        ]
                                        )
                                )
                            # Determine the weights for this cascade level
                            weights_model_only[:, i] = calculate_weights_spn(
                                correlations=rho_fc[1:, i], cov=cov
                                )
                    else:
                        # Same as correlation and noise is 1 - correlation
                        weights_model_only = calculate_weights_bps(rho_fc[1:, :])
                else:
                    raise ValueError(
                        "Unknown weights method %s: must be 'bps' or 'spn'" % weights_method
                        )
    
                # Determine the noise cascade and regress this to the subsequent
                # time step + regress the extrapolation component to the subsequent
                # time step
    
                # - determine the epsilon, a cascade of temporally independent
                # but spatially correlated noise
                if noise_method is not None:
                    # generate noise field
                    EPS = generate_noise(
                        pp, randstate=randgen_prec[j], fft_method=fft_objs[j], domain=domain
                        )
    
                    # decompose the noise field into a cascade
                    EPS = decompositor(
                        EPS,
                        bp_filter,
                        fft_method=fft_objs[j],
                        input_domain=domain,
                        output_domain=domain,
                        compute_stats=True,
                        normalize=True,
                        compact_output=True,
                        )
                else:
                    EPS = None
    
                # 8.3.2 regress the extrapolation component to the subsequent time
                # step
                # iterate the AR(p) model for each cascade level
                for i in range(n_cascade_levels):
                    # apply AR(p) process to extrapolation cascade level
                    if EPS is not None or vel_pert_method is not None:
                        precip_cascade[j][i] = autoregression.iterate_ar_model(
                            precip_cascade[j][i], PHI[i, :]
                        )
                        # Renormalize the cascade
                        precip_cascade[j][i][1] /= np.std(precip_cascade[j][i][1])
    
                    else:
                        # use the deterministic AR(p) model computed above if
                        # perturbations are disabled
                        precip_cascade[j][i] = R_m[i]
    
                # 8.3.3 regress the noise component to the subsequent time step
                # iterate the AR(p) model for each cascade level
                for i in range(n_cascade_levels):
                    # normalize the noise cascade
                    if EPS is not None:
                        EPS_ = EPS["cascade_levels"][i]
                        EPS_ *= noise_std_coeffs[i]
                    else:
                        EPS_ = None
                    # apply AR(p) process to noise cascade level
                    # (Returns zero noise if EPS is None)
                    noise_cascade[j][i] = autoregression.iterate_ar_model(
                        noise_cascade[j][i], PHI[i, :], eps=EPS_
                        )
                
                EPS = None
                EPS_ = None
    
                # - perturb and blend the advection fields + advect the
                # extrapolation and noise cascade to the current time step
                # (or subtimesteps if non-integer time steps are given)
    
                # Settings and initialize the output
                extrap_kwargs_ = extrap_kwargs.copy()
                extrap_kwargs_noise = extrap_kwargs.copy()
                extrap_kwargs_pb = extrap_kwargs.copy()
                velocity_pert = velocity
                R_f_ep_out = []
                Yn_ep_out = []
                R_pm_ep = []
    
                # Extrapolate per sub time step 
                for t_sub in subtimesteps:
                    if t_sub > 0:
                        t_diff_prev_int = t_sub - int(t_sub)
                        if t_diff_prev_int > 0.0:
                            R_f_ip = [
                                (1.0 - t_diff_prev_int) * forecast_prev[j][i][-1, :]
                                + t_diff_prev_int * precip_cascade[j][i][-1, :]
                                for i in range(n_cascade_levels)
                                ]
                            Yn_ip = [
                                (1.0 - t_diff_prev_int) * noise_prev[j][i][-1, :]
                                + t_diff_prev_int * noise_cascade[j][i][-1, :]
                                for i in range(n_cascade_levels)
                                ]
    
                        else:
                            R_f_ip = [
                                forecast_prev[j][i][-1, :] for i in range(n_cascade_levels)
                                ]
                            Yn_ip = [
                                noise_prev[j][i][-1, :] for i in range(n_cascade_levels)
                                ]
    
                        R_f_ip = np.stack(R_f_ip)
                        Yn_ip = np.stack(Yn_ip)
                        t_diff_prev = t_sub - t_prev[j]
                        t_total[j] += t_diff_prev
    
                        # compute the perturbed motion field - include the NWP
                        # velocities and the weights. Note that we only perturb
                        # the extrapolation velocity field, as the NWP velocity
                        # field is present per time step
                        if vel_pert_method is not None:
                            velocity_pert = velocity + generate_vel_noise(
                                vps[j], t_total[j] * timestep
                                )
    
                        # Stack the perturbed extrapolation and the NWP velocities
                        if blend_nwp_members:
                            V_stack = np.concatenate(
                                (
                                velocity_pert[None, :, :, :],
                                velocity_models_temp,
                                ),
                                axis=0,
                                )
                        else:
                            V_model_ = velocity_models_temp[j]
                            V_stack = np.concatenate(
                                (velocity_pert[None, :, :, :], V_model_[None, :, :, :]),
                                axis=0,
                                )
                            V_model_ = None
    
                        # Obtain a blended optical flow, using the weights of the
                        # second cascade following eq. 24 in BPS2006
                        velocity_blended = blending.utils.blend_optical_flows(
                            flows=V_stack,
                            weights=weights[
                                :-1, 1
                                ],  # [(extr_field, n_model_fields), cascade_level=2]
                            )
    
                        # Extrapolate both cascades to the next time step
                        # First recompose the cascade, advect it and decompose it again
                        # This is needed to remove the interpolation artifacts.
                        # In addition, the number of extrapolations is greatly reduced
                        # A. Rain
                        R_f_ip_recomp = blending.utils.recompose_cascade(
                            combined_cascade=R_f_ip,
                            combined_mean=mu_extrapolation,
                            combined_sigma=sigma_extrapolation,
                        )
                        # Put back the mask
                        R_f_ip_recomp[domain_mask] = np.NaN
                        extrap_kwargs["displacement_prev"] = D[j]
                        R_f_ep_recomp_, D[j] = extrapolator(
                            R_f_ip_recomp,
                            velocity_blended,
                            [t_diff_prev],
                            allow_nonfinite_values=True,
                            **extrap_kwargs,
                        )
                        R_f_ep_recomp = R_f_ep_recomp_[0].copy()
                        temp_mask = ~np.isfinite(R_f_ep_recomp)
                        
                        R_f_ep_recomp[~np.isfinite(R_f_ep_recomp)] = zerovalue
                        R_f_ep = decompositor(
                            R_f_ep_recomp,
                            bp_filter,
                            mask=MASK_thr,
                            fft_method=fft,
                            output_domain=domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        )["cascade_levels"]
                        for i in range(n_cascade_levels):
                            R_f_ep[i][temp_mask] = np.NaN
                        
                        # B. Noise
                        Yn_ip_recomp = blending.utils.recompose_cascade(
                            combined_cascade=Yn_ip,
                            combined_mean=mu_noise[j],
                            combined_sigma=sigma_noise[j],
                        )
                        extrap_kwargs_noise["displacement_prev"] = D_Yn[j]
                        extrap_kwargs_noise["map_coordinates_mode"] = "wrap"
                        Yn_ep_recomp_, D_Yn[j] = extrapolator(
                            Yn_ip_recomp,
                            velocity_blended,
                            [t_diff_prev],
                            allow_nonfinite_values=True,
                            **extrap_kwargs_noise,
                        )
                        Yn_ep_recomp = Yn_ep_recomp_[0].copy()
                        Yn_ep = decompositor(
                            Yn_ep_recomp,
                            bp_filter,
                            mask=MASK_thr,
                            fft_method=fft,
                            output_domain=domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        )["cascade_levels"]
                        for i in range(n_cascade_levels):
                            Yn_ep[i] *= noise_std_coeffs[i]
                        
                        # Append the results to the output lists
                        R_f_ep_out.append(R_f_ep.copy())
                        Yn_ep_out.append(Yn_ep.copy())
                        R_f_ip_recomp = None
                        R_f_ep_recomp_ = None
                        R_f_ep_recomp = None
                        Yn_ip_recomp = None
                        Yn_ep_recomp_ = None
                        Yn_ep_recomp = None
    
                        # Finally, also extrapolate the initial radar rainfall
                        # field. This will be blended with the rainfall field(s)
                        # of the (NWP) model(s) for Lagrangian blended prob. matching
                        # min_R = np.min(precip)
                        extrap_kwargs_pb["displacement_prev"] = D_pb[j]
                        # Apply the domain mask to the extrapolation component
                        R_ = precip.copy()
                        R_[domain_mask] = np.NaN
                        R_pm_ep_, D_pb[j] = extrapolator(
                            R_,
                            velocity_blended,
                            [t_diff_prev],
                            allow_nonfinite_values=True,
                            **extrap_kwargs_pb,
                            )
                        R_pm_ep.append(R_pm_ep_[0])
    
                        t_prev[j] = t_sub
    
    
                if len(R_f_ep_out) > 0:
                    R_f_ep_out = np.stack(R_f_ep_out)
                    Yn_ep_out = np.stack(Yn_ep_out)
                    R_pm_ep = np.stack(R_pm_ep)
    
                # advect the forecast field by one time step if no subtimesteps in the
                # current interval were found
                if not subtimesteps:
                    t_diff_prev = t + 1 - t_prev[j]
                    t_total[j] += t_diff_prev
    
                    # compute the perturbed motion field - include the NWP
                    # velocities and the weights
                    if vel_pert_method is not None:
                        velocity_pert = velocity + generate_vel_noise(
                            vps[j], t_total[j] * timestep
                            )
    
                    # Stack the perturbed extrapolation and the NWP velocities
                    if blend_nwp_members:
                        V_stack = np.concatenate(
                            (velocity_pert[None, :, :, :], velocity_models_temp),
                            axis=0,
                            )
                    else:
                        V_model_ = velocity_models_temp[j]
                        V_stack = np.concatenate(
                            (velocity_pert[None, :, :, :], V_model_[None, :, :, :]), axis=0
                            )
                        V_model_ = None
    
                    # Obtain a blended optical flow, using the weights of the
                    # second cascade following eq. 24 in BPS2006
                    velocity_blended = blending.utils.blend_optical_flows(
                        flows=V_stack,
                        weights=weights[
                            :-1, 1
                            ],  # [(extr_field, n_model_fields), cascade_level=2]
                        )
    
                    # Extrapolate the extrapolation and noise cascade
                    extrap_kwargs_["displacement_prev"] = D[j][i]
                    extrap_kwargs_noise["displacement_prev"] = D_Yn[j][i]
                    extrap_kwargs_noise["map_coordinates_mode"] = "wrap"
                    
                    _, D[j] = extrapolator(
                        None,
                        velocity_blended,
                        [t_diff_prev],
                        allow_nonfinite_values=True,
                        **extrap_kwargs_,
                        )
    
                    _, D_Yn[j] = extrapolator(
                        None,
                        velocity_blended,
                        [t_diff_prev],
                        allow_nonfinite_values=True,
                        **extrap_kwargs_noise,
                        )
    
                    # Also extrapolate the radar observation, used for the probability
                    # matching and post-processing steps
                    extrap_kwargs_pb["displacement_prev"] = D_pb[j]
                    _, D_pb[j] = extrapolator(
                        None,
                        velocity_blended,
                        [t_diff_prev],
                        allow_nonfinite_values=True,
                        **extrap_kwargs_pb,
                        )
    
                    t_prev[j] = t + 1
    
                forecast_prev[j] = precip_cascade[j]
    
                # Blend the cascades
                R_f_out = []
    
                for t_sub in subtimesteps:
                    # TODO: does it make sense to use sub time steps - check if it works?
                    if t_sub > 0:
                        t_index = np.where(np.array(subtimesteps) == t_sub)[0][0]
                        # First concatenate the cascades and the means and sigmas
                        # precip_models = [n_models,timesteps,n_cascade_levels,m,n]
                        if blend_nwp_members:
                            cascades_stacked = np.concatenate(
                                (
                                    R_f_ep_out[None, t_index],
                                    precip_models_cascade_temp,
                                    Yn_ep_out[None, t_index],
                                ),
                                axis=0,
                                )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                            means_stacked = np.concatenate(
                                (mu_extrapolation[None, :], mu_models_temp), axis=0
                                )
                            sigmas_stacked = np.concatenate(
                                (sigma_extrapolation[None, :], sigma_models_temp),
                                axis=0,
                                )
                        else:
                            cascades_stacked = np.concatenate(
                                (
                                    R_f_ep_out[None, t_index],
                                    precip_models_cascade_temp[None, j],
                                    Yn_ep_out[None, t_index],
                                ),
                                axis=0,
                                )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                            means_stacked = np.concatenate(
                                (mu_extrapolation[None, :], mu_models_temp[None, j]), axis=0
                                )
                            sigmas_stacked = np.concatenate(
                                (sigma_extrapolation[None, :], sigma_models_temp[None, j]),
                                axis=0,
                                )
    
                        # First determine the blending weights if method is spn. The
                        # weights for method bps have already been determined.
                        if weights_method == "spn":
                            weights = np.zeros(
                                (cascades_stacked.shape[0], n_cascade_levels)
                                )
                            for i in range(n_cascade_levels):
                                # Determine the normalized covariance matrix (containing)
                                # the cross-correlations between the models
                                cascades_stacked_ = np.stack(
                                    [
                                        cascades_stacked[n_model, i, :, :].flatten()
                                        for n_model in range(cascades_stacked.shape[0] - 1)
                                    ]
                                    )  # -1 to exclude the noise component
                                cov = np.ma.corrcoef(
                                    np.ma.masked_invalid(cascades_stacked_)
                                    )
                                # Determine the weights for this cascade level
                                weights[:, i] = calculate_weights_spn(
                                    correlations=rho_fc[:, i], cov=cov
                                    )
    
                        # Blend the extrapolation, (NWP) model(s) and noise cascades
                        R_f_blended = blending.utils.blend_cascades(
                            cascades_norm=cascades_stacked, weights=weights
                            )
                        # Also blend the cascade without the extrapolation component
                        R_f_blended_mod_only = blending.utils.blend_cascades(
                            cascades_norm=cascades_stacked[1:, :],
                            weights=weights_model_only,
                            )
    
                        # Blend the means and standard deviations
                        # Input is array of shape [number_components, scale_level, ...]
                        means_blended, sigmas_blended = blend_means_sigmas(
                            means=means_stacked, sigmas=sigmas_stacked, weights=weights
                            )
                        # Also blend the means and sigmas for the cascade without extrapolation
                        (
                            means_blended_mod_only,
                            sigmas_blended_mod_only,
                        ) = blend_means_sigmas(
                            means=means_stacked[1:, :],
                            sigmas=sigmas_stacked[1:, :],
                            weights=weights_model_only,
                            )
    
                        # Recompose the cascade to a precipitation field
                        # (The function first normalizes the blended cascade, R_f_blended
                        # again)
                        R_f_new = blending.utils.recompose_cascade(
                            combined_cascade=R_f_blended,
                            combined_mean=means_blended,
                            combined_sigma=sigmas_blended,
                            )
                        # The recomposed cascade without the extrapolation (for NaN filling
                        # outside the radar domain)
                        R_f_new_mod_only = blending.utils.recompose_cascade(
                            combined_cascade=R_f_blended_mod_only,
                            combined_mean=means_blended_mod_only,
                            combined_sigma=sigmas_blended_mod_only,
                            )
                        if domain == "spectral":
                            # TODO: Check this! (Only tested with domain == 'spatial')
                            R_f_new = fft_objs[j].irfft2(R_f_new)
                            R_f_new_mod_only = fft_objs[j].irfft2(R_f_new_mod_only)
    
    
                        # Post-processing steps - use the mask and fill no data with
                        # the blended NWP forecast. Probability matching following
                        # Lagrangian blended probability matching which uses the
                        # latest extrapolated radar rainfall field blended with the
                        # nwp model(s) rainfall forecast fields as 'benchmark'.
    
                        # TODO: Check probability matching method
                        # 8.7.1 first blend the extrapolated rainfall field (the field
                        # that is only used for post-processing steps) with the NWP
                        # rainfall forecast for this time step using the weights
                        # at scale level 2.
                        weights_pm = weights[:-1, 1]  # Weights without noise, level 2
                        weights_pm_normalized = weights_pm / np.sum(weights_pm)
                        # And the weights for outside the radar domain
                        weights_pm_mod_only = weights_model_only[
                            :-1, 1
                            ]  # Weights without noise, level 2
                        weights_pm_normalized_mod_only = weights_pm_mod_only / np.sum(
                            weights_pm_mod_only
                            )
                        # Stack the fields
                        if blend_nwp_members:
                            R_pm_stacked = np.concatenate(
                                (
                                    R_pm_ep[None, t_index],
                                    precip_models_pm_temp,
                                ),
                                axis=0,
                                )
                        else:
                            R_pm_stacked = np.concatenate(
                                (
                                    R_pm_ep[None, t_index],
                                    precip_models_pm_temp[None, j],
                                ),
                                axis=0,
                                )
    
                      # Blend it
                        R_pm_blended = np.sum(
                            weights_pm_normalized.reshape(
                                weights_pm_normalized.shape[0], 1, 1
                                )
                            * R_pm_stacked,
                            axis=0,
                            )
                        if blend_nwp_members:
                            R_pm_blended_mod_only = np.sum(
                                weights_pm_normalized_mod_only.reshape(
                                    weights_pm_normalized_mod_only.shape[0], 1, 1
                                    )
                                * precip_models_pm_temp,
                                axis=0,
                                )
                        else:
                            R_pm_blended_mod_only = precip_models_pm_temp[j]
    
                        # The extrapolation components are NaN outside the advected
                        # radar domain. This results in NaN values in the blended
                        # forecast outside the radar domain. Therefore, fill these
                        # areas with the "..._mod_only" blended forecasts, consisting
                        # of the NWP and noise components.
                        nan_indices = np.isnan(R_f_new)
                        R_f_new[nan_indices] = R_f_new_mod_only[nan_indices]
                        nan_indices = np.isnan(R_pm_blended)
                        R_pm_blended[nan_indices] = R_pm_blended_mod_only[nan_indices]
                        # Finally, fill the remaining nan values, if present, with
                        # the minimum value in the forecast
                        nan_indices = np.isnan(R_f_new)
                        R_f_new[nan_indices] = np.nanmin(R_f_new)
                        nan_indices = np.isnan(R_pm_blended)
                        R_pm_blended[nan_indices] = np.nanmin(R_pm_blended)
    
                        # 8.7.2. Apply the masking and prob. matching
                        if mask_method is not None:
                            # apply the precipitation mask to prevent generation of new
                            # precipitation into areas where it was not originally
                            # observed
                            R_cmin = R_f_new.min()
                            if mask_method == "incremental":
                                # The incremental mask is slightly different from
                                # the implementation in the non-blended steps.py, as
                                # it is not based on the last forecast, but instead
                                # on R_pm_blended. Therefore, the buffer does not
                                # increase over time.
                                # Get the mask for this forecast
                                MASK_prec = R_pm_blended >= precip_thr
                                # Buffer the mask
                                MASK_prec = _compute_incremental_mask(
                                    MASK_prec, struct, mask_rim
                                    )
                                # Get the final mask
                                R_f_new = R_cmin + (R_f_new - R_cmin) * MASK_prec
                                MASK_prec_ = R_f_new > R_cmin
                            elif mask_method == "obs":
                                # The mask equals the most recent benchmark
                                # rainfall field
                                MASK_prec_ = R_pm_blended >= precip_thr
    
                            # Set to min value outside of mask
                            R_f_new[~MASK_prec_] = R_cmin
    
                        if probmatching_method == "cdf":
                            # adjust the CDF of the forecast to match the most recent
                            # benchmark rainfall field (R_pm_blended)
                            R_f_new = probmatching.nonparam_match_empirical_cdf(
                                R_f_new, R_pm_blended
                                )
                        elif probmatching_method == "mean":
                            # Use R_pm_blended as benchmark field and
                            mu_0 = np.mean(R_pm_blended[R_pm_blended >= precip_thr])
                            MASK = R_f_new >= precip_thr
                            mu_fct = np.mean(R_f_new[MASK])
                            R_f_new[MASK] = R_f_new[MASK] - mu_fct + mu_0
    
                        R_f_out.append(R_f_new)
                # END OF LOOP OVER subtimesteps
                if t > 0:
                    R_rank.append(np.stack(R_f_out))
            if callback is not None and t>0:
                R_rank_stacked =  np.stack(R_rank)
                if R_rank_stacked.shape[1] > 0:
                    callback(R_rank_stacked.squeeze())
            if rank == root and is_nowcast_time_step:
                if measure_time:
                    print(f"{time.time() - starttime:.2f} seconds.",flush=True)
                else:
                    print("done.",flush=True)
    ##TODO fix this
    #        if return_output:
    #            for j in range(n_local_ens_members):
    #                R_f[j].extend(R_rank[j])
    
            R_rank = None
            
        if measure_time and rank == root:
            mainloop_time = time.time() - starttime_mainloop
    
        return None


         



        


        
        






def _check_inputs(
    precip, precip_models, velocity, velocity_models, timesteps, ar_order
):
    if precip is not None and precip.ndim != 3 :
        raise ValueError("precip must be a three-dimensional array")
    if precip is not None and precip.shape[0] < ar_order + 1:
        raise ValueError("precip.shape[0] < ar_order+1")
##TODO Find a way to work wih array instead of list of lists
#    if len(precip_models) != 0 and (precip_models.ndim != 2 and precip_models.ndim != 4):
#        raise ValueError(
#            "precip_models must be either a two-dimensional array containing dictionaries with decomposed model fields or a four-dimensional array containing the original (NWP) model forecasts"
#        )
    if velocity is not None and velocity.ndim != 3:
        raise ValueError("velocity must be a three-dimensional array")
    if velocity is not None and velocity.shape[0] != 2:
        raise ValueError(
                "velocity must have an x- and y-component, check the shape"
            )
    if not isinstance(velocity_models,list):
        raise ValueError("velocity_models must be a list")
    for vm in velocity_models:
        if vm.ndim != 4:
            raise ValueError("elements of velocity_models must be a four-dimensional array")
        if velocity_models[0].shape[1] != 2:
            raise ValueError(
                "velocity and velocity_models must have an x- and y-component, check the shape"
            )
    if (precip is not None and velocity is not None) and precip.shape[1:3] != velocity.shape[1:3]:
        raise ValueError(
            "dimension mismatch between precip and velocity: shape(precip)=%s, shape(velocity)=%s"
            % (str(precip.shape), str(velocity.shape))
        )
#    if precip_models.shape[0] != velocity_models.shape[0]:
#        raise ValueError(
#            "precip_models and velocity_models must consist of the same number of models"
#        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")
    if len(precip_models) != 0:
        if isinstance(timesteps, list):
            if len(precip_models[0]) != len(timesteps) + 1:
                raise ValueError(
                    "precip_models does not contain sufficient lead times for this forecast"
                )
        else:
            if len(precip_models[0]) != timesteps + 1:
                raise ValueError(
                    "precip_models does not contain sufficient lead times for this forecast"
                )
                


def broadcast(data,comm,root=0):
    import numpy as np
    rank = comm.Get_rank()
    if rank == root:
        if type(data).__module__ == np.__name__:
            is_numpy = True
            data_shape = data.shape
            data_dtype = data.dtype
        else:
            is_numpy = False
    else:
        is_numpy = None
        data_shape = None
        data_dtype = None
    is_numpy = comm.bcast(is_numpy,root=root)
    if is_numpy:
        data_shape=comm.bcast(data_shape,root=root)
        data_dtype=comm.bcast(data_dtype,root=root)
        if rank != root:
            data = np.zeros(data_shape,dtype=data_dtype)
        comm.Bcast(data,root=root)
    else:
        data=comm.bcast(data)
    return(data)

def scatter(data,n_ens_members,comm,root=0,tag=11):
    import numpy as np
    from mpi4py.util.dtlib import from_numpy_dtype
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    if rank == root:
        if len(data) != n_ens_members:
            raise ValueError(
                "The number of ensemble members is not equal to the length/first dimension of data."
                )
        split_sizes = np.array(
            [ len(np.array_split(np.ones(n_ens_members),nproc)[i]) for i in range(nproc) ]
            )
        displacements = np.insert(np.cumsum(split_sizes),0,0)[0:-1]
        if type(data).__module__ == np.__name__:
            is_numpy = True
            data_dtype = data.dtype
            rem_dim = data.shape[1:]
            split_sizes_input = split_sizes*np.prod(rem_dim)
            displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
        elif type(data) == list:
            is_numpy=False
            for i in range(1,nproc):
                comm.send(
                    data[int(displacements[i]):int(displacements[i]+split_sizes[i])],
                    dest=i,
                    tag=tag
                    )
            data=data[0:int(split_sizes[0])]
        else:
            raise TypeError(
                "Scatter is only implemented for lists or numpy arrays!"
                )
    else:
        is_numpy = None
        data_dtype = None
        split_sizes = None
        rem_dim =  None
        split_sizes_input = None
        displacements_input = None

    is_numpy = comm.bcast(is_numpy,root=root)
    if is_numpy:
        split_sizes = comm.bcast(split_sizes,root=root)
        rem_dim = comm.bcast(rem_dim,root=root)
        split_sizes_input = comm.bcast(split_sizes_input,root=root)
        displacements_input = comm.bcast(displacements_input,root=root)
        data_dtype = comm.bcast(data_dtype,root=root)
        mpi_type = from_numpy_dtype(data_dtype)
        rcvbuf = np.zeros(np.insert(rem_dim,0,split_sizes[rank]).astype(int),dtype=data_dtype)
        comm.Scatterv([data,split_sizes_input,displacements_input,mpi_type],rcvbuf,root=root)
        data = rcvbuf
    else:
        if rank != root:
            data = comm.recv(source=root,tag=tag)

    return(data)

def _send_receive_nwp(r_nwp,v_nwp,timestep,comm,timesteps=None,root=0):
    if timestep < 0 and timesteps == None:
        raise ValueError(
                "If timestep < 0, timesteps cannot be None"
                )
    rank = comm.Get_rank()
    has_nwp = (len(r_nwp) != 0)
    if has_nwp:
            r_nwp = np.stack(r_nwp)
            v_nwp = np.stack(v_nwp)
    else:
        r_nwp_t = None
        v_nwp_t = None
    if timestep > 0:
        if has_nwp:
            r_nwp_t = r_nwp[:,timestep]
            v_nwp_t = v_nwp[:,timestep,:,:,:]
        r_nwp_t = comm.gather(r_nwp_t,root=root)
        v_nwp_t = comm.gather(v_nwp_t,root=root)
        if rank ==  root:
            r_nwp_t = [item for item in r_nwp_t if item is not None]
            v_nwp_t = [item for item in v_nwp_t if item is not None]
            return((r_nwp_t,v_nwp_t))
        else:
            return((None,None))
    else:
        if rank == root:
            r_out_ = []
            v_out_ = []
        for t in range(timesteps+1):
            if has_nwp:
                r_nwp_t = r_nwp[:,timestep]
                v_nwp_t = v_nwp[:,timestep,:,:,:]
            else:
                r_nwp_t = None
                v_nwp_t = None
            r_nwp_t = comm.gather(r_nwp_t,root=root)
            v_nwp_t = comm.gather(v_nwp_t,root=root)
            if rank == root:
                r_nwp_t = [item for item in r_nwp_t if item is not None]
                v_nwp_t = [item for item in v_nwp_t if item is not None]
                r_out_.append(r_nwp_t.copy())
                v_out_.append(v_nwp_t.copy())
        if rank == root:
            r_out = []
            v_out = []
            for n in range(len(r_out_[0])):
                r_temp = []
                v_temp = []
                for t in range(len(r_out_)):
                    r_temp.append(r_out_[t][n])
                    v_temp.append(v_out_[t][n])
                v_temp = np.stack(v_temp)
                r_out.append(r_temp)
                v_out.append(v_temp)
            r_out=np.stack(r_out)[:,:,0]
            v_out=np.stack(v_out)[:,:,0,:,:,:]
            return((r_out,v_out))
        else:
            return((None,None))












