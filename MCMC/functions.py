from __future__ import division, print_function, unicode_literals

import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import emcee
print(emcee.__version__) #need version 3 (pip install emcee==3.0rc1)
import yaml

import scipy
from scipy.stats import gaussian_kde, ks_2samp

from astropy.table import Table, Column
from astropy.io import ascii, fits
from astropy.cosmology import FlatLambdaCDM

from halotools.sim_manager import TabularAsciiReader, HaloTableCache, CachedHaloCatalog
from halotools.sim_manager.rockstar_hlist_reader import RockstarHlistReader
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.mock_observables import delta_sigma_from_precomputed_pairs, total_mass_enclosed_per_cylinder
from halotools.utils import randomly_downsample_data

#Chris' SM
from cb.get_sm_for_sim import *

#Asher's likelihood
owd = os.getcwd()
os.chdir("/Users/fardila/Documents/GitHub/ddd")
from ddd.ddd import logp_ddd as ddd_lnlike
os.chdir(owd)
#memory profile
# from guppy import hpy
# import datetime


################################################################################
#Configuration
################################################################################
def parse_config(config_file):
    """Prepare configurations.
    Read configuration parameters from an input .yaml file.
    """
    cfg = yaml.load(open(config_file))

    return cfg

def load_observed_data(cfg, verbose=True):
    """Load the observed data."""

    #dwarf masses
    cosmos_dwarf_sample_data  = fits.open(os.path.join(cfg['data_location'],
                                                       cfg['cosmos_dir'],
                                                       cfg['cosmos_dwarf_file']))[1].data
    cosmos_dwarf_masses = cosmos_dwarf_sample_data['mass_med']

    #SMF
    cosmos_SMF_fit_table = ascii.read(os.path.join(cfg['data_location'],
                                                   cfg['cosmos_dir'],
                                                   cfg['cosmos_SMF_fit_file']))
    cosmos_SMF_points_table = ascii.read(os.path.join(cfg['data_location'],
                                                      cfg['cosmos_dir'],
                                                      cfg['cosmos_SMF_points_file']))

    #weak lensing
    cosmos_lensing_table = ascii.read(os.path.join(cfg['data_location'],
                                                   cfg['cosmos_dir'],
                                                   cfg['cosmos_wl_file']))

    cosmos_wl_r = cosmos_lensing_table['R(Mpc)']
    cosmos_wl_ds = cosmos_lensing_table['SigR(Msun/pc^2)']

    #cosmology
    cfg['cosmos_cosmo'] = FlatLambdaCDM(H0=cfg['cosmos_h0'] * 100,
                                     Om0=cfg['cosmos_omega_m'])

#     obs_volume = ((cfg['obs_cosmo'].comoving_volume(obs_zmax) -
#                    cfg['obs_cosmo'].comoving_volume(obs_zmin)) *
#                   (cfg['obs_area'] / 41254.0)).value
#     cfg['obs_volume'] = obs_volume

    return {'cosmos_wl_r': cosmos_wl_r,
            'cosmos_wl_ds': cosmos_wl_ds,
            'cosmos_wl_table': cosmos_lensing_table,
            'cosmos_SMF_fit_table': cosmos_SMF_fit_table,
            'cosmos_SMF_points_table': cosmos_SMF_points_table,
            'cosmos_dwarf_masses': cosmos_dwarf_masses}, cfg

def load_sim_data(cfg):
    """Load the UniverseMachine data."""


    #read in halocat
    halocat = CachedHaloCatalog(simname = cfg['sim_name'],
                                halo_finder = cfg['sim_halo_finder'],
                                version_name = cfg['sim_version_name'],
                                redshift = cfg['sim_z'],
                                ptcl_version_name=cfg['sim_ptcl_version_name']) # doctest: +SKIP

    #read in particle table
    ptcl_table = Table.read(os.path.join(cfg['data_location'],
                                         cfg['sim_dir'],
                                         cfg['sim_particle_file']), path='data')
    px = ptcl_table['x']
    py = ptcl_table['y']
    pz = ptcl_table['z']
    particles = np.vstack((px, py, pz)).T
    ptcl_table = 0

    #downsample
    num_ptcls_to_use = int(1e4)
    particles = randomly_downsample_data(particles, num_ptcls_to_use)
    particle_masses = np.zeros(num_ptcls_to_use) + halocat.particle_mass
    downsampling_factor = (cfg['sim_particles_per_dimension']**3)/float(len(particles))

    # other parameters
    cfg['sim_cosmo'] = FlatLambdaCDM(H0=cfg['sim_h0'] * 100.0,
                                    Om0=cfg['sim_omega_m'])

    cfg['sim_volume'] = np.power(cfg['sim_lbox'] / cfg['sim_h0'], 3)
#     if verbose:
#         print("# The volume of the UniverseMachine mock is %15.2f Mpc^3" %
#               cfg['um_volume'])

    return {'halocat' : halocat, 'particles' : particles,
            'particle_masses' : particle_masses, 'downsampling_factor' : downsampling_factor}, cfg


def initial_model(config, verbose=True):
    """Initialize the model."""

    print(config['model_type'])

    # Configuration for COSMOS data
    data_obs, config_obs = load_observed_data(config, verbose=verbose)

    # Configuration for simulation data.
    data_sim, config_obs_sim = load_sim_data(config_obs)

    #setup model
    config_obs_sim['mcmc_npz_file'] = os.path.join(
        config_obs_sim['mcmc_out_dir'], config_obs_sim['mcmc_prefix'] + '_run_{0}.npz'.format(str(config_obs_sim['config'])))
    config_obs_sim['mcmc_backend_file'] = os.path.join(
        config_obs_sim['mcmc_out_dir'], config_obs_sim['mcmc_prefix'] + '_run_{0}.hdf5'.format(str(config_obs_sim['config'])))


    return config_obs_sim, data_obs, data_sim

################################################################################
# Measurements
################################################################################
def predict_model(param, config, obs_data, sim_data,
                  smf_only=False, ds_only=False):
    """Return all model predictions.
    Parameters
    ----------
    param: list, array, or tuple.
        Input model parameters.
    config : dict
        Configurations of the data and model.
    obs_data: dict
        Dictionary for observed data.
    sim_data: dict
        Dictionary for UniverseMachine data.
    constant_bin : boolen
        Whether to use constant bin size for logMs_tot or not.
    return_all : bool, optional
        Return all model information.
    show_smf : bool, optional
        Show the comparison of SMF.
    show_dsigma : bool, optional
        Show the comparisons of WL.
    """

    mass_x_field = config['sim_mass_x_field']
    halotools = config['sim_halotools']

    if halotools:
        print("USING HALOTOOLS with halo_mvir")
    # build_model and populate mock
        if 'model' in sim_data: # save memory if model already exists
            for i, model_param in enumerate(config['param_labels']):
                sim_data['model'].param_dict[model_param] = param[i]

            # set redshift dependence to 0
            for i, model_param in enumerate(config['redshift_param_labels']):
                sim_data['model'].param_dict[model_param] = 0

            sim_data['model'].mock.populate()
            print('mock.populate')

        else:
            sim_data['model'] = PrebuiltSubhaloModelFactory('behroozi10',
                                                            redshift=config['sim_z'],
                                                            scatter_abscissa=[12, 15],
                                                            scatter_ordinates=[param[0], param[1]])

            for i, model_param in enumerate(config['param_labels']):
                sim_data['model'].param_dict[model_param] = param[i]

            # set redshift dependence to 0
            for i, model_param in enumerate(config['redshift_param_labels']):
                sim_data['model'].param_dict[model_param] = 0

            # populate mock
            # sim_data['model'].populate_mock(deepcopy(sim_data['halocat']))
            sim_data['model'].populate_mock(sim_data['halocat'])
            print('populate_mock')

        print(sim_data['model'].param_dict)

        stellar_masses =  np.log10(sim_data['model'].mock.galaxy_table['stellar_mass'])

    else: #use Chris' code instead of halotools
        print("USING CHRIS' CODE with {0}".format(mass_x_field))
        stellar_masses = get_chris_stellar_masses(param, config, sim_data)

    # Predict SMFs
    smf_mass_bins, smf_log_phi = compute_SMF(stellar_masses, config, nbins=100)
    print('SMF computed')
    if smf_only:
        return smf_mass_bins, smf_log_phi, None, None, stellar_masses

    # Predict DeltaSigma profiles
    wl_r, wl_ds = compute_deltaSigma(stellar_masses, config, obs_data, sim_data)
    print('DS computed')
    if ds_only:
        return None, None, wl_r, wl_ds, stellar_masses

    return smf_mass_bins, smf_log_phi, wl_r, wl_ds, stellar_masses

def compute_SMF(log_stellar_masses, config, nbins=100):

    # Survey volume in Mpc3
    L=config['sim_lbox']
    h0=config['sim_h0']
    V = (L/h0)**3

    # Unnormalized histogram and bin edges
    Phi,edg = np.histogram(log_stellar_masses,bins=nbins)

    # Bin size
    dM    = edg[1] - edg[0]
    bin_centers   = edg[0:-1] + dM/2.

    # Normalize to volume and bin size
    Phi   = Phi / float(V) / dM
    logPhi= np.log10(Phi)

    return bin_centers, logPhi

def compute_deltaSigma(stellar_masses, config, cosmos_data, sim_data):

    # n_nearest = 40
    n_nearest = 90

    # select subsample of dwarfs from galaxy catalog
    # galaxies have the same positions as subhalos
    mock_galaxies = sim_data['halocat'].halo_table['halo_x', 'halo_y', 'halo_z']
    mock_galaxies['stellar_mass'] = stellar_masses
    mock_galaxies = np.array(mock_galaxies[(mock_galaxies['stellar_mass']>=min(cosmos_data['cosmos_dwarf_masses'])) & \
                                  (mock_galaxies['stellar_mass']<9.0)])
    # half_mock_galaxies = np.random.choice(mock_galaxies,500000)
    print('cut galaxies table', len(mock_galaxies))

    #if mock catalog does not contain enough dwarfs, return 0 probability
    if len(mock_galaxies) < len(cosmos_data['cosmos_dwarf_masses'])*(n_nearest*1.25):
        print('Too few mock dwarfs!')
        return 0, 0

    galaxies_table= create_dwarf_catalog_with_matched_mass_distribution(cosmos_data['cosmos_dwarf_masses'],
                                                                    mock_galaxies,
                                                                    n_nearest = n_nearest)

    print(ks_2samp(galaxies_table['stellar_mass'],cosmos_data['cosmos_dwarf_masses']))
    if ks_2samp(galaxies_table['stellar_mass'],cosmos_data['cosmos_dwarf_masses'])[1] < 0.95:
        print('Mock and COSMOS distributions don\'t match!')
        return 0, 0

    # read in galaxy positions
    x = galaxies_table['halo_x']
    y = galaxies_table['halo_y']
    z = galaxies_table['halo_z']
    galaxies = np.vstack((x, y, z)).T

    # mass enclosed by cylinders around each galaxy
    period=np.ones(3)*config['sim_lbox']
    r_bins = np.logspace(-2.1,0,20)

    mass_encl = total_mass_enclosed_per_cylinder(galaxies, sim_data['particles'],
                                                 sim_data['particle_masses'],
                                                 sim_data['downsampling_factor'],
                                                 r_bins, period)

    # delta Sigma
    rp, ds = delta_sigma_from_precomputed_pairs(galaxies, mass_encl, r_bins,
                                                period, cosmology=config['sim_cosmo'])

    #convert to correct physical units
    ds = ds/1e12 # convert units pc^-2 --> Mpc^-2
    ds = ds*config['sim_h0']*((1+config['sim_z'])**2) #convert from comoving h-inverse to physical real-h
    rp = rp / float(config['sim_h0']*(1+config['sim_z'])) #convert from comoving h-inverse to physical real-h

    return rp, ds

################################################################################
# Plotting
################################################################################
def plot_SMF(sim_mass_centers, sim_logPhi, cosmos_SMF_points_table,
             cosmos_SMF_fit_table, labels=None):

    plt.figure(figsize=[10,8])
    # plot COSMOS
    plt.errorbar(cosmos_SMF_points_table['logM'], cosmos_SMF_points_table['Phi'],
                 yerr=[cosmos_SMF_points_table['Phi_err+'],
                 cosmos_SMF_points_table['Phi_err-']], fmt='o', elinewidth=3,
                 markersize=8, label='COSMOS z~0.2 data',
                 zorder=3)
     # plot sim
    plt.plot(sim_mass_centers, sim_logPhi, label='Best fit model',
             linewidth = 3, linestyle='--')

    #plot details
    plt.xlabel(r'$\log(M_\star [M_{\odot}])$', fontsize=20)
    plt.ylabel(r'$\log(dN/dM_\star [Mpc^{-3} dex^{-1}])$', fontsize=20)
    plt.xlim([8,11.6])
    plt.ylim([-4.5,-1.25])
    #labels for legend. matplotlib does not have an easy way of doing this when plotting multiple arrays at once
    if not labels:
         plt.legend(loc='lower left', fontsize=15)
    else:
        plt.legend(labels+['COSMOS z~0.2 data'], loc='lower left', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.savefig('/Users/fardila/Documents/GitHub/dwarf_lensing/MCMC/plots/SMF.pdf',
                dpi=100, format='pdf')

    plt.show()

def plot_deltaSigma(observed_signal_table, sim_r, sim_ds, labels=None):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    plt.loglog()

    #plot observations
    ax.errorbar(observed_signal_table['R(Mpc)'], observed_signal_table['SigR(Msun/pc^2)'],
                yerr=observed_signal_table['err(weights)'], marker='o',
                label=r'COSMOS data', linewidth=3, zorder=1)

    #plot sim
    ax.plot(sim_r, sim_ds, linestyle='--', zorder=3, marker='o',
            label='Best fit model')
    # ax.fill_between(sim_r, sim_ds,sim_ds, alpha=0.5)


    # ax.fill_between(cosmos_lensing_signal['R(Mpc)'], cosmos_lensing_signal['SigR(Msun/pc^2)']+cosmos_lensing_signal['err(weights)'],
    #                 cosmos_lensing_signal['SigR(Msun/pc^2)']-cosmos_lensing_signal['err(weights)'], alpha=0.5)

    __=ax.set_xlim(xmin = 0.01, xmax = 1)
    # __=ax.set_ylim(ymin = 0.5, ymax = 200)

    __=ax.set_xlabel(r'$R_{\rm p} $  $\rm{[Mpc]}$', fontsize=16)
    __=ax.set_ylabel(r'$\Delta\Sigma(R_{\rm p})$  $[M_{\odot} / {\rm pc}^2]$', fontsize=16)
    if not labels:
         __=ax.legend( loc='best', fontsize=13)
    else:
        __=ax.legend(labels+['COSMOS z~0.2 data'], loc='best', fontsize=13)
    __=plt.xticks(fontsize=15); plt.yticks(fontsize=15)

    # plt.title('Matched Mass distribution')
    plt.show()

def plot_from_params(params, config, cosmos_data, sim_data, labels=None):
    if isinstance(params[0], list): #more than one param tuple, i.e. comparing models
        for param_tuple in params:
            print(param_tuple)
        model_data = [predict_model(param_tuple, config, cosmos_data, sim_data) for param_tuple in params]

        smf_mass_bins = np.array([d[0] for d in model_data])
        smf_log_phi = np.array([d[1] for d in model_data])
        wl_r = np.array([d[2] for d in model_data])
        wl_ds = np.array([d[3] for d in model_data])
        plot_SMF(smf_mass_bins.T, smf_log_phi.T, cosmos_data['cosmos_SMF_points_table'],
                 cosmos_data['cosmos_SMF_fit_table'], labels=labels)
        plot_deltaSigma(cosmos_data['cosmos_wl_table'], wl_r.T, wl_ds.T,
                        labels=labels)
    else: #single param tuple
        param_tuple = params
        print(param_tuple)
        smf_mass_bins, smf_log_phi, wl_r, wl_ds = predict_model(params, config, cosmos_data, sim_data)
        plot_SMF(smf_mass_bins, smf_log_phi, cosmos_data['cosmos_SMF_points_table'],
                 cosmos_data['cosmos_SMF_fit_table'])
        plot_deltaSigma(cosmos_data['cosmos_wl_table'], wl_r, wl_ds)

################################################################################
# Probability functions
################################################################################
def flat_prior(param_tuple, param_low, param_upp):
    """Priors of parameters. Return -inf if all parameters are not within bounds."""
    if not np.all([low <= param <= upp for param, low, upp in
                   zip(list(param_tuple), param_low, param_upp)]):
        print('All parameters not within bounds!')
        return -np.inf

    return 0.0

def smf_lnlike(obs_smf_points_table, sim_smf_mass_bins, sim_smf_log_phi):
    """Calculate the likelihood for SMF."""
    print('smf_lnlike')
    # get same bins in simulations as in observations
    non_inf_mask=~np.isinf(sim_smf_log_phi) #remove any infinities
    sim_smf_log_phi_interpolated = np.interp(obs_smf_points_table['logM'],
                                            sim_smf_mass_bins[non_inf_mask],
                                             sim_smf_log_phi[non_inf_mask])

    # difference
    smf_diff = (np.array(sim_smf_log_phi_interpolated) - np.array(obs_smf_points_table['Phi']))

    # variance
    obs_mean_smf_error = np.nanmean([obs_smf_points_table['Phi_err+'],
                                    obs_smf_points_table['Phi_err-'] ], axis=0)
    smf_var = np.array(obs_mean_smf_error ** 2)

    # chi2
    smf_chi2 = (smf_diff ** 2 / smf_var).sum()

    # likelihood
    smf_lnlike = -0.5 * (smf_chi2 + np.log(2 * np.pi * smf_var).sum())


    # print("SMF Tot lnlike/chi2: %f,%f" % (smf_mtot_lnlike,
    #                                       smf_mtot_chi2))


    return smf_lnlike

def dsigma_lnlike(obs_wl_table, sim_wl_r, sim_wl_ds, cosmos_data):
    """Calculate the likelihood for WL profile."""
    print('dsigma_lnlike')

    #0 when number of dwarfs in mock is less than COSMOS dwarf catalog
    if np.all(sim_wl_r == 0) and np.all(sim_wl_ds == 0):
        return -np.inf

    r_obs = obs_wl_table['R(Mpc)']
    dsigma_obs = obs_wl_table['SigR(Msun/pc^2)']
    dsigma_obs_err = obs_wl_table['err(weights)']
    dsigma_var = (dsigma_obs_err ** 2)

    # interpolate to same bins
    sim_wl_ds_interpolated = np.interp(r_obs, sim_wl_r, sim_wl_ds)

    dsigma_diff = (dsigma_obs - sim_wl_ds_interpolated) ** 2

    dsigma_chi2 = (dsigma_diff / dsigma_var).sum()

    dsigma_lnlike = -0.5 * (dsigma_chi2 + np.log(2 * np.pi * dsigma_var).sum())
    # print("DSigma likelihood / chi2: %f, %f" % (dsigma_lnlike, dsigma_chi2))

    return dsigma_lnlike

def ln_like(param_tuple, config, obs_data, sim_data,
            smf_only=False, ds_only=False):
    """Calculate the lnLikelihood of the model."""
    # Unpack the input parameters
    parameters = list(param_tuple)
    print('lnlike')
    # print(datetime.datetime.now())
    # h=hpy()
    # print(h.heap())

    # Generate the model predictions
    model_outputs = predict_model(parameters, config, obs_data, sim_data,
                                  smf_only, ds_only)

    sim_smf_mass_bins, sim_smf_log_phi, sim_wl_r, sim_wl_ds = model_outputs[:-1]
    print('model predicted')

    # Likelihood for SMFs.
    smf_lnlike_value = smf_lnlike(obs_data['cosmos_SMF_points_table'],
                                    sim_smf_mass_bins, sim_smf_log_phi)
    if smf_only:
        print('SMF ONLY')
        return smf_lnlike_value

    # Likelihood for DeltaSigma
    dsigma_lnlike_value = dsigma_lnlike(obs_data['cosmos_wl_table'],
                                        sim_wl_r, sim_wl_ds, obs_data)
    if ds_only:
        print('DS ONLY')
        return dsigma_lnlike_value

    # Likelihood Dark Deficient Doppelgangers
    logM_halo = sim_data['halocat'].halo_table['halo_mvir']
    logM_stellar = model_outputs[-1]
    logMvir, logcvir = parameters[7,8]
    vsys_df2, vsys_df4 = parameters[9,10]
    logUpsilon, sym_beta, f_Re = parameters[11:]
    ddd_lnlike_value = ddd_lnlike(logMvir_um, logMst_um,
                                  logMvir, logcvir,
                                  vsys_df2, vsys_df4,
                                  logUpsilon, sym_beta, f_Re)

    if not np.isfinite(smf_lnlike_value) or not np.isfinite(dsigma_lnlike_value):
        return -np.inf

    return smf_lnlike_value + dsigma_lnlike_value + ddd_lnlike_value

def ln_prob_global(param_tuple, config, cosmos_data, sim_data,
            smf_only=False, ds_only=False):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    param_tuple: tuple of model parameters.

    """

    if config['flat_scatter']:
        param_tuple[1] = param_tuple[0]
    if config['fixed_scatter']:
        param_tuple[0] = 0.2
        param_tuple[1] = 0.2

    print(param_tuple)

    lp = flat_prior(param_tuple, config['param_low'], config['param_upp'])

    if not np.isfinite(lp):
        return -np.inf

    l_like = ln_like(param_tuple, config, cosmos_data, sim_data, smf_only, ds_only)
    print(l_like)

    return lp + l_like

################################################################################
# MCMC functions
################################################################################
def mcmc_initial_guess(param_initial, param_sigma, n_walkers, n_dims):
    """Initialize guesses for the MCMC run. One guess for each dimension (model parameter) per walker,
    with a small sigma deviation from param_initial. """
    mcmc_position = np.zeros([n_walkers, n_dims])

    for ii, param_0 in enumerate(param_initial):
        mcmc_position[:, ii] = (param_0 + param_sigma[ii] * np.random.randn(n_walkers))

    return mcmc_position

def mcmc_setup_moves(config, move_col):
    """Choose the Move object for emcee."""
    if config[move_col] == 'snooker':
        emcee_moves = emcee.moves.DESnookerMove()
    elif config[move_col] == 'stretch':
        emcee_moves = emcee.moves.StretchMove(a=config['mcmc_stretch_a'],
                                                live_dangerously=config['mcmc_live_dangerously'])
    elif config[move_col] == 'walk':
        emcee_moves = emcee.moves.WalkMove()
    elif config[move_col] == 'kde':
        emcee_moves = emcee.moves.KDEMove()
    elif config[move_col] == 'de':
        emcee_moves = emcee.moves.DEMove(config['mcmc_de_sigma'])
    else:
        raise Exception("Wrong option: stretch, walk, kde, de, snooker")

    return emcee_moves

def mcmc_save_results(mcmc_position, mcmc_sampler, mcmc_file,
                      mcmc_ndims, verbose=True):
    """Save the MCMC run results."""

    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
        (-1, mcmc_ndims))
    mcmc_chains = mcmc_sampler.chain
    mcmc_lnprob = mcmc_sampler.lnprobability
    acceptance_fraction = mcmc_sampler.acceptance_fraction
    ind_1, ind_2 = np.unravel_index(np.argmax(mcmc_lnprob, axis=None),
                                    mcmc_lnprob.shape)
    mcmc_best = mcmc_chains[ind_2, ind_1, :]
#     mcmc_params_stats = mcmc_samples_stats(mcmc_samples)

    print(mcmc_file)
    np.savez(str(mcmc_file),
             samples=mcmc_samples, lnprob=np.array(mcmc_lnprob),
             best=np.array(mcmc_best), chains=mcmc_chains,
             position=np.asarray(mcmc_position),
             acceptance=np.array(acceptance_fraction))

    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(mcmc_sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" % np.max(mcmc_lnprob))
        print(mcmc_best)
        print("#------------------------------------------------------")
#         for param_stats in mcmc_params_stats:
#             print(param_stats)
#         print("#------------------------------------------------------")

    return

#skip paralleization for now
def emcee_fit(config, cosmos_data, sim_data, verbose=True,
              smf_only=False, ds_only=False):

    print('{0} thread(s)'.format(config['mcmc_nthreads']))

    # Initialize the model
    mcmc_ini_position = mcmc_initial_guess(
        config['param_ini'], config['param_sig'], config['mcmc_nwalkers'],
        config['mcmc_ndims'])

    if config['mcmc_nthreads'] > 1:
        pass
        # from multiprocessing import Pool
        # from contextlib import closing
        #
        # with closing(Pool(processes=config['mcmc_nthreads'])) as pool:
        #
        #     # Decide the Ensemble moves for walkers during burnin
        #     burnin_move = mcmc_setup_moves(config, 'mcmc_moves_burnin')
        #
        #     burnin_sampler = emcee.EnsembleSampler(
        #         config['mcmc_nwalkers_burnin'],
        #         config['mcmc_ndims'],
        #         ln_prob_global,
        #         moves=burnin_move,
        #         pool=pool,
        #         args = [config, cosmos_data, sim_data])
        #
        #     # Burn-in
        #     mcmc_burnin_pos, mcmc_burnin_lnp, mcmc_burnin_state = mcmc_burnin(
        #         burnin_sampler, mcmc_ini_position, config, verbose=True)
        #
        #     # Estimate the Kernel density distributions of final brun-in positions
        #     # Resample the distributions to get starting positions of the actual run
        #     mcmc_kde = gaussian_kde(np.transpose(mcmc_burnin_pos),
        #                        bw_method='silverman')
        #     mcmc_new_pos = np.transpose(mcmc_kde.resample(config['mcmc_nwalkers']))
        #
        #     mcmc_new_ini = (mcmc_new_pos, mcmc_burnin_lnp, mcmc_burnin_state)
        #
        #     # TODO: Convergence test
        #     burnin_sampler.reset()
        #
        #     # Change the moves
        #     # Decide the Ensemble moves for walkers during the official run
        #     mcmc_move = mcmc_setup_moves(config, 'mcmc_moves')
        #
        #     mcmc_sampler = emcee.EnsembleSampler(
        #         config['mcmc_nwalkers'],
        #         config['mcmc_ndims'],
        #         ln_prob_global,
        #         moves=mcmc_move,
        #         pool=pool,
        #         args = [config, cosmos_data, sim_data])
        #
        #     # MCMC run
        #     mcmc_run_result = emcee_run(
        #         mcmc_sampler, mcmc_new_ini, config, verbose=True)
    else:

        # Decide the Ensemble moves for walkers during burnin
        mcmc_move = mcmc_setup_moves(config, 'mcmc_moves')

        # Set up the backend to save results
        # Don't forget to clear it in case the file already exists
        filename = config['mcmc_backend_file']
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(config['mcmc_nwalkers'], config['mcmc_ndims'])

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(config['mcmc_nwalkers'],
                                             config['mcmc_ndims'],
                                             ln_prob_global,
                                             moves=mcmc_move,
                                             args = [config, cosmos_data, sim_data,
                                                         smf_only, ds_only],
                                             backend=backend)

        if verbose:
            print("# Phase: MCMC run ...")

        for sample in sampler.sample(mcmc_ini_position,
                                              iterations=config['mcmc_nsamples'],
                                              progress=True,
                                              store= True):
            mcmc_save_results(sample.coords, sampler,
                              config['mcmc_npz_file'], config['mcmc_ndims'],
                              verbose=True)



    return

################################################################################
# Helper functions
################################################################################
## matching dwarf catalog with mock functions
def find_nearest_new_indices_sorted(index, n, existing_indices, length):
    """
    index: index of closest value
    n: number of closest that we want
    existing_indices: indices that already exist to avoid repeats
    length: of array from which we are sampling (i.e. mock_galaxies)

    return:
    indices of n nearest rows
    """

    # indices nearest to index
    nearest_indices=[]

    # not already in subsample and index within sample
    if index not in existing_indices and index <length:
            nearest_indices.append(index)

    #append next nearest indices
    i=0
    while len(nearest_indices) < n:
        # print('length of nearest: ', len(nearest_indices) )
        # print('distance from nearest index (i) ', i)

        if index + i not in existing_indices and \
           index + i not in nearest_indices and \
           index + i < length:
            # print('above')
            nearest_indices.append(index + i)

        if len(nearest_indices) < n and \
           index - i not in existing_indices and \
           index - i not in nearest_indices and \
           index - i >= 0 and \
           index - i < length:
            # print('below')
            nearest_indices.append(index - i)

        i += 1

        # if it can't find nearest indices within 10000 of nearest, then just
        # return what we have. if the distributions are dissimilar it won't
        # accept anyways
        if i > 10000:
            return nearest_indices


    return nearest_indices

def create_dwarf_catalog_with_matched_mass_distribution(dwarf_masses, mock_galaxies, n_nearest):
    """
    dwarf_masses: masses of COSMOS dwarfs
    mock_galaxies: array of mock galaxies from halotools
    n_nearest: number of nearest mock galaxies to use (i.e. number of times to
               sample full COSMOS dwarf catalog)

    return:
    array of the sample of mock dwarfs with an identical distribution
    of masses as COSMOS
    """

    # set for speeding up finding location
    subsample_indices=set()

    # for debugging
    # import pdb;  pdb.set_trace()
    # sort first to speed up future calculations
    mock_galaxies.sort(order = str('stellar_mass'))

    # indices of single nearest mock mass for each dwarf mass
    indices = np.searchsorted(mock_galaxies['stellar_mass'], dwarf_masses)

    #for each dwarf mass, add additional nearest indices
    for i, index in enumerate(indices):

        #return the n_nearest indices closest to index (the index of closest mass)
        matched_indices = find_nearest_new_indices_sorted(index,
                                                          n_nearest,
                                                          subsample_indices,
                                                          len(mock_galaxies))

        #add elements of matched_indices to subsample_indices
        subsample_indices.update(matched_indices)

        # if i==3: print(foo)

    subsample = mock_galaxies[list(subsample_indices)]

    return subsample

def GM_data_location(config):
    '''
    Defines data location in GRAYMALKIN
    '''
    config['data_location'] = '/data/ateam/fardila/dwarf_lensing/'

    return config

################################################################################
# Functions from Chris
################################################################################
def get_chris_stellar_masses(params, config, sim_data):
    """params : 2 scatter params + 5 SHMR params
        x_field: 'halo_mvir' or "halo_Vmax@Mpeak" """

    x_field = config['sim_mass_x_field']
    halo_data = sim_data['halocat'].halo_table
    #2 params when testing
    if len(params) == 2:
        scatter_params = params
        if x_field == 'halo_mvir':
            b_params = [12.52, 10.91, 0.45, 0.6, 1.83]
        elif x_field == 'halo_Vmax@Mpeak':
            b_params = [2.4, 10.91, 0.45, 0.3, 0.2]

        return get_sm_for_sim(halo_data, b_params, scatter_params, x_field)

    #convert scatter parameters from 2 points to slope and intercept
    if x_field == 'halo_mvir':
        scatter_range = [12,15]
    elif x_field == 'halo_Vmax@Mpeak':
        scatter_range = [2,3]

    scatter_params = np.polyfit(scatter_range,[params[0],params[1]],1)
    b_params = params[2:]

    return get_sm_for_sim(halo_data, b_params, scatter_params, x_field)
