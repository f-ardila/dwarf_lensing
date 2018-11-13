from __future__ import division, print_function, unicode_literals

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import emcee
print(emcee.__version__) #need version 3 (pip install emcee==3.0rc1)
import yaml

from scipy.stats import gaussian_kde, ks_2samp

from astropy.table import Table, Column
from astropy.io import ascii, fits
from astropy.cosmology import FlatLambdaCDM

from halotools.sim_manager import TabularAsciiReader, HaloTableCache, CachedHaloCatalog
from halotools.sim_manager.rockstar_hlist_reader import RockstarHlistReader
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.mock_observables import delta_sigma_from_precomputed_pairs, total_mass_enclosed_per_cylinder
from halotools.utils import randomly_downsample_data

#memory profile
from guppy import hpy
import datetime


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
    cosmos_dwarf_sample_data  = fits.open(os.path.join(cfg['data_location'], cfg['cosmos_dir'],
                                       cfg['cosmos_dwarf_file']))[1].data
    cosmos_dwarf_masses = cosmos_dwarf_sample_data['mass_med']

    #SMF
    cosmos_SMF_fit_table = ascii.read(os.path.join(cfg['data_location'], cfg['cosmos_dir'],
                                       cfg['cosmos_SMF_fit_file']))
    cosmos_SMF_points_table = ascii.read(os.path.join(cfg['data_location'], cfg['cosmos_dir'],
                                       cfg['cosmos_SMF_points_file']))

    #weak lensing
    cosmos_lensing_table = ascii.read(os.path.join(cfg['data_location'], cfg['cosmos_dir'], cfg['cosmos_wl_file']))

    cosmos_wl_r = cosmos_lensing_table['R(Mpc)']
    cosmos_wl_ds = cosmos_lensing_table['SigR(Msun/pc^2)']

    #cosmology
    cfg['cosmos_cosmo'] = FlatLambdaCDM(H0=cfg['cosmos_h0'] * 100,
                                     Om0=cfg['cosmos_omega_m'])

#     obs_volume = ((cfg['obs_cosmo'].comoving_volume(obs_zmax) -
#                    cfg['obs_cosmo'].comoving_volume(obs_zmin)) *
#                   (cfg['obs_area'] / 41254.0)).value
#     cfg['obs_volume'] = obs_volume

    return {'cosmos_wl_r': cosmos_wl_r, 'cosmos_wl_ds': cosmos_wl_ds, 'cosmos_wl_table': cosmos_lensing_table,
            'cosmos_SMF_fit_table': cosmos_SMF_fit_table, 'cosmos_SMF_points_table': cosmos_SMF_points_table,
             'cosmos_dwarf_masses': cosmos_dwarf_masses}, cfg

def load_sim_data(cfg):
    """Load the UniverseMachine data."""


    #read in halocat
    halocat = CachedHaloCatalog(simname = cfg['sim_name'], halo_finder = cfg['sim_halo_finder'],
                            version_name = cfg['sim_version_name'], redshift = cfg['sim_z'],
                                ptcl_version_name=cfg['sim_ptcl_version_name']) # doctest: +SKIP

    #read in particle table
    ptcl_table = Table.read(os.path.join(cfg['data_location'], cfg['sim_dir'], cfg['sim_particle_file']), path='data')
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
    config_obs_sim['mcmc_burnin_file'] = os.path.join(
        config_obs_sim['mcmc_out_dir'], config_obs_sim['mcmc_prefix'] + '_burnin_{0}.npz'.format(str(config_obs_sim['config'])))
    config_obs_sim['mcmc_run_file'] = os.path.join(
        config_obs_sim['mcmc_out_dir'], config_obs_sim['mcmc_prefix'] + '_run_{0}.npz'.format(str(config_obs_sim['config'])))


    return config_obs_sim, data_obs, data_sim

################################################################################
# Measurements
################################################################################
def compute_SMF(model, config, nbins=100):

    # Read stellar masses
    M = model.mock.galaxy_table['stellar_mass']
    print('new')
    #M = M[(M>1) & (M<np.inf)] #apparently some models produce galaxies with 0 mass and some with tiny mass and some with infinite mass

    # Take logarithm
    logM = np.log10(M)

    # Survey volume in Mpc3
    L=config['sim_lbox']
    h0=config['sim_h0']
    V = (L/h0)**3

    # Unnormalized histogram and bin edges
    Phi,edg = np.histogram(logM,bins=nbins)

    # Bin size
    dM    = edg[1] - edg[0]
    bin_centers   = edg[0:-1] + dM/2.

    # Normalize to volume and bin size
    Phi   = Phi / float(V) / dM
    logPhi= np.log10(Phi)

    return bin_centers, logPhi

def compute_deltaSigma(model, config, cosmos_data, sim_data):

    n_nearest = 40

    # select subsample of dwarfs from galaxy catalog
    mock_galaxies = model.mock.galaxy_table
    mock_galaxies = mock_galaxies['x', 'y', 'z', 'stellar_mass']
    mock_galaxies = np.array(mock_galaxies[(np.log10(mock_galaxies['stellar_mass'])>=min(cosmos_data['cosmos_dwarf_masses'])) & \
                                  (np.log10(mock_galaxies['stellar_mass'])<9.0)])
    # half_mock_galaxies = np.random.choice(mock_galaxies,500000)
    print('cut galaxies table', len(mock_galaxies))

    #if mock catalog does not contain enough dwarfs, return 0 probability
    if len(mock_galaxies) < len(cosmos_data['cosmos_dwarf_masses'])*(n_nearest*1.25):
        print('Too few mock dwarfs!')
        return 0, 0

    galaxies_table= create_dwarf_catalog_with_matched_mass_distribution(cosmos_data['cosmos_dwarf_masses'],
                                                                    mock_galaxies,
                                                                    n_nearest = n_nearest)

    print(ks_2samp(np.log10(galaxies_table['stellar_mass']),cosmos_data['cosmos_dwarf_masses']))
    if ks_2samp(np.log10(galaxies_table['stellar_mass']),cosmos_data['cosmos_dwarf_masses'])[1] < 0.95:
        print('Mock and COSMOS distributions don\'t match!')
        return 0, 0

    # read in galaxy positions
    x = galaxies_table['x']
    y = galaxies_table['y']
    z = galaxies_table['z']
    galaxies = np.vstack((x, y, z)).T

    # mass enclosed by cylinders around each galaxy
    period=model.mock.Lbox
    r_bins=[float(r) for r in config['sim_wl_bins'].split()]

    mass_encl = total_mass_enclosed_per_cylinder(galaxies, sim_data['particles'], sim_data['particle_masses'],
                                                 sim_data['downsampling_factor'], r_bins, period)

    # delta Sigma
    rp, ds = delta_sigma_from_precomputed_pairs(galaxies, mass_encl, r_bins, period, cosmology=config['sim_cosmo'])
    ds = ds/1e12 # there seems to be a discrepancy in units pc^2 --> Mpc^2

    return rp, ds


def predict_model(param, config, obs_data, sim_data,
                       constant_bin=False, return_all=False,
                       show_smf=False, show_dsigma=False):
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
        sim_data['model'] = PrebuiltSubhaloModelFactory('behroozi10', redshift=config['sim_z'],
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

    print(sim_data['model'].param_dict())

    # Predict SMFs
    smf_mass_bins, smf_log_phi = compute_SMF(sim_data['model'],config, nbins=100)
    print('SMF computed')

    # Predict DeltaSigma profiles
    wl_r, wl_ds = compute_deltaSigma(sim_data['model'], config, obs_data, sim_data)
    print('DS computed')


#     if show_smf:
#         um_smf_tot_all = get_smf_bootstrap(logms_mod_tot_all,
#                                            cfg['um_volume'],
#                                            20, cfg['obs_min_mtot'], 12.5,
#                                            n_boots=1)
#         _ = plot_mtot_minn_smf(
#             obs_data['obs_smf_tot'], obs_data['obs_smf_inn'],
#             obs_data['obs_mtot'], obs_data['obs_minn'],
#             um_smf_tot, um_smf_inn,
#             logms_mod_tot, logms_mod_inn,
#             obs_smf_full=obs_data['obs_smf_full'],
#             um_smf_tot_all=um_smf_tot_all,
#             not_table=True)

#     if show_dsigma:
#         um_mhalo_tuple = asap_predict_mhalo(
#             obs_data['obs_wl_dsigma'], um_data['um_mock'][mask_mtot],
#             logms_mod_tot, logms_mod_inn)
#         _ = plot_dsigma_profiles(obs_data['obs_wl_dsigma'],
#                                  um_dsigma_profs, obs_mhalo=None,
#                                  um_mhalo=um_mhalo_tuple)



    return smf_mass_bins, smf_log_phi, wl_r, wl_ds

################################################################################
# Plotting
################################################################################
def plot_SMF(sim_mass_centers, sim_logPhi, cosmos_SMF_points_table, cosmos_SMF_fit_table):

    # plot sim
    plt.plot(sim_mass_centers, sim_logPhi, c='r', label='Bolshoi-Planck halos')

    # plot COSMOS
    plt.plot(cosmos_SMF_fit_table['log_m'], cosmos_SMF_fit_table['log_phi'], label='COSMOS z~0.2 fit')
    plt.fill_between(cosmos_SMF_fit_table['log_m'], cosmos_SMF_fit_table['log_phi_inf'],
                     cosmos_SMF_fit_table['log_phi_sup'], alpha=0.5)
    plt.errorbar(cosmos_SMF_points_table['logM'], cosmos_SMF_points_table['Phi'],
                 yerr=[cosmos_SMF_points_table['Phi_err+'],cosmos_SMF_points_table['Phi_err-']], fmt='o', elinewidth=3,
                markersize=5, c='#1f77b4', label='COSMOS z~0.2 points')

    #plot details
    plt.xlabel('log(M)')
    plt.ylabel('log(Phi)')
    plt.xlim([8,12.5])
    plt.ylim([-10,0])
    plt.legend(loc='lower left')

    plt.show()

def plot_deltaSigma(observed_signal_table, sim_r, sim_ds):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    plt.loglog()

    #plot sim
    ax.plot(sim_r, sim_ds, label=r'Bol-Planck: ', linestyle='--', zorder=3, marker='o')
    # ax.fill_between(sim_r, sim_ds,sim_ds, alpha=0.5)

    #plot observations
    ax.errorbar(observed_signal_table['R(Mpc)'], observed_signal_table['SigR(Msun/pc^2)'],
                yerr=observed_signal_table['err(weights)'], marker='o', label=r'Laigle+2016: ', linewidth=3, zorder=1)
    # ax.fill_between(cosmos_lensing_signal['R(Mpc)'], cosmos_lensing_signal['SigR(Msun/pc^2)']+cosmos_lensing_signal['err(weights)'],
    #                 cosmos_lensing_signal['SigR(Msun/pc^2)']-cosmos_lensing_signal['err(weights)'], alpha=0.5)

    __=ax.set_xlim(xmin = 0.01, xmax = 1)
    # __=ax.set_ylim(ymin = 0.5, ymax = 200)

    __=ax.set_xlabel(r'$R_{\rm p} $  $\rm{[Mpc / h]}$', fontsize=16)
    __=ax.set_ylabel(r'$\Delta\Sigma(R_{\rm p})$  $[h M_{\odot} / {\rm pc}^2]$', fontsize=16)
    __=ax.legend(loc='best', fontsize=13)
    __=plt.xticks(fontsize=15); plt.yticks(fontsize=15)

    # plt.title('Matched Mass distribution')
    plt.show()

def plot_from_params(params, config, cosmos_data, sim_data):
    smf_mass_bins, smf_log_phi, wl_r, wl_ds = predict_model(params, config, cosmos_data, sim_data)
    plot_SMF(smf_mass_bins, smf_log_phi, cosmos_data['cosmos_SMF_points_table'], cosmos_data['cosmos_SMF_fit_table'])
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

def smf_lnlike(obs_smf_fit_table, obs_smf_points_table, sim_smf_mass_bins, sim_smf_log_phi):
    """Calculate the likelihood for SMF."""
    print('smf_lnlike')
    # get same bins in simulations as in observations
    sim_smf_log_phi_interpolated = np.interp(obs_smf_points_table['logM'], sim_smf_mass_bins, sim_smf_log_phi)

    # difference
    smf_diff = (np.array(sim_smf_log_phi_interpolated) - np.array(obs_smf_points_table['Phi']))

    # variance
    obs_mean_smf_error = np.mean([obs_smf_points_table['Phi_err+'] , obs_smf_points_table['Phi_err-'] ])
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

    # make sure same bins
    decimal_places_to_round_bins = 5
    assert np.all(sim_wl_r.round(decimal_places_to_round_bins) == \
                  np.array(cosmos_data['cosmos_wl_table']['R(Mpc)']).round(decimal_places_to_round_bins))

    dsigma_obs = obs_wl_table['SigR(Msun/pc^2)']
    dsigma_obs_err = obs_wl_table['err(weights)']

    dsigma_var = (dsigma_obs_err ** 2)
    dsigma_diff = (dsigma_obs - sim_wl_ds) ** 2

    dsigma_chi2 = (dsigma_diff / dsigma_var).sum()
    dsigma_lnlike = -0.5 * (dsigma_chi2 + np.log(2 * np.pi * dsigma_var).sum())
    # print("DSigma likelihood / chi2: %f, %f" % (dsigma_lnlike, dsigma_chi2))

    return dsigma_lnlike

def ln_like(param_tuple, config, obs_data, sim_data, chi2=False,
                 sep_return=False):
    """Calculate the lnLikelihood of the model."""
    # Unpack the input parameters
    parameters = list(param_tuple)
    print('lnlike')
    # print(datetime.datetime.now())
    # h=hpy()
    # print(h.heap())

    # Generate the model predictions
    sim_smf_mass_bins, sim_smf_log_phi, sim_wl_r, sim_wl_ds = predict_model(parameters, config, obs_data, sim_data)

    print('model_prediced')
    # Likelihood for SMFs.
    smf_lnlike_value = smf_lnlike(obs_data['cosmos_SMF_fit_table'], obs_data['cosmos_SMF_points_table'],
        sim_smf_mass_bins, sim_smf_log_phi)

    # Likelihood for DeltaSigma
    dsigma_lnlike_value = dsigma_lnlike(obs_data['cosmos_wl_table'], sim_wl_r, sim_wl_ds, obs_data)

    if not np.isfinite(smf_lnlike_value) or not np.isfinite(dsigma_lnlike_value):
        return -np.inf

    return smf_lnlike_value + config['mcmc_wl_weight'] * dsigma_lnlike_value

def ln_prob_global(param_tuple, config, cosmos_data , sim_data):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    param_tuple: tuple of model parameters.

    """
    print(param_tuple)

    lp = flat_prior(param_tuple, config['param_low'], config['param_upp'])

    if not np.isfinite(lp):
        return -np.inf


    return lp + ln_like(param_tuple, config, cosmos_data, sim_data)

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

def mcmc_burnin(mcmc_sampler, mcmc_position, config, verbose=True):
    """Run the MCMC chain."""
    # Burn-in
    if verbose:
        print("# Phase: Burn-in ...")

    mcmc_burnin_result = mcmc_sampler.run_mcmc(
        mcmc_position, config['mcmc_nburnin'],
        progress=True)

    mcmc_save_results(mcmc_burnin_result, mcmc_sampler,
                      config['mcmc_burnin_file'], config['mcmc_ndims'],
                      verbose=True)

    # Rest the chains
    mcmc_sampler.reset()

    return mcmc_burnin_result

def emcee_run(mcmc_sampler, mcmc_burnin_result, config, verbose=True):
    """Run the MCMC chain."""
    mcmc_burnin_position, _, mcmc_burnin_state = mcmc_burnin_result

    if verbose:
        print("# Phase: MCMC run ...")
    mcmc_run_result = mcmc_sampler.run_mcmc(
        mcmc_burnin_position, config['mcmc_nsamples'],
        rstate0=mcmc_burnin_state,
        progress=True)

    mcmc_save_results(mcmc_run_result, mcmc_sampler,
                      config['mcmc_run_file'], config['mcmc_ndims'],
                      verbose=True)

    return mcmc_run_result

def mcmc_save_results(mcmc_results, mcmc_sampler, mcmc_file,
                      mcmc_ndims, verbose=True):
    """Save the MCMC run results."""
    (mcmc_position, mcmc_lnprob, _) = mcmc_results

    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
        (-1, mcmc_ndims))
    mcmc_chains = mcmc_sampler.chain
    mcmc_lnprob = mcmc_sampler.lnprobability
    ind_1, ind_2 = np.unravel_index(np.argmax(mcmc_lnprob, axis=None),
                                    mcmc_lnprob.shape)
    mcmc_best = mcmc_chains[ind_2, ind_1, :]
#     mcmc_params_stats = mcmc_samples_stats(mcmc_samples)

    np.savez(mcmc_file,
             samples=mcmc_samples, lnprob=np.array(mcmc_lnprob),
             best=np.array(mcmc_best), chains=mcmc_chains,
             position=np.asarray(mcmc_position),
             acceptance=np.array(mcmc_sampler.acceptance_fraction))

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
def emcee_fit(config, cosmos_data, sim_data, verbose=True):

    print('{0} thread(s)'.format(config['mcmc_nthreads']))

    # Initialize the model
    mcmc_ini_position = mcmc_initial_guess(
        config['param_ini'], config['param_sig'], config['mcmc_nwalkers'],
        config['mcmc_ndims'])

    if config['mcmc_nthreads'] > 1:
        from multiprocessing import Pool
        from contextlib import closing

        with closing(Pool(processes=config['mcmc_nthreads'])) as pool:

            # Decide the Ensemble moves for walkers during burnin
            burnin_move = mcmc_setup_moves(config, 'mcmc_moves_burnin')

            burnin_sampler = emcee.EnsembleSampler(
                config['mcmc_nwalkers_burnin'],
                config['mcmc_ndims'],
                ln_prob_global,
                moves=burnin_move,
                pool=pool,
                args = [config, cosmos_data, sim_data])

            # Burn-in
            mcmc_burnin_pos, mcmc_burnin_lnp, mcmc_burnin_state = mcmc_burnin(
                burnin_sampler, mcmc_ini_position, config, verbose=True)

            # Estimate the Kernel density distributions of final brun-in positions
            # Resample the distributions to get starting positions of the actual run
            mcmc_kde = gaussian_kde(np.transpose(mcmc_burnin_pos),
                               bw_method='silverman')
            mcmc_new_pos = np.transpose(mcmc_kde.resample(config['mcmc_nwalkers']))

            mcmc_new_ini = (mcmc_new_pos, mcmc_burnin_lnp, mcmc_burnin_state)

            # TODO: Convergence test
            burnin_sampler.reset()

            # Change the moves
            # Decide the Ensemble moves for walkers during the official run
            mcmc_move = mcmc_setup_moves(config, 'mcmc_moves')

            mcmc_sampler = emcee.EnsembleSampler(
                config['mcmc_nwalkers'],
                config['mcmc_ndims'],
                ln_prob_global,
                moves=mcmc_move,
                pool=pool,
                args = [config, cosmos_data, sim_data])

            # MCMC run
            mcmc_run_result = emcee_run(
                mcmc_sampler, mcmc_new_ini, config, verbose=True)
    else:

        # define the ensemble moves object for walkers during burn in
        burnin_move = mcmc_setup_moves(config, 'mcmc_moves_burnin')
        # define sampler
        burnin_sampler = emcee.EnsembleSampler(config['mcmc_nwalkers_burnin'],
                                               config['mcmc_ndims'],
                                                ln_prob_global,
                                                moves=burnin_move,
                                                args = [config, cosmos_data, sim_data])

        # Burn-in
        mcmc_burnin_pos, mcmc_burnin_lnp, mcmc_burnin_state = mcmc_burnin(
            burnin_sampler, mcmc_ini_position, config, verbose=True)

        # Estimate the Kernel density distributions of final burn-in positions
        # Resample the distributions to get starting positions of the actual run
        mcmc_kde = gaussian_kde(np.transpose(mcmc_burnin_pos),
                            bw_method='silverman')
        mcmc_new_pos = np.transpose(mcmc_kde.resample(config['mcmc_nwalkers']))

        mcmc_new_ini = (mcmc_new_pos, mcmc_burnin_lnp, mcmc_burnin_state)


        # MCMC run

        # Decide the Ensemble moves for walkers during burnin
        mcmc_move = mcmc_setup_moves(config, 'mcmc_moves')
        # define sampler
        mcmc_sampler = emcee.EnsembleSampler(config['mcmc_nwalkers'],
                                             config['mcmc_ndims'],
                                             ln_prob_global,
                                             moves=mcmc_move,
                                             args = [config, cosmos_data, sim_data])

        mcmc_run_result = emcee_run(mcmc_sampler, mcmc_new_ini, config, verbose=True)

    return mcmc_run_result

################################################################################
# Helper functions
################################################################################
## matching dwarf catalog with mock functions
def find_nearest_new_indices_sorted(index, n, existing_indices, length):
    """
    index: index of closest value
    n: number of closest
    existing_indices: indices that already exist to avoid repeats

    return:
    indices of n nearest rows
    """

    # indices of nearest rows
    nearest_rows=[]

    # append index
    i=0
    if index not in existing_indices and index <length:
            nearest_rows.append(index)

    #append next nearest indices
    while len(nearest_rows) < n:

        if index + i not in existing_indices and index + i not in nearest_rows and index + i<length:
            nearest_rows.append(index + i)

        if len(nearest_rows) < n and index - i not in existing_indices and index - i not in nearest_rows and index - i>0 and index - i<length:
            nearest_rows.append(index - i)

        i += 1

    return nearest_rows

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

    # indices of nearest mock mass for each dwarf mass
    indices = np.searchsorted(mock_galaxies['stellar_mass'], 10**dwarf_masses)

    #add additional nearest
    for index in indices:

        matched_indices = find_nearest_new_indices_sorted(index, n_nearest, subsample_indices, len(mock_galaxies))

        subsample_indices.update(matched_indices)

    subsample = mock_galaxies[list(subsample_indices)]

    return subsample

def GM_data_location(config):
    '''
    Defines data location in GRAYMALKIN
    '''
    config['data_location'] = '/data/ateam/fardila/dwarf_lensing/'

    return config

################################################################################
