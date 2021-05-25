This is data used in dwarf lensing project (https://github.com/f-ardila/dwarf_lensing)


############################
- bplanck (Bolshoi-Planck); z = 0.278
	- hlist_0.78209.list : halo catalog
	- bplanck_particles_100m_a0.78209.hdf5 : particles catalog

############################
- cosmos: 
	- smf: Cosmos SMF at z~0.2 from Iary Davidzon
		- cosmos2015_dic2017_smf_z01-04_Vmax0.dat : SMF measurements
		- cosmos2015_dic2017_smf_z01-04_STY0.dat : best fit to SMF
	- lensing: lensing from Laigle2015
		- dwarf_sample_for_paper.fits : sample used to calculate lensing
		- dwarf_all_lens_laigle2015_nozp2.out : lensing signal from Alexie

############################
- MCMC_samples: numpy arrays with MCMC samples from various runs. Samples SHMR model parameters. Model run given at the end of each file name.

############################
- mock_dwarfs: mock catalogs of dwarf galaxies.
	- best_fit_mock_catalogs: Generated using the best fit SHMR parameters. Shown in plot_sigma_and_make_catalogs.ipynb.
		- "short" columns : ['halo_mvir','halo_mpeak','halo_Vmax@Mpeak','M_*']
		- full columns: ['M_*','halo_mpeak','halo_mvir','halo_Vmax@Mpeak', 'halo_pid']
	- bplanck_resamples: generated from resampling the BPlanck galaxy catalog X number of times while still matching the mass distribution of the COSMOS dwarf sample. X is given in each file name. Shown in bplanck_dwarfs_matched_distribution.ipynb .

############################
- UM (UniverseMachine)
	- UM_hlist_0.78209.hdf5 : halotools catalog
	- UM_sfr_catalog_0.782092_proper_bounds.txt: halo catalog from UM with all positions converted to be within simulation volume.
	- UM_sfr_catalog_0.782092.txt: halo catalog from UM (some positions are outside volume)
