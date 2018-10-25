import numpy as np
from astropy.io import fits
from scipy.stats import ks_2samp
from halotools.sim_manager import HaloTableCache, CachedHaloCatalog
from halotools.empirical_models import PrebuiltSubhaloModelFactory

"""
script to reproduce the distribution of masses in COSMOS dwarf catalog with mock
catalog from bplanck. purpose is to use these functions at each step of MCMC to
determine ideal SHMR parameters for dwarf galaxies.
"""

################################################################################
#COSMOS data
dwarf_sample_file = '/Users/fardila/Documents/GitHub/dwarf_lensing/Data/cosmos/dwarf_sample_for_paper.fits'

dwarf_sample_data  = fits.open(dwarf_sample_file)[1].data
dwarf_masses = dwarf_sample_data['mass_med']

################################################################################
#Bol-Planck model
simname = 'bolshoi-planck'
halo_finder = 'rockstar'
version_name = 'bplanck_dwarfs'
ptcl_version_name='bplanck_dwarfs'
redshift = 0.278625 #(1/0.78209)-1 ; a=0.78209
Lbox, particle_mass = 250, 1.5e8

#load halocat object and populate using subhalo model
cache = HaloTableCache()
halocat = CachedHaloCatalog(simname = simname, halo_finder = halo_finder,
                            version_name = version_name, redshift = redshift,
                             ptcl_version_name=ptcl_version_name) # doctest: +SKIP


model = PrebuiltSubhaloModelFactory('behroozi10', redshift=redshift,
                                    scatter_abscissa=[12, 15],
                                     scatter_ordinates=[0.4, 0.2])
model.populate_mock(halocat)


#create galaxy table limited to mass range of dwarfs
mock_galaxies = model.mock.galaxy_table
mock_galaxies = mock_galaxies['x', 'y', 'z', 'stellar_mass']
mock_galaxies = np.array(mock_galaxies[(np.log10(mock_galaxies['stellar_mass'])>=min(dwarf_masses)) & \
                              (np.log10(mock_galaxies['stellar_mass'])<9.0)])
#reduce size of table in half
half_mock_galaxies = np.random.choice(mock_galaxies,500000)
################################################################################
def find_nearest_new_indices_sorted(index, n, existing_indices, length):
    """
    index: index of closest value
    n: number of closest
    existing_indices: indices that already exist to avoid repeats

    return:
    indices of n nearest rows
    """

    #indices of nearest rows
    nearest_rows=[]

    #append index
    i=0
    if index not in existing_indices:
            nearest_rows.append(index)

    #append next nearest indices
    while len(nearest_rows) < n:

        if index + i not in existing_indices and index + i not in nearest_rows and index + i<length:
            nearest_rows.append(index + i)

        if len(nearest_rows) < n and index - i not in existing_indices and index - i not in nearest_rows and index - i>0:
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

    #set for speeding up finding location
    subsample_indices=set()

    #sort first to speed up future calculations
    mock_galaxies.sort(order = 'stellar_mass')

    #indices of nearest mock mass for each dwarf mass
    indices = np.searchsorted(mock_galaxies['stellar_mass'], 10**dwarf_masses)

    #add additional nearest
    for index in indices:

        matched_indices = find_nearest_new_indices_sorted(index, n_loops, subsample_indices, len(mock_galaxies))

        subsample_indices.update(matched_indices)

    subsample = mock_galaxies[list(subsample_indices)]

    return subsample
################################################################################
# run functions on catalogs

subsample= create_dwarf_catalog_with_matched_mass_distribution(dwarf_masses,
                                                                mock_galaxies,
                                                                n_nearest = 50)
print 'subsample: ' + str(len(subsample))

#check that they are indistinguishable
print ks_2samp(np.log10(subsample['stellar_mass']),dwarf_masses)
