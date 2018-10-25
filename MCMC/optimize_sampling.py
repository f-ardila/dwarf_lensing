import numpy as np
import pandas as pd
import random
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
mock_galaxies = mock_galaxies['galid', 'x', 'y', 'z', 'stellar_mass']
mock_galaxies = mock_galaxies[(np.log10(mock_galaxies['stellar_mass'])>=min(dwarf_masses)) & \
                              (np.log10(mock_galaxies['stellar_mass'])<9.0)]
#reduce size of table in half for memory purposes
mock_galaxies = mock_galaxies[random.sample(np.arange(len(mock_galaxies)),500000)]


################################################################################
def find_nearest_rows_pandas(data_frame, value, n):

    """
    data_frame: pandas DF from which we will find closest
    value: closest to this value
    n: number of closest

    return:
    all columns of data_frame indexed to rows with closest values
    """

    nearest_rows = np.abs(np.log10(data_frame['stellar_mass']) - value).sort_values()[:n]

    return data_frame.loc[nearest_rows.index]

def create_dwarf_catalog_with_matched_mass_distribution(dwarf_masses, mock_galaxies, n_nearest = 1):
    """
    dwarf_masses: masses of COSMOS dwarfs
    mock_galaxies: array of mock galaxies from halotools
    n_nearest: number of nearest mock galaxies to use (i.e. number of times to
               sample full COSMOS dwarf catalog)

    return:
    pandas DF containing the sample of mock dwarfs with an identical distribution
    of masses as COSMOS
    """
    subsample=[]

    #convert to pandas DF. byte order error occurs if not converted in this way
    mock_galaxies_pd = pd.DataFrame(mock_galaxies.as_array())

    for dwarf in dwarf_masses:

        #reduce search space
        gal_masses=np.log10(mock_galaxies_pd['stellar_mass'])
        mock_galaxies_reduced = mock_galaxies_pd[(gal_masses<dwarf+0.001) & \
                                                (gal_masses>dwarf-0.001)]


        #find mock galaxies with mass closest to dwarf
        matched_galaxies = find_nearest_rows_pandas(mock_galaxies_reduced,dwarf,
                                                    n=n_nearest)

        # append to subsample
        subsample += [matched_galaxies]

        # sample without replacement in array of mock galaxy masses
        # faster to set to 0 than to delete or mask
        mock_galaxies_pd['stellar_mass'].loc[matched_galaxies.index] = 0

    return pd.concat(subsample)

################################################################################
# run functions on catalogs

subsample= create_dwarf_catalog_with_matched_mass_distribution(dwarf_masses,
                                                                mock_galaxies,
                                                                n_nearest = 50)
print 'subsample: ' + str(len(subsample))

#check that they are indistinguishable
print ks_2samp(np.log10(subsample['stellar_mass']),dwarf_masses)
