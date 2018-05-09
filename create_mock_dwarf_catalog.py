import numpy as np
from astropy.io import ascii, fits
from scipy.stats import anderson_ksamp, ks_2samp
from halotools.sim_manager import HaloTableCache, CachedHaloCatalog
from halotools.empirical_models import PrebuiltSubhaloModelFactory
import time

init_time=time.time()

'''
create mock catalog of dwarfs with the same distribution of masses as the COSMOS
sample
'''

def find_nearest_index(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    
    return idx



#Bol-Planck model
simname = 'bolshoi-planck'
halo_finder = 'rockstar'
version_name = 'bplanck_dwarfs'
ptcl_version_name='bplanck_dwarfs_downsampled2'
ptcl_version_name='bplanck_dwarfs'

redshift = 0.278625 #(1/0.78209)-1 ; a=0.78209
Lbox, particle_mass = 250, 1.5e8

#read in halocat
cache = HaloTableCache()
halocat = CachedHaloCatalog(simname = simname, halo_finder = halo_finder,
                            version_name = version_name, redshift = redshift, ptcl_version_name=ptcl_version_name) # doctest: +SKIP

print(halocat.redshift) # doctest: +SKIP
print(halocat.ptcl_version_name) # doctest: +SKIP


model = PrebuiltSubhaloModelFactory('behroozi10', redshift=redshift,
                                    scatter_abscissa=[12, 15], scatter_ordinates=[0.4, 0.2])
model.populate_mock(halocat)


#COSMOS data
dwarf_sample_file = '/Users/fardila/Documents/GitHub/dwarf_lensing/Data/cosmos/dwarf_sample_for_paper.fits'

dwarf_sample_data  = fits.open(dwarf_sample_file)[1].data
dwarf_masses = dwarf_sample_data['mass_med']

#galaxy table limited to mass range of dwarfs
mock_galaxies = model.mock.galaxy_table
mock_galaxies = mock_galaxies['galid', 'x', 'y', 'z', 'stellar_mass']
mock_galaxies = mock_galaxies[(np.log10(mock_galaxies['stellar_mass'])>=min(dwarf_masses)) & (np.log10(mock_galaxies['stellar_mass'])<9.0)]

model = 0 
# create subsample with same distribution
subsample=[]
copy_mock_galaxies = [list(a) for a in mock_galaxies]
gal_masses=[g[-1] for g in copy_mock_galaxies]

for i in range(50):
    print i

    for dwarf in dwarf_masses:

        #find index of mock galaxy with mass closes to dwarf
        index = find_nearest_index(np.log10(gal_masses), dwarf)

	# append to subsample
	subsample.append(copy_mock_galaxies[index])

    	#do not replace in array of mock galaxy masses 
    	del copy_mock_galaxies[index]
    	del gal_masses[index]

subsample_masses= [np.log10(s[-1]) for s in subsample]

print 'subsample: ' + str(len(subsample))
print 'total mock galaxies: ' + str(len(mock_galaxies))
print 'remaining mock galaxies: ' + str(len(copy_mock_galaxies))

#check that they are indistinguishable
#anderson_ksamp([subsample,dwarf_masses])
print ks_2samp(subsample_masses,dwarf_masses)

#save subsample
outfile='/Users/fardila/Documents/GitHub/dwarf_lensing/bplanck_dwarfs.npy'
np.save(outfile,subsample)

print time.time() - init_time, ' seconds'
