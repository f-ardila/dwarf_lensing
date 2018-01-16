from halotools.sim_manager.user_supplied_ptcl_catalog import  UserSuppliedPtclCatalog
import numpy as np

#particle_catalog_file='Data/bplanck_particles_100m_a0.78209_downsampled2'
particle_catalog_file='Data/bplanck_particles_100m_a0.78209'

f=open(particle_catalog_file,"r")
lines=f.readlines()
x=[]
y=[]
z=[]
vx=[]
vy=[]
vz=[]
id=[]

for l in lines:
    l=l.split(' ')
    
    x.append(float(l[0]))
    y.append(float(l[1]))
    z.append(float(l[2]))
    vx.append(float(l[3]))
    vy.append(float(l[4]))
    vz.append(float(l[5]))
    id.append(int(l[6]))
    
f.close()
print('file closed')

simname = 'bolshoi-planck'
halo_finder = 'rockstar'
#version_name = 'bplanck_dwarfs_downsampled2'
version_name = 'bplanck_dwarfs'
redshift = 0.278625 #a=0.78209; z=(1/0.78209)-1
Lbox, particle_mass = 250, 1.5e8

#make numpy arrays
x=np.array(x)
print('read x')
y=np.array(y)
print('read y')
z=np.array(z)
print('read z')
vx=np.array(vx)
print('read vx')
vy=np.array(vy)
print('read vy')
vz=np.array(vz)
print('read vz')
id=np.array(id)
print('read id')

ptcl_catalog = UserSuppliedPtclCatalog(redshift=redshift, Lbox=Lbox, 
                                       particle_mass=particle_mass, x=x, y=y, 
                                       z=z, vx=vx, vy=vy, vz=vz, ptcl_ids=id)
                                       
print('read in catalog')
                        
#fname = 'Data/bplanck_particles_100m_a0.78209_downsampled2.hdf5'
#processing_notes = 'A random sample of 1e6 lines from bplanck_particles_100m_a0.78209 were used, out of 99993334.'
#processing_notes = 'Only the first 1e6 lines from bplanck_particles_100m_a0.78209 were used, out of 99993334.'
fname = 'Data/bplanck_particles_100m_a0.78209.hdf5'
processing_notes = 'All 99993334 lines from bplanck_particles_100m_a0.78209 file.'

x=0
y=0
z=0
vx=0
vy=0
vz=0
id=0

#add to cache
ptcl_catalog.add_ptclcat_to_cache(fname, simname, version_name, processing_notes, overwrite=True) # doctest: +SKIP

#Your particle catalog has now been cached and is accessible whenever 
#you load the associated halo catalog into memory. For example:
#from halotools.sim_manager import CachedHaloCatalog
#halocat = CachedHaloCatalog(simname=simname, halo_finder='some halo-finder', 
#                            version_name='some version-name', redshift=redshift, 
#                            ptcl_version_name=version_name) # doctest: +SKIP
#                            
#                            

