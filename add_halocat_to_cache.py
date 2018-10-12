from halotools.sim_manager import TabularAsciiReader
from halotools.sim_manager.rockstar_hlist_reader import RockstarHlistReader
#python 3 gives error about converting unicode 'TypeError: No conversion path for dtype: dtype('<U12')'


#halotools catalog

halo_cat_file='/Users/fardila/Documents/GitHub/dwarf_lensing/Data/bplanck/hlist_0.78209.list'
#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) desc_pid(7) phantom(8) sam_mvir(9) mvir(10)
#rvir(11) rs(12) vrms(13) mmp?(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) vz(22) Jx(23) Jy(24)
#Jz(25) Spin(26) Breadth_first_ID(27) Depth_first_ID(28) Tree_root_ID(29) Orig_halo_ID(30) Snap_num(31)
#Next_coprogenitor_depthfirst_ID(32) Last_progenitor_depthfirst_ID(33) Last_mainleaf_depthfirst_ID(34) Tidal_Force(35)
#Tidal_ID(36) Rs_Klypin(37) Mvir_all(38) M200b(39) M200c(40) M500c(41) M2500c(42) Xoff(43) Voff(44) Spin_Bullock(45)
#b_to_a(46) c_to_a(47) A[x](48) A[y](49) A[z](50) b_to_a(500c)(51) c_to_a(500c)(52) A[x](500c)(53) A[y](500c)(54)
#A[z](500c)(55) T/|U|(56) M_pe_Behroozi(57) M_pe_Diemer(58) Macc(59) Mpeak(60) Vacc(61) Vpeak(62) Halfmass_Scale(63)
#Acc_Rate_Inst(64) Acc_Rate_100Myr(65) Acc_Rate_1*Tdyn(66) Acc_Rate_2*Tdyn(67) Acc_Rate_Mpeak(68) Mpeak_Scale(69)
#Acc_Scale(70) First_Acc_Scale(71) First_Acc_Mvir(72) First_Acc_Vmax(73) Vmax@Mpeak(74) Tidal_Force_Tdyn(75)
#Log_(Vmax/Vmax_Tdyn)(76)


output_fname = '/Users/fardila/Documents/GitHub/dwarf_lensing/Data/bplanck/hlist_0.78209.hdf5'
columns_to_keep_dict = {'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'), 'halo_mvir': (10, 'f4'), 'halo_x': (17, 'f4'),
                        'halo_y': (18, 'f4'), 'halo_z': (19, 'f4'),'halo_vx': (20, 'f4'), 'halo_vy': (21, 'f4'),
                        'halo_vz': (22, 'f4'), 'halo_rvir': (11, 'f4'), 'halo_mpeak': (60, 'f4')}
#is there a way to read in all columns? NO

simname = 'bolshoi-planck'
halo_finder = 'rockstar'
version_name = 'bplanck_dwarfs'
ptcl_version_name='bplanck_dwarfs_downsampled2'
ptcl_version_name='bplanck_dwarfs'

z_sim = 0.278625 #(1/0.78209)-1 ; a=0.78209
Lbox, particle_mass = 250, 1.5e8

reader = RockstarHlistReader(halo_cat_file, columns_to_keep_dict, output_fname, simname, halo_finder, z_sim,
                             version_name, Lbox, particle_mass, overwrite=True) # doctest: +SKIP
reader.read_halocat(['halo_rvir'], write_to_disk = True, update_cache_log = True) # doctest: +SKIP
