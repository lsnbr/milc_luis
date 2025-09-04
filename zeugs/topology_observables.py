from pathlib import Path
import re
import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt


Array4D = np.ndarray




def a_in_fm(beta : float, r0 : float = 0.5) -> float:
    '''Calculate lattice spacing scale using gattringer p.67 formula. (5.7 < beta < 6.92)'''

    return r0 * np.exp(-1.6804 - 1.7331*(beta - 6) + 0.7849*(beta - 6)**2 - 0.4428*(beta - 6)**3)



def read_topo(topo_file : Path) -> Array4D:
    '''Read topology density file into 4d numpy array.'''

    with open(topo_file) as f:
        _, nxl,nyl,nzl,ntl, *tcd_lines = f.readlines()
        nx,ny,nz,nt = (int(l.split()[1]) for l in (nxl,nyl,nzl,ntl))

        tcd_lat = np.zeros(shape=(nx,ny,nz,nt), dtype=float)
        for tcd_line in tcd_lines:
            x,y,z,t, tcd = tcd_line.split()
            tcd_lat[int(x),int(y),int(z),int(t)] = float(tcd)

    return tcd_lat




def topological_charge(tcd_lat : Array4D) -> float:
    '''Calculates total topological charge of the configuration.'''
    
    return tcd_lat.sum()



def correlators_fourier(tcd_lat : Array4D) -> Array4D:
    '''Calculates correlators using O(nlogn) fourier method.'''

    tcd_ft_lat = fftn(tcd_lat)
    correlators = ifftn(tcd_ft_lat.real * tcd_ft_lat.real + tcd_ft_lat.imag * tcd_ft_lat.imag)
    return correlators



def correlators_direct(tcd_lat : Array4D) -> Array4D:
    '''Calculates correlators using O(n^2) direct computation.'''
    
    nx,ny,nz,nt = tcd_lat.shape
    volume = nx*ny*nz*nt
    correlators = np.empty_like(tcd_lat, dtype=float)

    for x,y,z,t in np.ndindex(tcd_lat.shape):
        correlators[x,y,z,t] = sum( tcd_lat[x0+x,y0+y,z0+z,t0+t] * tcd_lat[x0,y0,z0,t0] 
                                    for x0,y0,z0,t0 in np.ndindex(tcd_lat.shape)        ) / volume
        
    return correlators



def top_sus_corr(correlators : Array4D) -> float:
    '''Calculates topological susceptibility as intgral over correators.'''

    return correlators.sum()



def top_sus_total(top_charge : float, volume : float) -> float:
    '''Calculates topological susceptibility as Q*Q/V.'''

    return top_charge**2 / volume



def get_ensemble_tcd(folder : Path, flow_time : float) -> list[Path]:
    '''Get all files (non-recursive) that end with ".tcd" and have specified flow_time.'''

    files = []

    for f in folder.iterdir():
        if not (f.is_file() and f.name.endswith('.tcd')): continue

        ft = float(re.search(r't=(\d+(\.\d+)?)', f.name)[1])
        if abs(ft - flow_time) < 1e-6:
            files.append(f)

    return sorted(files)



def top_charge_trajectory(ensemble : list[Path]) -> list[float]:
    '''List of topological charges of ensemble (sorted).'''

    return [topological_charge(read_topo(f)) for f in ensemble]








if __name__ == '__main__':

    ensemble = get_ensemble_tcd(Path('./gauge_configs'), 0.72)
    top_ch_traj = top_charge_trajectory(ensemble)

    plt.plot(top_ch_traj, marker='o', markersize=2, linestyle='none')

    plt.ylim(-2.2, 2.2)
    
    for ly in (-2,-1,1,2):
        plt.axhline(ly, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.xlabel('config')
    plt.ylabel('Q')

    #plt.legend()

    plt.savefig("top_traj.png", dpi=150, bbox_inches="tight")



