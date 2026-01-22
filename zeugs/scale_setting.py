import numpy as np
import matplotlib.pyplot as plt




def summer_beta(beta : float) -> float:
    '''Calculates r0/a as a function of beta through sommer scale scale setting.
    Fitted for  5.7 <= beta <= 6.8.
    Parametrisation from [A. Francis, O. Kaczmarek, M. Laine, T. Neuhaus, and H. Ohno, Phys. Rev. D 91, 096002 (2015)]
    Updated coefficients from [Y. Burnier, H. T. Ding, O. Kaczmarek, A. L. Kruse, M. Laine, H. Ohno, and H. Sandmeyer, JHEP 11, 206 (2017)]'''

    b0 = 11 / (4*np.pi)**2
    b1 = 102 / (4*np.pi)**4

    c1 = -8.9664
    c2 = 19.21
    c3 = -5.25217
    c4 = 0.606828

    ln_r0_a = (beta/(12*b0) + b1/(2*b0**2) * np.log(6*b0/beta)) * (1 + c1/beta + c2/beta**2) / (1 + c3/beta + c4/beta**2)

    return np.exp(ln_r0_a)




def find_beta(Nt : int, T_Tc : float, b_min : float = 4, b_max : float = 10) -> float:
    '''Given ...
    r_0 * T_c from [A. Francis, O. Kaczmarek, M. Laine, T. Neuhaus, and H. Ohno, Phys. Rev. D 91, 096002 (2015)]'''

    r0_Tc = 0.7457

    r0_a = r0_Tc * Nt * T_Tc

    return min(
        np.linspace(b_min, b_max, 100_000),
        key = lambda b : abs(r0_a - summer_beta(b))
    )




def temp_rel_crit(beta : float, Nt : int) -> float:
    '''Temperatur in units of critical temperature from beta and Nt.'''

    r0_Tc = 0.7457
    return summer_beta(beta) / Nt / r0_Tc








if __name__ == '__main__':

    Nt = 16
    T_Tc = 0.99

    beta = find_beta(Nt, T_Tc)
    print(beta)

    print(temp_rel_crit(beta, Nt))

    print()



    Nt = 16
    beta = 6.542
    print(temp_rel_crit(beta, Nt))
