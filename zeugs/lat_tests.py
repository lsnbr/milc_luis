from itertools import product
from collections import defaultdict



def s2_max(ns):
    return 3 * (ns/2)**2



def f(ns):
    n_d = defaultdict(int)

    for i,j,k in product(range(ns), repeat=3):
        i,j,k = (min(x, ns-x) for x in (i,j,k))
        d = i*i + j*j + k*k
        n_d[d] += 1

    return n_d



if __name__ == '__main__':

    ns = 4
    print(f'box size {ns}x{ns}x{ns}')

    print('s2_max =', s2_max(ns))

    n_d = f(ns)
    print('\n'.join(f'{d} {n}' for d, n in sorted(n_d.items(), key=lambda x:x[0])))

    print('total sum =', sum(n_d.values()))