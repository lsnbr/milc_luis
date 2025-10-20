#include "wilson_flow_includes.h"




void tcd_corrs_by_fourier() {

    int i;
    site *s;
    

    // Copy (double site->ch_dens) into (complexdouble site->ch_dens_corr)
    FORALLSITES(i,s) {
        s->ch_dens_corr.real = s->ch_dens;
        s->ch_dens_corr.imag = 0;
    }
    g_sync();


    // Do fourier transform on site->ch_dens_corr
    int key[4];
    int slice[4];
    key[XUP] = 1;
    key[YUP] = 1;
    key[ZUP] = 1;
    key[TUP] = 1;
    setup_restrict_fourier(key, slice);
    restrict_fourier_site(F_OFFSET(ch_dens_corr), sizeof(dcomplex), FORWARDS);


    // Compute |q(k)|^2 for each site->ch_dens_corr
    FORALLSITES(i,s) {
        s->ch_dens_corr.real = s->ch_dens_corr.real * s->ch_dens_corr.real + s->ch_dens_corr.imag * s->ch_dens_corr.imag;
        s->ch_dens_corr.imag = 0;
    }
    g_sync();


    // Inverse fourier transform site->ch_dens_corr, thus computing the correlators
    setup_restrict_fourier(key, slice);
    restrict_fourier_site(F_OFFSET(ch_dens_corr), sizeof(dcomplex), BACKWARDS);

}




/* - corrs_dist[nt/2][s^2_max+1] 
   - s^2_max is the maximal squared distance, s^2_max = 3 * (ns/2)^2
   - correlators are not normalized by d_s
*/
void corr_by_spatial_distance(double **corrs_dist) {
    
    int i;
    site *s;

    int s2_max = (nx/2)*(nx/2) + (ny/2)*(ny/2) + (nz/2)*(nz/2);
    int count = (nt/2) * (s2_max + 1);


    for (int ii = 0; ii < nt/2; ii++) {
        for (int jj = 0; jj <= s2_max; jj++) {
            corrs_dist[ii][jj] = 0.0;
        }
    }

    FORALLSITES(i, s) {
        if (s->t >= nt/2) continue;

        int sx = s->x > nx/2 ? nx - s->x : s->x;
        int sy = s->y > ny/2 ? ny - s->y : s->y;
        int sz = s->z > nz/2 ? nz - s->z : s->z;

        int s2 = sx*sx + sy*sy + sz*sz;
        if (s2 > s2_max) {printf("bad distance, s2=%d, s2max=%d\n", s2, s2_max); break;}    /* should never happen */

        corrs_dist[s->t][s2] += s->ch_dens_corr.real;
    }

    g_vecdoublesum(corrs_dist[0], count);


}






void somecorrtests() {

    ;

}



