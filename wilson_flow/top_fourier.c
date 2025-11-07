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
    // Divide by volume**2 to obtain correctly normalized correlators (could also be done after the backwards fft)
    // Multiply by nt^8 to convert from a^2 units to 1/T units
    FORALLSITES(i,s) {
        s->ch_dens_corr.real = (s->ch_dens_corr.real * s->ch_dens_corr.real + s->ch_dens_corr.imag * s->ch_dens_corr.imag)
                               / (pow(volume, 2) / pow(nt, 8));
        s->ch_dens_corr.imag = 0;
    }
    g_sync();


    // Inverse fourier transform site->ch_dens_corr, thus computing the correlators
    setup_restrict_fourier(key, slice);
    restrict_fourier_site(F_OFFSET(ch_dens_corr), sizeof(dcomplex), BACKWARDS);

}




/* - corrs_dist[nt/2+1][s^2_max+1] 
   - s^2_max is the maximal squared distance, s^2_max = 3 * (ns/2)^2
   - correlators are not normalized by d_s
*/
void corr_by_spatial_distance(double **corrs_dist) {
    
    int i;
    site *s;

    int s2_max = (nx/2)*(nx/2) + (ny/2)*(ny/2) + (nz/2)*(nz/2);
    int count = (nt/2 + 1) * (s2_max + 1);


    for (int ii = 0; ii <= nt/2; ii++) {
        for (int jj = 0; jj <= s2_max; jj++) {
            corrs_dist[ii][jj] = 0.0;
        }
    }

    FORALLSITES(i, s) {
        int st = s->t > nt/2 ? nt - s->t : s->t;
        int sx = s->x > nx/2 ? nx - s->x : s->x;
        int sy = s->y > ny/2 ? ny - s->y : s->y;
        int sz = s->z > nz/2 ? nz - s->z : s->z;

        int s2 = sx*sx + sy*sy + sz*sz;
        if (s2 > s2_max) {printf("bad distance, s2=%d, s2max=%d\n", s2, s2_max); break;}    /* should never happen */

        int t_degen = (st==0 || st==nt/2) ? 1 : 2;      // how many t-separations equal st?
        corrs_dist[st][s2] += s->ch_dens_corr.real / t_degen / (nt*nt);     // /(nt*nt) since C(t, r) has dim=L^(-6) instead of dim=L^(-8)
    }

    g_sync();
    g_vecdoublesum(corrs_dist[0], count);


}








void somecorrtests(double **corrs_dist) {

    int i;
    site *s;

    // build d_r array (for each radial distance r^2, the number of sites with that distance)
    int s2_max = (nx/2)*(nx/2) + (ny/2)*(ny/2) + (nz/2)*(nz/2);
    int *dr_array = calloc(s2_max+1, sizeof(int));
    for(int x=0; x<nx; x++) for(int y=0; y<ny; y++) for(int z=0; z<nz; z++) {
        int sx = x > nx/2 ? nx - x : x;
        int sy = y > ny/2 ? ny - y : y;
        int sz = z > nz/2 ? nz - z : z;
        dr_array[sx*sx + sy*sy + sz*sz] += 1;
    }



    /* ############################################### */
    /* ####### Test 0: fourier normalization ######### */
    /* ############################################### */
    g_sync();
    node0_printf("Test 0:\n");
    g_sync();

    FORALLSITES(i,s) {
        s->ch_dens_corr.real = 1;
        s->ch_dens_corr.imag = 0;
    }
    g_sync();

    int key[4];
    int slice[4];
    key[XUP] = 1;
    key[YUP] = 1;
    key[ZUP] = 1;
    key[TUP] = 1;
    setup_restrict_fourier(key, slice);
    restrict_fourier_site(F_OFFSET(ch_dens_corr), sizeof(dcomplex), FORWARDS);

    g_sync();
    setup_restrict_fourier(key, slice);
    restrict_fourier_site(F_OFFSET(ch_dens_corr), sizeof(dcomplex), BACKWARDS);

    // test for volume factor (fourier forward then back gives original function times volume)
    g_sync();
    FORALLSITES(i,s) {
        if (fabs(s->ch_dens_corr.imag) > 1e-8) {
            printf("expected Im=0, got Im=%.16g\n", s->ch_dens_corr.imag);
            fflush(stdout);
            break;
        }
        if (fabs(s->ch_dens_corr.real - volume) > 1e-8) {
            printf("expected Re=V=%d, got Re=%.16g\n", (int)volume, s->ch_dens_corr.real);
            fflush(stdout);
            break;
        }
    }



    /* ############################################### */
    /* ############ Test 1: single site ############## */
    /* ############################################### */
    g_sync();
    node0_printf("Test 1:\n");

    FORALLSITES(i,s) {
        s->ch_dens = pow(0.001234, 0.5);
    }
    tcd_corrs_by_fourier();
    corr_by_spatial_distance(corrs_dist);

    // test if real
    g_sync();
    node0_printf("Test if real:\n");
    g_sync();
    FORALLSITES(i,s) {
        if (fabs(s->ch_dens_corr.imag) > 1e-8) {
            printf("imag dens-corr at (x,y,z,t)=(%d,%d,%d,%d) with Im(C)=%.16g\n", s->x, s->y, s->z, s->t, s->ch_dens_corr.imag);
            fflush(stdout);
            break;
        }
    }

    // test correlation values
    g_sync();
    node0_printf("Test C(x):\n");
    g_sync();
    FORALLSITES(i,s) {
        double expected = 0.001234 * pow(nt, 8);
        double actual = s->ch_dens_corr.real;
        if (fabs(actual - expected) > 1e-8) {
            printf("site (x=%d,y=%d,z=%d,t=%d): expected: %.16g, actual: %.16g\n", s->x, s->y, s->z, s->t, expected, actual);
            fflush(stdout);
            break;
        }
    }

    // test C(t,r)
    g_sync();
    node0_printf("Test C(t,r):\n");
    g_sync();
    for (int t=0; t<=nt/2; t++) {
        int passed = 1;
        for (int s2=0; s2<=s2_max; s2++) {
            double expected = 0.001234 * dr_array[s2] * pow(nt, 6);
            double actual = corrs_dist[t][s2];
            if (fabs(actual - expected) > 1e-6) {
                node0_printf("t=%d, r^2=%d: expected: %.16g, actual: %.16g\n", t, s2, expected, actual);
                fflush(stdout);
                passed = 0;
                break;
            }
        }
        if (passed == 0) break;
    }



    /* ############################################### */
    /* ############ Test 2: single site ############## */
    /* ############################################### */
    g_sync();
    node0_printf("Test 2:\n");

    int t0 = nt/4;
    int x0 = nx/4;
    int y0 = ny/2;
    int z0 = nz/3;

    FORALLSITES(i,s) {
        if (s->x==x0 && s->y==y0 && s->z==z0 && s->t==t0) s->ch_dens = 1.0540398474441086;    // sqrt of 1.111
        else                                              s->ch_dens = 0;
    }
    tcd_corrs_by_fourier();
    corr_by_spatial_distance(corrs_dist);

    // test correlators
    g_sync();
    node0_printf("Test C(x):\n");
    g_sync();
    FORALLSITES(i,s) {
        double expected = (s->x==0 && s->y==0 && s->z==0 && s->t==0) ? 1.111 / volume * pow(nt, 8) : 0;
        double actual = s->ch_dens_corr.real;
        if (fabs(actual - expected) > 1e-8) {
            printf("site (x=%d,y=%d,z=%d,t=%d): expected: %.16g, actual: %.16g\n", s->x, s->y, s->z, s->t, expected, actual);
            fflush(stdout);
            break;
        }
    }

    // test C(t,r)
    g_sync();
    node0_printf("Test C(t,r):\n");
    g_sync();
    for(int t=0; t<=nt/2; t++) {
        int passed = 1;
        for(int s2=0; s2<=s2_max; s2++) {
            double expected = (t==0 && s2==0) ? 1.111 / volume * pow(nt, 6) : 0;
            double actual = corrs_dist[t][s2];
            if (fabs(actual - expected) > 1e-8) {
                node0_printf("t=%d, r^2=%d: expected: %.16g, actual: %.16g\n", t, s2, expected, actual);
                fflush(stdout);
                passed = 0;
                break;
            }
        }
        if (passed == 0) break;
    }



    /* ############################################### */
    /* ############## Test 3: two site ############### */
    /* ############################################### */
    g_sync();
    node0_printf("Test 3:\n");
    
    double qa = 1.0540398474441086;    // sqrt of 1.111
    double qb = 0.3507135583350036;    // sqrt of 0.123

    FORALLSITES(i,s) {
        if      (s->x==x0   && s->y==y0 && s->z==z0 && s->t==t0) s->ch_dens = qa;
        else if (s->x==x0+1 && s->y==y0 && s->z==z0 && s->t==t0) s->ch_dens = qb;
        else                                                     s->ch_dens = 0;
    }
    tcd_corrs_by_fourier();
    corr_by_spatial_distance(corrs_dist);

    // test correlators
    g_sync();
    node0_printf("Test C(x):\n");
    g_sync();
    FORALLSITES(i,s) {
        double expected;
        if      (s->x==0    && s->y==0 && s->z==0 && s->t==0) expected = (qa*qa + qb*qb) / volume;
        else if (s->x==1    && s->y==0 && s->z==0 && s->t==0) expected = qa*qb / volume;
        else if (s->x==nx-1 && s->y==0 && s->z==0 && s->t==0) expected = qa*qb / volume;
        else                                                  expected = 0;
        expected = expected * pow(nt, 8);
        double actual = s->ch_dens_corr.real;
        if (fabs(actual - expected) > 1e-8) {
            printf("site (x=%d,y=%d,z=%d,t=%d): expected: %.16g, actual: %.16g\n", s->x, s->y, s->z, s->t, expected, actual);
            fflush(stdout);
            break;
        }
    }

    // test C(t,r)
    g_sync();
    node0_printf("Test C(t,r):\n");
    g_sync();
    for(int t=0; t<=nt/2; t++) {
        int passed = 1;
        for(int s2=0; s2<=s2_max; s2++) {
            double expected;
            if      (t==0 && s2==0) expected = (qa*qa + qb*qb) / volume;
            else if (t==0 && s2==1) expected = 2 * qa*qb / volume;
            else                    expected = 0;
            expected = expected * pow(nt, 6);
            double actual = corrs_dist[t][s2];
            if (fabs(actual - expected) > 1e-8) {
                node0_printf("t=%d, r^2=%d: expected: %.16g, actual: %.16g\n", t, s2, expected, actual);
                fflush(stdout);
                passed = 0;
                break;
            }
        }
        if (passed == 0) break;
    }



}



