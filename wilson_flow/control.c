/************************* control.c *******************************/
/* Integrates the gauge fields with Wilson or Symanzik flow        */

/* Flag for control file of application */
#define CONTROL

/* Definitions, files, and prototypes */
#include "wilson_flow_includes.h"

/* Additional includes */
#ifdef HAVE_QIO
#include <qio.h>
#include "../include/io_scidac.h"
#endif

#ifdef DEBUG_FIELDS
void dump_double_lattice();
#endif




int
main( int argc, char **argv )
{
  /* control variables */
  int prompt;
  double dtime, dtimec, dclock();
  int i;
  site *s;


#ifdef SPHALERON
  double dtimebulk,dtimebdry; 
  Real q_bulk[3];
#endif




  /* Initialization */
  initialize_machine(&argc, &argv);
  if( remap_stdio_from_args(argc, argv) == 1 )
    terminate(1);
  g_sync();

  /* Start application timer */
  dtime = -dclock();

  /* Setup lattice parameters */
  prompt = setup();

  /* Allocate memory for temporary gather matricies */
  for( i=0; i<N_TEMPORARY; i++ )
    tempmat[i] = (su3_matrix *)malloc(sites_on_node * sizeof(su3_matrix));





  /* Loop over configurations */
  while( readin(prompt) == 0 ) {


    /* Start timer for this configuration (doesn't include load time) */
    dtimec = -dclock();   // node0_printf("starttime_flow = %e\n", -dtimec);


    /* integrate the flow */
#ifdef REGIONS
    run_gradient_flow( FULLVOL );
    // run_gradient_flow();
#else
    run_gradient_flow();
#endif


    /* Stop and print timer for this configuration */
    dtimec += dclock();
    node0_printf("Time to complete flow = %e seconds\n", dtimec);
    fflush(stdout);

    /* Save lattice if requested */
    if( saveflag != FORGET )
      save_lattice( saveflag, savefile, stringLFN );

    // /* Save topological charge density */
    // char topo_name[512];
    // strncpy(topo_name, savefile, sizeof(topo_name)-1);
    // topo_name[sizeof(topo_name)-1] = '\0';
    // char *dot = strrchr(topo_name, '.');
    // if (dot) *dot = '\0';
    // strncat(topo_name, ".tcd", sizeof(topo_name) - strlen(topo_name) - 1);
    // save_topo(topo_name);


    // /* Start timer for calculating this configurations correlation functions */
    // dtimec = -dclock();                         //node0_printf("time1 = %e, ", dtimec);


    // /* Computing charge correlators */
    // tcd_corrs_by_fourier();                       //node0_printf("time3 = %e, ", dtimec);


    // /* test fourier fourier^-1 = 1 */
    // double difff = -1;
    // FORALLSITES(i, s) {
    //   double difffn = fabs(s->ch_dens - s->ch_dens_corr.real);
    //   if (difffn > difff) difff = difffn;
    // }
    // g_doublemax(&difff);
    // node0_printf("Largest ff difference = %f\n", difff);


    // /* Stop and print timer for this configurations carroelation functions */
    // dtimec += dclock();
    // node0_printf("Time to compute correlators = %e seconds\n", dtimec);
    // fflush(stdout);



#ifdef DEBUG_FIELDS
      dump_double_lattice();
#endif

    

#ifdef SPHALERON

#ifdef DEBUG_BLOCKING
    test_blocking();
    normal_exit(0);
#else
#ifndef HALF_LATTICE_TEST

    /* Start timer for bulk flow (doesn't include 4D preflow time) */
    dtimebulk = -dclock();
    bulk_flow( q_bulk );
    /* Stop and print timer for bulk flow  */
    dtimebulk += dclock();
    node0_printf("Time to complete bulk flow = %e seconds\n", dtimebulk);
    fflush(stdout);
#else
    report_bulk( stoptime, q_bulk );
#endif
    /* Start timer for bdry flow (doesn't include 4D pre- and bulk-flow time) */
    dtimebdry = -dclock();
    bdry_flow( q_bulk );
    /* Stop and print timer for this configuration */
    dtimebdry += dclock();
    node0_printf("Time to complete bdry flow = %e seconds\n", dtimebdry);
    fflush(stdout);
#endif

#endif

  }/* end: loop over configurations */






  /* Notify user application is done */
  node0_printf("RUNNING COMPLETED\n");
  fflush(stdout);

  /* Stop and print application timer */
  dtime += dclock();
  node0_printf("Time = %e seconds\n", dtime);
  fflush(stdout);

  normal_exit(0);
  return 0;
}
