/*********************** save_topo.c *************************/
/* MIMD version 7 */

/* routine for FFdual output. */
/* This works for both Intel and Ncube, but other machines may need
   special treatment */

#include "wilson_flow_includes.h"

/* ~ from ../smooth_inst/save_topo.c */
void save_topo(char *filenam)
{
   FILE *fp = NULL;
   int currentnode,newnode;
   int i,x,y,z,t;
   Real lbuf;
  /* Hack to distinguish single and double precision files */
   // int32type topo_magic_number = TOPO_VERSION_NUMBER ;
   // int32type tmp;

   /* node 0 does all the writing */
   if(this_node==0)
   {
      fp = fopen(filenam,"w");   /* ascii mode */
      if(!fp){ printf("Cannot open %s\n",filenam); terminate(1); }
      /* header */
      fprintf(fp, "source lattice %s\n", startfile );
      fprintf(fp, "nx %d\nny %d\nnz %d\nnt %d\n", nx,ny,nz,nt);
   }
   g_sync();
   currentnode=0;

   for(t=0;t<nt;t++)for(z=0;z<nz;z++)for(y=0;y<ny;y++)for(x=0;x<nx;x++)
   {
      newnode=node_number(x,y,z,t);
      if(newnode != currentnode)
      { /* switch to another node */
         g_sync();
         currentnode=newnode;
      }

      if(this_node==0)
      {
         if(currentnode==0)
         {
            i=node_index(x,y,z,t);
            lbuf = (Real)lattice[i].ch_dens;
         }
         else
         {
            get_field((char *)&lbuf, sizeof(Real), currentnode);
         }
         fprintf(fp, "%d %d %d %d   %.8g\n", x,y,z,t, (double)lbuf);
      }
      else       /* for nodes other than 0 */
      {
         if(this_node==currentnode)
         {
            i=node_index(x,y,z,t);
            lbuf=lattice[i].ch_dens;
            send_field((char*)&lbuf, sizeof(Real), 0);
         }
      }
   }
   g_sync();
   if(this_node==0)
   {
      fclose(fp);
      printf("Saved ASCII topo in %s\n", filenam);
   }
}
