// ==================================================================
// Use OpenMP to test Uncertain Qualification
// Date: 11st March 2023
// ==================================================================
#include <fstream>
#include <chrono>
#include "Model_wall.hpp"
#include "Time_solver.hpp"
#include <random>
#include <mpi.h>

double test(const double * P_k, const double * P_G, const double * P_c);

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  uniform_real_distribution<double> c_m3(0.875, 6.125);
  uniform_real_distribution<double> c_c3(5.5, 38.5);
  default_random_engine e(time(NULL));
  int num_sim = 200;
  double * mean_value = new double[num_sim];

  int num_procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_sim_per_proc = num_sim / num_procs;
  double * local_mean_value = new double[num_sim_per_proc];

  double * P_k = new double[4];
  P_k[0] = 1.0; // K_c1
  P_k[1] = 1.0; // K_c2
  P_k[2] = 1.0; // K_m1
  P_k[3] = 1.0; // K_m2

  double * P_G = new double [4];
  P_G[0] = 1.08; // G_ch
  P_G[1] = 1.20; // G_mh
  P_G[2] = 1.40; // G_et
  P_G[3] = 1.40; // G_ez

  double * P_c = new double[2];

  int seed = time(NULL) + rank;
  default_random_engine local_e(seed); 

  for (int ii = rank * num_sim_per_proc; ii < (rank + 1) * num_sim_per_proc; ii++)
  {
    P_c[0] = c_m3(local_e); // c_m3
    P_c[1] = c_c3(local_e); // c_c3

    local_mean_value[ii - rank * num_sim_per_proc] = test(P_k, P_G, P_c); 
    cout << "This is No." << ii << '\t';
    cout << " simualtion used Processor " << rank << '\t';
    cout << "Mean value is " << local_mean_value[ii - rank * num_sim_per_proc] << '\n';
  }
  // Delete P_G, P_k, and P_c after each loop
  delete[] P_G;
  delete[] P_k;
  delete[] P_c;

  MPI_Gather(local_mean_value, num_sim_per_proc, MPI_DOUBLE, mean_value, num_sim_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    ofstream MC_mean;
    MC_mean.open("mean-data.txt");
    double sum_time = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      sum_time += mean_value[ii];
      MC_mean << sum_time / double(ii+1) << endl;
    }
    MC_mean.close();

    ofstream MC_var;
    MC_var.open("var-data.txt");
    double var = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      var += pow(mean_value[ii] - (sum_time / double(num_sim)), 2);
      MC_var << var / double(ii+1) << endl;
    }
    MC_var.close();
  }

  MPI_Finalize();
}


double test(const double * P_k, const double * P_G, const double * P_c )
{
  const double pi = atan(1) * 4;

  // ----------- Time Solver ---------------
  const int steps_pday = 10;
  const int lifespan = 1000;
  const int simlength = 1000;
  const int ref_days = 0; 
  Time_solver * tsolver = new Time_solver(steps_pday, lifespan, simlength);

  // tsolver->print_timeinfo();
  // ---------------------------------------

  // ----------- Wall Object ---------------
  const double dP = 1.0;
  const double dQ = 1.3;
  Model_wall * wall = new Model_wall(pi, dP, dQ, tsolver->get_num_t(),
      tsolver->get_num_DL(), tsolver->get_dt(), P_k, P_G, P_c);

  // wall->print_fluid_properties();

  // wall->print_solid_properties();

  // wall->check_initial_parameters();

  const double alpha_ckh[4] = {0.0, 0.5*pi, 0.25*pi, 0.75*pi};
  // wall->check_initial_angle(alpha_ckh);

  // wall->check_initial_stress();
  // ---------------------------------------


  // ------------ Nonlinear Solver ---------
  const double Max_error_a = 2.0e-9, Max_error_m = 1.0e-9;
  const int Max_it = 90;
  int num_it1 = 0, num_it2 = 0; 
  double beta = 0.3, tol_a = 100.0, tol_m = 100.0; 
  // ---------------------------------------


  // ----- Working variable in solver -----
  double L_z = 1.0, L_t;
  double a_act_p  = wall->get_a_M();
  double da_act_p = 0.0;
  const double k_act = 1.0 / 20.0; 

  double dwdLt_c, dwdLz_c, dwdLt_m, dwdLt_e;
  double ddwddLt_c, ddwddLt_m, ddwddLt_e;
  double M_ck[4] = {0.0, 0.0, 0.0, 0.0};
  double M_m = 0.0, M_c = 0.0;
  double Lc_k[4] = {0.0, 0.0, 0.0, 0.0};
  double Lc_kn[4] = {0.0, 0.0, 0.0, 0.0};
  double Lc_k_tau[4] = {0.0, 0.0, 0.0, 0.0};
  double alpha_tau[4] = {0.0, 0.5*pi, 0.0, 0.0};
  int tn0;
  double wt;
  double Lt_tau;
  double Lz_tau = 1.0;
  double Lm_n;
  double C_t, dC_t, T_act, dT_act;
  double Fa, dFa_da;
  // double h_h;
  // double total_M;
  // double tau_w;
  double t_homeostasis = 0.0;
  double radius_t[tsolver->get_num_t()];
  double M_m_t[tsolver->get_num_t()];
  double M_ck_t[tsolver->get_num_t()][4];
  M_m_t[0] = wall->get_M_mh();
  for( int ii = 0; ii < 4; ii++)
  {
    M_ck_t[0][ii] = wall->get_M_ckh(ii);
  }
  radius_t[0] = wall->get_a_M();
  // --------------------------------------

  // ----- Prepare file for recording -----
  //ofstream outfile( "results", ofstream::out | ofstream::trunc );

  //if(!outfile)
  //{
  //cerr<<"Error: unable to open file to record results. \n";
  //exit(EXIT_FAILURE);
  //}

  // --------------------------------------
  for( int n_t = 1; n_t < tsolver->get_num_t(); ++n_t )
  {
    double t = n_t * tsolver->get_dt();

    double P = wall->get_P(t,ref_days); 
    double Q = wall->get_Q(t,ref_days);

    // ! Warning : This predictor is not a standard one
    wall->predictor(n_t, 0.1);

    double a_t = wall->get_Da(n_t);

    // ! Warning : This predictor is not a standard one 
    double a_act = a_act_p + 0.5 * tsolver->get_dt() *
      (da_act_p + k_act * (a_t - (a_act_p + tsolver->get_dt()*da_act_p)));

    tol_m = 100.0; num_it1 = 0;

    tn0 = SYS_T::get_tn0(n_t, tsolver->get_num_DL()); 

    //outfile<<t<<'\t'<<P<<'\t'<<Q<<'\t';

    while( (tol_m > Max_error_m) && (num_it1 < Max_it) )
    {
      num_it1 += 1; 
      tol_a = 100.0; num_it2 = 0;
      while( (tol_a > Max_error_a) && (num_it2 < Max_it) )
      {
        num_it2 += 1;

        double tau_w = 4.0 * wall->get_mu() * Q / (pi*a_t*a_t*a_t);

        L_t = a_t / wall->get_a_M();

        // Update the angle based on L_t
        wall->set_Dalpha(n_t, L_t, L_z); 

        // calculate the stress and d_stress 
        // -- stress
        dwdLt_c = 0.0;
        dwdLz_c = 0.0;
        dwdLt_m = 0.0;
        ddwddLt_c = 0.0;
        ddwddLt_m = 0.0;

        // -- mass initialization
        for(int ii=0; ii<4; ++ii) M_ck[ii] = 0.0;

        M_m = 0.0;

        // Calculate initial mass/energy with degradation
        if( n_t <= tsolver->get_num_DL() )
        {
          wall->get_Lk(Lc_k, L_t, L_z, alpha_ckh);

          for(int ii=0; ii<4; ++ii)
          {
            M_ck[ii]   = wall->get_M_ck(ii, n_t);
            dwdLt_c   += wall->get_dwdLt_c(M_ck[ii], L_t, L_z, 
                alpha_ckh[ii], Lc_k[ii], 1.0);
            dwdLz_c   += wall->get_dwdLz_c(M_ck[ii], L_t, L_z, 
                alpha_ckh[ii], Lc_k[ii], 1.0);
            ddwddLt_c += wall->get_ddwddLt_c(M_ck[ii], L_t, L_z, 
                alpha_ckh[ii], Lc_k[ii], 1.0);
          }

          M_m        = wall->get_M_m(n_t);
          dwdLt_m   += wall->get_dwdLt_m(M_m, L_t, 1.0);
          ddwddLt_m += wall->get_ddwddLt_m(M_m, L_t, 1.0);
        }


        // Calculate viscoelasticity
        for(int n_tau = tn0; n_tau <= n_t; ++n_tau)
        {
          if(n_tau == tn0 || n_tau == n_t) wt = 0.5 * tsolver->get_dt();
          else wt = tsolver->get_dt();

          alpha_tau[2] = wall->get_Dalpha(n_tau);
          alpha_tau[3] = 2.0 * pi - alpha_tau[2];

          Lt_tau = wall->get_Da(n_tau) / wall->get_a_M();      

          wall->get_Lk(Lc_k_tau, Lt_tau, Lz_tau, alpha_tau);

          // This following is from the old code !!!
          wall->get_Lk(Lc_k, L_t, L_z, alpha_tau); 

          for(int ii=0; ii<4; ++ii)
          {
            Lc_kn[ii] = wall->get_Gch() * Lc_k[ii] / Lc_k_tau[ii];
            if(Lc_kn[ii] <= wall->get_y_Lkn())
            {
              const double new_cmass = wall->get_mc_tau(n_t, n_tau, 
                  ii, tsolver->get_dt(), wt);
              M_ck[ii] += new_cmass;
              dwdLt_c  += wall->get_dwdLt_c(new_cmass, L_t, L_z, 
                  alpha_tau[ii], Lc_k[ii], Lc_k_tau[ii]);
              dwdLz_c  += wall->get_dwdLz_c(new_cmass, L_t, L_z, 
                  alpha_tau[ii], Lc_k[ii], Lc_k_tau[ii]);
              ddwddLt_c += wall->get_ddwddLt_c(new_cmass, L_t, L_z,
                  alpha_tau[ii], Lc_k[ii], Lc_k_tau[ii]);
            }
          }

          Lm_n = wall->get_Gmh() * L_t / Lt_tau;

          if(Lm_n <= wall->get_y_Lmn())
          {
            const double new_mmass = wall->get_mm_tau(n_t, n_tau,
                tsolver->get_dt(), wt);
            M_m += new_mmass;

            dwdLt_m += wall->get_dwdLt_m(new_mmass, L_t, Lt_tau);

            ddwddLt_m += wall->get_ddwddLt_m(new_mmass, L_t, Lt_tau);
          }
        }

        dwdLt_e = wall->get_dwdLt_e( L_t * wall->get_Get(), 
            L_z * wall->get_Gez() );
        ddwddLt_e = wall->get_ddwddLt_e( L_t * wall->get_Get(), 
            L_z * wall->get_Gez() );

        // calculate active stress
        const double L_m_act = a_t / a_act;

        C_t    = wall->get_C_t( tau_w );
        dC_t   = wall->get_dC_t( Q, a_t );
        T_act  = wall->get_T_act( M_m, L_m_act, L_t*L_z, C_t );
        dT_act = wall->get_dT_act( M_m, L_m_act, a_t, L_z, a_act,
            L_t * L_z, C_t, dC_t );

        // calculate a_t and related data
        Fa = ( (dwdLt_c + dwdLt_m + dwdLt_e) / L_z ) + T_act - P * a_t;
        dFa_da = ( (ddwddLt_c + ddwddLt_m + ddwddLt_e) / (L_z*wall->get_a_M()) ) 
          + dT_act - P;

        a_t -= beta * Fa / dFa_da;

        a_act = (a_act_p + 0.5*tsolver->get_dt()*(da_act_p + k_act*a_t))
          /(1.0 + 0.5*tsolver->get_dt()*k_act);

        L_t = a_t / wall->get_a_M();

        tol_a = wall->l2error_a(a_t, n_t);

        // set in wall object
        wall->set_Da(n_t, a_t);
        wall->set_Dalpha(n_t, L_t, L_z); 
      } // end while tol_a > Max_error && num_it2 < Max_it

      if(num_it2 == Max_it) beta = 0.1;

      M_c = 0.0;

      for(int ii=0; ii<4; ++ii) M_c += M_ck[ii];

      double error_c, error_bottom_c, error_m, error_bottom_m;

      // calculate the new mass for collagen and muscle
      wall->update_m_c( n_t, L_t, L_z, dwdLt_c, dwdLz_c, M_c, C_t,
          error_c, error_bottom_c );

      wall->update_m_m( n_t, L_t, L_z, dwdLt_m, T_act, M_m, C_t,
          error_m, error_bottom_m );

      tol_m = sqrt((error_c + error_m) / (error_bottom_c + error_bottom_m));
    } // end while tol_m > Max_error && num_it1 < Max_it

    a_act_p = a_act;
    da_act_p = k_act * (a_t - a_act);

    wall->set_Dalpha(n_t, L_t, L_z);

    // double M_e = wall->get_M_eh();
    // total_M = M_c + M_e + M_m;
    // h_h = total_M / (wall->get_rho_s() * L_t * L_z);
    // tau_w = 4.0 * wall->get_mu() * Q / (pi*a_t*a_t*a_t);
    radius_t[n_t] = a_t;
    M_m_t[n_t] = M_m;
    for (int ii = 0; ii < 4; ii++)
    {
      M_ck_t[n_t][ii] = M_ck[ii];
    }
    //outfile<<a_t<<'\t'<<h_h<<'\t'<<M_c<<'\t'<<M_m<<'\t'<<M_e<<'\t'<<total_M<<'\t';
    //outfile<<wall->get_Dalpha(n_t);
    //outfile<<endl;
 
    const double tol_homeostasis = 1.0e-5;
    bool cdt1 = ( abs(radius_t[n_t]/radius_t[n_t] - 1.0) <= tol_homeostasis );
    bool cdt2 = ( abs(M_m_t[n_t]/M_m_t[n_t-1] - 1.0) <= tol_homeostasis );
    bool cdt3 = ( abs(M_ck_t[n_t][0]/M_ck_t[n_t-1][0] - 1.0) <= tol_homeostasis );
    bool cdt4 = ( abs(M_ck_t[n_t][1]/M_ck_t[n_t-1][1] - 1.0) <= tol_homeostasis );
    bool cdt5 = ( abs(M_ck_t[n_t][2]/M_ck_t[n_t-1][2] - 1.0) <= tol_homeostasis );
    bool cdt6 = ( abs(M_ck_t[n_t][3]/M_ck_t[n_t-1][3] - 1.0) <= tol_homeostasis );
    if ( cdt1 && cdt2 && cdt3 && cdt4 && cdt5 && cdt6 )
    {
      t_homeostasis = t; 
      break;
    }
    //cout<<"Time t= "<<t<<'\t';
    //cout<<"num_it1 = "<<num_it1<<'\t'<<"tol_a = "<<tol_a<<'\t';
    //cout<<"L_t = "<<L_t<<'\t'<<"h_h = "<<h_h<<'\t';
    //cout<<"total_M = "<<total_M<<'\t';
    //cout<<endl; 
  }
  return t_homeostasis;
  //outfile.close(); 
  delete wall; delete tsolver;
}
// EOF
