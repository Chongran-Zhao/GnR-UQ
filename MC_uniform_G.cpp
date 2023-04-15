// ==================================================================
// Use OpenMPI to test Uncertain Qualification
// Date: 11st March 2023
// ==================================================================
#include <fstream>
#include <chrono>
#include "Model_wall.hpp"
#include "Time_solver.hpp"
#include <random>
#include <mpi.h>

double * run_sim(const double * P_k, const double * P_G, const double * P_c);

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  uniform_real_distribution<double> G_ch(1.026, 1.134); // 1.08 x [0.95, 1.05]
  uniform_real_distribution<double> G_mh(1.14, 1.26);   // 1.20 x [0.95, 1.05]
  uniform_real_distribution<double> G_et(1.33, 1.47);   // 1.40 x [0.95, 1.05]
  uniform_real_distribution<double> G_ez(1.33, 1.47);   // 1.40 x [0.95, 1.05]
  default_random_engine e(time(NULL));
  int num_sim = 6400;
  double * mean_value_radius = new double[num_sim];
  double * mean_value_thickness  = new double[num_sim];
  double * mean_value_mass   = new double[num_sim];

  int num_procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_sim_per_proc = num_sim / num_procs;
  double * local_mean_value_radius = new double[num_sim_per_proc];
  double * local_mean_value_thickness  = new double[num_sim_per_proc];
  double * local_mean_value_mass   = new double[num_sim_per_proc];

  double P_k[4] = {1.0, 1.0, 1.0, 1.0}; // K_c1, K_c2, K_m1, K_m2
  double P_G[4];                        // G_ch, G_mh, G_et, G_ez
  double P_c[2] = {3.5, 22.0};          // c_m3, c_c3

  int seed = time(NULL) + rank;
  default_random_engine local_e(seed);
  int counter = num_sim / num_procs;
  for (int ii = rank * num_sim_per_proc; ii < (rank + 1) * num_sim_per_proc; ii++)
  {
    P_G[0] = G_ch(local_e);
    P_G[1] = G_mh(local_e);
    P_G[2] = G_et(local_e);
    P_G[3] = G_ez(local_e);

    double * result = run_sim(P_k, P_G, P_c);
    counter -= 1;
    local_mean_value_radius[ii - rank * num_sim_per_proc] = result[0];
    local_mean_value_thickness [ii - rank * num_sim_per_proc] = result[1];
    local_mean_value_mass  [ii - rank * num_sim_per_proc] = result[2];
    if (rank == 0)
    {
      cout << "There still exist " << counter << " * " << num_procs << " simulations" << endl; 
    }
  }

  MPI_Gather(local_mean_value_radius, num_sim_per_proc, MPI_DOUBLE, mean_value_radius, num_sim_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(local_mean_value_thickness , num_sim_per_proc, MPI_DOUBLE, mean_value_thickness , num_sim_per_proc, MPI_DOUBLE, 1, MPI_COMM_WORLD);
  MPI_Gather(local_mean_value_mass  , num_sim_per_proc, MPI_DOUBLE, mean_value_mass  , num_sim_per_proc, MPI_DOUBLE, 2, MPI_COMM_WORLD);
  MPI_Gather(local_mean_value_radius, num_sim_per_proc, MPI_DOUBLE, mean_value_radius, num_sim_per_proc, MPI_DOUBLE, 3, MPI_COMM_WORLD);
  MPI_Gather(local_mean_value_thickness , num_sim_per_proc, MPI_DOUBLE, mean_value_thickness , num_sim_per_proc, MPI_DOUBLE, 4, MPI_COMM_WORLD);
  MPI_Gather(local_mean_value_mass  , num_sim_per_proc, MPI_DOUBLE, mean_value_mass  , num_sim_per_proc, MPI_DOUBLE, 5, MPI_COMM_WORLD);


  if (rank == 0) {
    double tol = 1.0e-8; 
    ofstream MC_mean_radius;
    MC_mean_radius.open("G-mean-value-radius.txt");
    double sum_radius = 0.0;
    int counter = 0;
    double error = 1.0;
    while (counter < num_sim && error > tol)
    { 
      if (counter > 0) {error = sum_radius / double(counter);}
      sum_radius += mean_value_radius[counter];
      if (counter > 0) {error = abs(error - sum_radius/double(counter+1)) / error;}
      MC_mean_radius << sum_radius / double(counter+1) << endl;
      counter += 1;
    }
    MC_mean_radius << "It took " << counter << " samples to converge." << endl;
    MC_mean_radius.close();

    ofstream MC_var_radius;
    MC_var_radius.open("G-var-radius.txt");
    double var_radius = 0.0;
    for (int ii = 0; ii < counter; ii++)
    {
      var_radius += pow(mean_value_radius[ii] - (sum_radius / double(num_sim)), 2);
      MC_var_radius << var_radius / double(ii+1) << endl;
    }
    MC_var_radius.close();
  }

  if (rank == 1) {
    double tol = 1.0e-8;
    ofstream MC_mean_thickness;
    MC_mean_thickness.open("G-mean-value-thickness.txt");
    double sum_thickness = 0.0;
    int counter = 0;
    double error = 1.0;
    while (counter < num_sim && error > tol)
    {
      if (counter > 0) {error = sum_thickness / double(counter);}
      sum_thickness += mean_value_thickness[counter];
      if (counter > 0) {error = abs(error - sum_thickness/double(counter+1)) / error;}
      MC_mean_thickness << sum_thickness / double(counter+1) << endl;
      counter += 1;
    }
    MC_mean_thickness << "It took " << counter << " samples to converge." << endl;
    MC_mean_thickness.close();

    ofstream MC_var_thickness;
    MC_var_thickness.open("G-var-thickness.txt");
    double var_thickness = 0.0;
    for (int ii = 0; ii < counter; ii++)
    {
      var_thickness += pow(mean_value_thickness[ii] - (sum_thickness / double(num_sim)), 2);
      MC_var_thickness << var_thickness / double(ii+1) << endl;
    }
    MC_var_thickness.close();
  }

  if (rank == 2) {
    double tol = 1.0e-8;
    ofstream MC_mean_mass;
    MC_mean_mass.open("G-mean-value-mass.txt");
    double sum_mass = 0.0;
    int counter = 0;
    double error = 1.0;
    while (counter < num_sim && error > tol)
    {
      if (counter > 0) {error = sum_mass / double(counter);}
      sum_mass += mean_value_mass[counter];
      if (counter > 0) {error = abs(error - sum_mass/double(counter+1)) / error;}
      MC_mean_mass << sum_mass / double(counter+1) << endl;
      counter += 1;
    }
    MC_mean_mass << "It took " << counter << " samples to converge." << endl;
    MC_mean_mass.close();

    ofstream MC_var_mass;
    MC_var_mass.open("G-var-mass.txt");
    double var_mass = 0.0;
    for (int ii = 0; ii < counter; ii++)
    {
      var_mass += pow(mean_value_mass[ii] - (sum_mass / double(num_sim)), 2);
      MC_var_mass << var_mass / double(ii+1) << endl;
    }
    MC_var_mass.close();
  }
  if (rank == 3) {
    ofstream MC_global_mean_radius;
    MC_global_mean_radius.open("G-global-mean-radius.txt");
    double sum_radius = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      sum_radius += mean_value_radius[ii];
      MC_global_mean_radius << sum_radius / double(ii+1) << endl;
    }
    MC_global_mean_radius.close();

    ofstream MC_global_var_radius;
    MC_global_var_radius.open("G-global-var-radius.txt");
    double global_var_radius = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      global_var_radius += pow(mean_value_radius[ii] - (sum_radius / double(num_sim)), 2);
      MC_global_var_radius << global_var_radius / double(ii+1) << endl;
    }
    MC_global_var_radius.close();
  }
  if (rank == 4) {
    ofstream MC_global_mean_thickness;
    MC_global_mean_thickness.open("G-global-mean-thickness.txt");
    double sum_thickness = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      sum_thickness += mean_value_thickness[ii];
      MC_global_mean_thickness << sum_thickness / double(ii+1) << endl;
    }
    MC_global_mean_thickness.close();

    ofstream MC_global_var_thickness;
    MC_global_var_thickness.open("G-global-var-thickness.txt");
    double global_var_thickness = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      global_var_thickness += pow(mean_value_thickness[ii] - (sum_thickness / double(num_sim)), 2);
      MC_global_var_thickness << global_var_thickness / double(ii+1) << endl;
    }
    MC_global_var_thickness.close();
  }

  if (rank == 5) {
    ofstream MC_global_mean_mass;
    MC_global_mean_mass.open("G-global-mean-mass.txt");
    double sum_mass = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      sum_mass += mean_value_mass[ii];
      MC_global_mean_mass << sum_mass / double(ii+1) << endl;
    }
    MC_global_mean_mass.close();

    ofstream MC_global_var_mass;
    MC_global_var_mass.open("G-global-var-mass.txt");
    double global_var_mass = 0.0;
    for (int ii = 0; ii < num_sim; ii++)
    {
      global_var_mass += pow(mean_value_mass[ii] - (sum_mass / double(num_sim)), 2);
      MC_global_var_mass << global_var_mass / double(ii+1) << endl;
    }
    MC_global_var_mass.close();
  }

  MPI_Finalize();
}

double * run_sim(const double * P_k, const double * P_G, const double * P_c )
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
  double h_h;
  double total_M; 
  double * result = new double[3];
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

    double M_e = wall->get_M_eh();
    total_M = M_c + M_e + M_m;
    h_h = total_M / (wall->get_rho_s() * L_t * L_z);
    // tau_w = 4.0 * wall->get_mu() * Q / (pi*a_t*a_t*a_t);
    radius_t[n_t] = a_t;
    M_m_t[n_t] = M_m;
    for (int ii = 0; ii < 4; ii++)
    {
      M_ck_t[n_t][ii] = M_ck[ii];
    }

    const double tol_homeostasis = 1.0e-5;
    bool cdt1 = ( abs(radius_t[n_t]/radius_t[n_t] - 1.0) <= tol_homeostasis );
    bool cdt2 = ( abs(M_m_t[n_t]/M_m_t[n_t-1] - 1.0) <= tol_homeostasis );
    bool cdt3 = ( abs(M_ck_t[n_t][0]/M_ck_t[n_t-1][0] - 1.0) <= tol_homeostasis );
    bool cdt4 = ( abs(M_ck_t[n_t][1]/M_ck_t[n_t-1][1] - 1.0) <= tol_homeostasis );
    bool cdt5 = ( abs(M_ck_t[n_t][2]/M_ck_t[n_t-1][2] - 1.0) <= tol_homeostasis );
    bool cdt6 = ( abs(M_ck_t[n_t][3]/M_ck_t[n_t-1][3] - 1.0) <= tol_homeostasis );
    if ( cdt1 && cdt2 && cdt3 && cdt4 && cdt5 && cdt6 )
    {
      result[0] = a_t;
      result[1] = h_h;
      result[2] = total_M; 
      break;
    }
  }
  return result; 
  delete wall; delete tsolver;
}
// EOF
