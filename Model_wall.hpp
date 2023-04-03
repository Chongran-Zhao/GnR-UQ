#ifndef MODEL_WALL_HPP
#define MODEL_WALL_HPP

#include <cmath>
#include <cstdlib>
#include <cassert>
#include "Sys_Tools.hpp"

class Model_wall
{
  public:
    Model_wall( const double &in_pi, const double &in_deltaP,
        const double &in_deltaQ, const int &in_num_t,
        const int &in_num_DL, const double &in_dt,
        const double * in_K, const double * in_G, const double * in_C);  
     

    virtual ~Model_wall();

    void print_fluid_properties() const;
    
    void print_solid_properties() const;

    void check_initial_angle( const double * const &in_alpha ) const;

    void check_initial_stress() const;

    // 1st order derivatives
    double dWkdLn( const double &stretch ) const;

    double dWmdLn( const double &stretch ) const;

    double dWedLn( const double &stretch_t, 
        const double &stetch_z  ) const;

    // 2nd order derivatives
    double ddWkddLn( const double &stretch ) const;

    double ddWmddLn( const double &stretch ) const;
   
    double ddWeddLn( const double &stretch_t,
       const double &stretch_z ) const;

    void check_initial_parameters() const;

    double q_i_m( const double &s ) const 
    {return exp( -kq_m * s );}

    double q_i_c( const double &s ) const 
    {return exp( -kq_c * s );}

    // get pressure & flow rate
    double get_P(const double &time, const double &ref_time) const;
    double get_Q(const double &time, const double &ref_time) const;
    

    // predictor for solutions
    void predictor(const int &nt, const double &tol);

    // get functions
    // get data structure values
    double get_mu() const {return mu;}
    double get_a_M() const {return a_M;}
    double get_Da(const int &ii) const {return Da[ii];}
    double get_M_ck(const int &ii, const int &tstep) const; 
    double get_M_m(const int &tstep) const {return M_mh * DQ_m[tstep];}
    
    double get_M_eh() const {return M_eh;}
    double get_M_mh() const {return M_mh;}
    double get_M_ckh(const int &ii) const {return M_ckh[ii];}
    double get_rho_s() const {return rho_s;}

    double get_Dalpha(const int &ii) const {return Dalpha[ii];}
    double get_Gch() const {return G_ch;}
    double get_Gmh() const {return G_mh;}
    double get_Get() const {return G_et;}
    double get_Gez() const {return G_ez;}

    //double get_DQ_m(const int &tstep) const {return DQ_m[tstep];}
    //double get_DQ2_m(const int &tstep) const {return DQ2_m[tstep];}
    //double get_DQ_c(const int &tstep) const {return DQ_c[tstep];}
    //double get_DQ2_c(const int &tstep, const int &ii) {return DQ2_c[tstep][ii];}

    double get_y_Lkn() const {return y_Lkn;}
    double get_y_Lmn() const {return y_Lmn;}

    double get_mc_tau(const int &s, const int &tau, const int &ii, 
        const double &dt, const double &wt) const;

    double get_mm_tau(const int &s, const int &tau, const double &dt,
        const double &wt) const;

    // Given stretch Lt & Lz, return stretch based on alpha_ckh
    void get_Lk( double * const &Lc_k, const double &Lt, 
        const double &Lz, const double * const &angle ) const 
    {SYS_T::get_Lk(Lc_k, Lt, Lz, angle, 4);}


    // Collagen
    // get dw_dLt
    double get_dwdLt_c( const double &mass,
        const double &Lt, const double &Lz, const double &angle, 
        const double &lambda_s, const double &lambda_tau ) const;

    // get dwdLz
    double get_dwdLz_c( const double &mass,
        const double &Lt, const double &Lz, const double &angle, 
        const double &lambda_s, const double &lambda_tau ) const;

    // get ddwddLt
    double get_ddwddLt_c( const double &mass,
       const double &Lt, const double &Lz, const double &angle,
       const double &lambda_s, const double &lambda_tau ) const;


    // Muscle
    // get dw_dLt
    double get_dwdLt_m( const double &mass,
       const double &lambda_s, const double &lambda_tau ) const;

    // get ddw_ddLt
    double get_ddwddLt_m( const double &mass, 
       const double &lambda_s, const double &lambda_tau ) const;


    // Elastin
    double get_dwdLt_e( const double &Lt, const double &Lz ) const;

    double get_ddwddLt_e( const double &Lt, const double &Lz ) const;


    // Active stress
    double get_C_t(const double &in_tau_w) const;

    double get_dC_t( const double &in_Q_M, const double &in_a_t) const;

    double get_T_act( const double &mmass, const double &inLm_act,
        const double &J, const double &in_C_t ) const;

    double get_dT_act( const double &mmass, const double &inLm_act,
        const double &in_a_t, const double &L_z, const double in_a_act,
        const double &J, const double &in_C_t, 
        const double &in_dC_t ) const;


    // mass rate
    void update_m_c( const int &tstep, const double &Lt, const double &Lz, 
        const double &dwdLt, const double &dwdLz, const double &Mc,
        const double &C_t, double &error, double &error_bottom );


    void update_m_m( const int &tstep, const double &Lt, const double &Lz,
        const double &dwdLt, const double &Tact, const double &Mm,
        const double &C_t, double &error, double &error_bottom );

    // error function
    double l2error_a( const double &in_a, const int &tstep ) const;


    // set functions
    void set_Da(const int &ii, const double &in_at) {Da[ii] = in_at;}

    void set_Dalpha(const int &ii, const double &Lt, const double &Lz) 
    {Dalpha[ii] = atan( Lt * tan(alpha_ckh[2]) / Lz );}    


  private:
    const double pi;

    // properties of the solids
    const double rho_s; // wall density g/cm^3
    const double sigma_h; // hemeostatic circumferential stress

    // properties of the fluids
    const double a_M; // initial vessel radius    
    const double tau_wh; // hemeostatic WSS 
    const double mu; // viscosity
    const double P_h; //transmural pressure
    const double Q_h; // homeostatic flow rate 
    const double h_h; // vessel thickness
    const double delta_P; // P = P_h * delta_P for growth simulation
    const double delta_Q; // Q = Q_h * delta_Q for growth simulation

    // mass production rate for collagen
    const double K_c1, K_c2;
    // mass production rate for muscle
    const double K_m1, K_m2;

    // Deposition stretches
    const double G_ch, G_mh, G_et, G_ez;

    // mass fractions
    const double phi_c, phi_e, phi_m;

    double c_frac[4];
    double phi_ck[4];

    // Fiber angles
    double alpha_ckh[4];

    // Homeostatic mass
    const double M_h, M_mh, M_eh;
    double M_ckh[4];

    // Basal mass production rate
    double m_basal_ck[4];
    double m_basal_m;

    // Homeostatic stress
    const double sigma_ch, sigma_mh, sigma_eh;

    // Muscle activation stress parameters
    const double Lambda_M, Lambda_0, C_basal, ratio_C, T_S0, T_M;
    double sigma_act_h;

    // Yield limit stretch
    const double y_Lkn, y_Lmn;

    // Half-life length
    const double kq_m, kq_c;

    // Rate parameter for active VSM tone
    // const double k_act;

    // Remaining material parameter
    double c_e, c_m2, c_m3, c_c2, c_c3; 
    double c_m2_c, c_m3_c, c_c2_c, c_c3_c;

    // Degradation functions
    const int num_t, num_DL;
    double * DQ_m;
    double * DQ_c;

    // solutions
    double * Da;
    double * Dm_m;
    double ** Dm_c;
    double * Dalpha;

};


#endif
