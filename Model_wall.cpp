#include "Model_wall.hpp"

Model_wall::Model_wall( const double &in_pi, const double &in_deltaP,
    const double &in_deltaQ, const int &in_num_t,
    const int &in_num_DL, const double &in_dt,
    const double * in_K, const double * in_G, 
    const double * in_C ) 
: pi(in_pi), rho_s(1.050), sigma_h(100.0 * 1000.0 * 10.0),
  a_M(0.142), tau_wh(50.6), mu(0.037), P_h(93.0*1333.2237),
  Q_h(tau_wh * in_pi * a_M * a_M * a_M / (4.0 * mu)),
  h_h(P_h * a_M / sigma_h), delta_P(in_deltaP),
  delta_Q(in_deltaQ), 
  K_c1(in_K[0]), K_c2(in_K[1]), K_m1(in_K[2]), K_m2(in_K[3]),
  G_ch(in_G[0]), G_mh(in_G[1]), G_et(in_G[2]), G_ez(in_K[3]),
  phi_c(0.22), phi_e(0.02), phi_m(1.0 - phi_c - phi_e),
  M_h(h_h * rho_s), M_mh(phi_m * M_h), M_eh(phi_e * M_h),
  sigma_ch(sigma_h), sigma_mh(sigma_h), 
  sigma_eh( (sigma_h - phi_c * sigma_ch - phi_m * sigma_mh) / phi_e ),
  Lambda_M(1.1), Lambda_0(0.4), C_basal(0.68), ratio_C(20.0),
  T_S0(ratio_C * C_basal), T_M(150.0 * 1000.0 * 10.0),
  y_Lkn(3.0), y_Lmn(3.0), kq_m(0.04953), kq_c(0.34687),
  c_m3(in_C[0]), c_c3(in_C[1]),
  num_t(in_num_t), num_DL(in_num_DL) 
{
  c_frac[0] = 0.1;
  c_frac[1] = 0.1;
  c_frac[2] = 0.4;
  c_frac[3] = 0.4;

  for(int ii=0; ii<4; ++ii) phi_ck[ii] = phi_c * c_frac[ii];

  alpha_ckh[0] = 0.0;
  alpha_ckh[1] = 0.50 * in_pi;
  alpha_ckh[2] = 0.25 * in_pi;
  alpha_ckh[3] = 0.75 * in_pi;

  for(int ii=0; ii<4; ++ii) M_ckh[ii] = M_h * phi_ck[ii];

  sigma_act_h = T_M * (1.0 - exp(-1.0 * C_basal*C_basal))
    *(1.0 - pow((Lambda_M - 1.0)/(Lambda_M - Lambda_0), 2.0) );

  if(sigma_act_h < 0.0) sigma_act_h = 0.0;

  c_e = sigma_eh / (G_et*G_et - 1.0/(G_et*G_et*G_ez*G_ez));
  
  c_m2 = (sigma_mh - sigma_act_h) / (G_mh*G_mh*(G_mh*G_mh - 1.0)
      * exp(c_m3 * pow((G_mh * G_mh - 1.0), 2.0)));

  double str_ch = 0.0;
  for(int ii=0; ii<4; ++ii)
    str_ch += c_frac[ii] * G_ch * G_ch * (G_ch*G_ch-1.0)
      * exp(c_c3 * pow((G_ch*G_ch -1.0), 2.0)) * sin(alpha_ckh[ii])
      * sin(alpha_ckh[ii]) ;

  c_c2 = sigma_ch / str_ch;

  // Compressive parameters
  c_c2_c = c_m2; c_c3_c = c_m3;
  c_m2_c = c_m2; c_m3_c = c_m3;

  // Allocate memory for Degradation functions
  DQ_m = new double [num_DL];
  DQ_c = new double [num_DL]; 
 
  // Allocate memory for solutions 
  Da = new double [num_t];
  Dm_m = new double [num_t];
  Dm_c = new double * [num_t];
  Dalpha = new double [num_t];

  for(int ii=0; ii<num_t; ++ii) Dm_c[ii] = new double [4];

  // DQ_m & DQ_c initialization
  for(int ii=0; ii<num_DL; ++ii)
  {
    double t = ii * in_dt;
    DQ_m[ii] = q_i_m(t); 
    DQ_c[ii] = q_i_c(t);
  }
  
  const double mean_age_m = 1.0 / kq_m;
  const double mean_age_c = 1.0 / kq_c;

  for(int ii=0; ii<4; ++ii) m_basal_ck[ii] = M_ckh[ii] / mean_age_c;
  
  m_basal_m = M_mh / mean_age_m;
  
  // Initialization for solutions at time 0
  Da[0] = a_M;

  for(int ii=0; ii<4; ++ii) Dm_c[0][ii] = m_basal_ck[ii];

  Dm_m[0] = m_basal_m;

  Dalpha[0] = alpha_ckh[2];

  // Initialization for degradation functions
}

Model_wall::~Model_wall()
{
  delete [] DQ_m; delete [] DQ_c;
  delete [] Da; delete [] Dm_m; delete [] Dalpha;
  for(int ii=0; ii<num_t; ++ii)
  {
    delete [] Dm_c[ii];
  }
  delete [] Dm_c;
  cout<<"Model_wall class is deleted. \n";
}


void Model_wall::print_fluid_properties() const
{
  cout<<"Properties of the fluid: \n";
  cout<<" -- initial vessel radius : "<<a_M<<" cm \n";
  cout<<" -- homeostatic WSS : "<<tau_wh<<" dyn / cm^2 \n";
  cout<<" -- viscosity mu = "<<mu<<" g/cm^3 \n";
  cout<<" -- homeostatic transmural pressure "<<P_h<<" dyn/cm^2 \n";
  cout<<" -- homeostatic flow rate "<<Q_h<<" cm^3/s \n";
  cout<<" -- vessel thickness "<<h_h<<" cm \n";
  cout<<" -- delta_P = "<<delta_P<<'\n';
  cout<<" -- delta_Q = "<<delta_Q<<endl;
}


void Model_wall::print_solid_properties() const
{
  cout<<"Properties of the aterial wall: \n";
  cout<<" -- wall density rho_s = "<<rho_s<<'\n';
  cout<<" -- homeostatic circumferential stress : "<<sigma_h<<" \n";
  cout<<" -- mass production parameters: \n";
  cout<<"     K_c1 = "<<K_c1<<'\t';
  cout<<"     K_c2 = "<<K_c2<<'\t';
  cout<<"     K_m1 = "<<K_m1<<'\t';
  cout<<"     K_m2 = "<<K_m2<<'\n';
  cout<<" -- Deposition stretches : \n";
  cout<<"     G_ch = "<<G_ch<<"\t G_mh = "<<G_mh<<"\t G_et = "<<G_et;
  cout<<"\t G_ez = "<<G_ez<<'\n';
  cout<<" -- initial mass fractions: \n";
  cout<<"     phi_c = "<<phi_c<<'\t';
  cout<<"phi_e = "<<phi_e<<'\t'<<"phi_m = "<<phi_m<<'\n';
  cout<<" -- Collagen fractions : "<<c_frac[0]<<'\t'<<c_frac[1]<<'\t';
  cout<<c_frac[2]<<'\t'<<c_frac[3]<<'\n';
  cout<<" -- Collagen mass fractions : "<<phi_ck[0]<<'\t'<<phi_ck[1]<<'\t';
  cout<<phi_ck[2]<<'\t'<<phi_ck[3]<<'\n';
  cout<<" -- Collagen alignments : "<<alpha_ckh[0]<<'\t'<<alpha_ckh[1]<<'\t';
  cout<<alpha_ckh[2]<<'\t'<<alpha_ckh[3]<<'\n';
  cout<<" -- Homeostatic mass M_h = "<<M_h<<'\n';
  cout<<"    M_mh = "<<M_mh<<'\t'<<"M_eh = "<<M_eh<<'\n';
  cout<<"    M_ckh = "<<M_ckh[0]<<'\t'<<M_ckh[1]<<'\t'<<M_ckh[2]<<'\t'
    <<M_ckh[3]<<'\n';
  cout<<" -- Homeostatic stress \n";
  cout<<"    sigma_ch = "<<sigma_ch<<"\t sigma_mh = "<<sigma_mh;
  cout<<"\t sigma_eh = "<<sigma_eh<<'\n';
  cout<<" -- Muscle activation stress parameters: \n";
  cout<<"    Lambda_M = "<<Lambda_M<<"\t Lambda_0 = "<<Lambda_0<<'\n';
  cout<<"    C_basal = "<<C_basal<<"\t ratio_C = "<<ratio_C;
  cout<<"\t T_S0 = "<<T_S0<<"\t T_M = "<<T_M<<'\n'; 
  cout<<"    sigma_act_h = "<<sigma_act_h<<'\n';
  cout<<" -- Yield limit stretch: y_Lkn = "<<y_Lkn<<"\t y_Lmn = "<<y_Lmn<<'\n'; 
  cout<<" -- Half-life length: kq_m = "<<kq_m<<"\t kq_c = "<<kq_c<<'\n';
  cout<<" -- c_e = "<<c_e<<"\t c_m2 = "<<c_m2<<"\t c_c2 = "<<c_c2<<'\n';
  cout<<" -- Passive response parameters: c_m3 = "<<c_m3<<"\t c_c3 = "
    <<c_c3<<endl;
}


void Model_wall::check_initial_angle(const double * const &in_alpha) const
{
  for(int ii=0; ii<4; ++ii) assert(in_alpha[ii] == alpha_ckh[ii]);
  cout<<"===> Angle matched. \n";
}


void Model_wall::check_initial_stress() const
{
  double dwdLt_c = 0.0;
  for(int ii=0; ii<4; ++ii)
    dwdLt_c += phi_ck[ii] * h_h * dWkdLn(G_ch) * G_ch 
      * sin(alpha_ckh[ii]) * sin(alpha_ckh[ii]);

  const double dwdLt_m = h_h * phi_m * dWmdLn(G_mh) * G_mh;
  const double dwdLt_e = h_h * phi_e * dWedLn(G_et, G_ez) * G_et;
  const double dwdLt = dwdLt_c + dwdLt_m + dwdLt_e;
  const double T_act = sigma_act_h * phi_m * h_h;
  const double F = dwdLt + T_act - P_h * a_M;

  cout<<"The initial stress F = "<<F<<endl;
}


double Model_wall::dWkdLn( const double &Ln ) const
{
  const double temp = Ln * Ln - 1.0;
  if( Ln >= 1.0 )
    return c_c2 * Ln * temp * exp(c_c3 * temp * temp);
  else
    return c_c2_c * Ln * temp * exp(c_c3_c * temp * temp);
}


double Model_wall::dWmdLn( const double &Ln ) const
{
  const double temp = Ln * Ln - 1.0;
  if( Ln >= 1.0 )
    return c_m2 * Ln * temp * exp(c_m3 * temp * temp);
  else
    return c_m2_c * Ln * temp * exp(c_m3_c * temp * temp);
}


double Model_wall::dWedLn( const double &Ln_t, const double &Ln_z ) const
{
  const double Ln_t2_Ln_z2 = Ln_t * Ln_t * Ln_z * Ln_z;
  return c_e * (Ln_t - 1.0 / (Ln_t * Ln_t2_Ln_z2));
}


double Model_wall::ddWkddLn( const double &Ln ) const
{
  const double temp = Ln * Ln - 1.0;
  if(Ln >= 1.0)
    return c_c2 * (3.0 * Ln * Ln - 1.0 + 4.0 * c_c3 * Ln * Ln * temp * temp)
      *exp(c_c3*temp*temp);
  else
    return c_c2_c * (3.0 * Ln * Ln - 1.0 + 4.0 * c_c3_c * Ln * Ln * temp * temp)
      *exp(c_c3_c*temp*temp);
}


double Model_wall::ddWmddLn( const double &Ln ) const
{
  const double temp = Ln * Ln - 1.0;
  if(Ln >= 1.0)
    return c_m2 * (3.0 * Ln * Ln - 1.0 + 4.0 * c_m3 * Ln * Ln * temp * temp)
      *exp(c_m3*temp*temp);
  else
    return c_m2_c * (3.0 * Ln * Ln - 1.0 + 4.0 * c_m3_c * Ln * Ln * temp * temp)
      *exp(c_m3_c*temp*temp);
}


double Model_wall:: ddWeddLn( const double &Ln_t, const double &Ln_z ) const
{
  const double Ln_t2_Ln_z2 = Ln_t * Ln_t * Ln_z * Ln_z;
  return c_e * ( 1.0 + 3.0 / (Ln_t * Ln_t * Ln_t2_Ln_z2) );
}


void Model_wall::check_initial_parameters() const
{
  double residual = 0.0;

  for(int ii=0; ii<4; ++ii)
    residual += phi_ck[ii] * h_h * dWkdLn(G_ch) * G_ch 
      * sin(alpha_ckh[ii]) * sin(alpha_ckh[ii]);

  residual += h_h * phi_m * dWmdLn(G_mh) * G_mh;

  residual += h_h * phi_e * dWedLn(G_et, G_ez) * G_et;

  residual += sigma_act_h * phi_m * h_h;

  residual -= P_h * a_M;

  if( abs(residual) > 1.0e-1 )
  {
    cerr<<"residual F = "<<residual<<endl;
    exit(EXIT_FAILURE);
  }

  cout<<"The initial stress balance residual F := dwdLt + T_act - P_h x a_M = ";
  cout <<residual<<endl;
}


double Model_wall::get_P(const double &time, const double &ref_time) const
{
  return time>ref_time ? P_h * delta_P : P_h;
}


double Model_wall::get_Q(const double &time, const double &ref_time) const
{
  return time>ref_time ? Q_h * delta_Q : Q_h;
}


void Model_wall::predictor(const int &n_t, const double &tol)
{
  if( (n_t>2) && (abs(Da[n_t-1] - Da[n_t-2]) / Da[n_t-1] < tol) )
  {
    Da[n_t]      = 2.0 * Da[n_t - 1]    - Da[n_t - 2];
    Dm_c[n_t][0] = 2.0 * Dm_c[n_t-1][0] - Dm_c[n_t - 2][0];
    Dm_c[n_t][1] = 2.0 * Dm_c[n_t-1][1] - Dm_c[n_t - 2][1];
    Dm_c[n_t][2] = 2.0 * Dm_c[n_t-1][2] - Dm_c[n_t - 2][2];
    Dm_c[n_t][3] = 2.0 * Dm_c[n_t-1][3] - Dm_c[n_t - 2][3];
    Dm_m[n_t]    = 2.0 * Dm_m[n_t-1]    - Dm_m[n_t - 2];
  }
  else
  {
    Da[n_t]      = Da[n_t - 1];
    Dm_c[n_t][0] = Dm_c[n_t-1][0];
    Dm_c[n_t][1] = Dm_c[n_t-1][1];
    Dm_c[n_t][2] = Dm_c[n_t-1][2];
    Dm_c[n_t][3] = Dm_c[n_t-1][3];
    Dm_m[n_t]    = Dm_m[n_t-1];
  }
}


double Model_wall::get_M_ck(const int &ii, const int &tstep) const
{
  return M_ckh[ii] * DQ_c[tstep];
}


double Model_wall::get_dwdLt_c( const double &mass,
    const double &Lt, const double &Lz, const double &angle, 
    const double &lambda_s, const double &lambda_tau ) const
{
  const double dLndLt = G_ch * Lt * sin(angle) * sin(angle) / (lambda_s * lambda_tau);
  const double stretch = G_ch * lambda_s / lambda_tau;
  return mass * dWkdLn(stretch) * dLndLt / rho_s;
}


double Model_wall::get_dwdLz_c( const double &mass,
    const double &Lt, const double &Lz, const double &angle, 
    const double &lambda_s, const double &lambda_tau ) const
{
  const double dLndLz = G_ch * Lz * cos(angle) * cos(angle) / (lambda_s * lambda_tau);
  const double stretch = G_ch * lambda_s / lambda_tau;
  return mass * dWkdLn(stretch) * dLndLz / rho_s;
}


double Model_wall::get_ddwddLt_c( const double &mass,
    const double &Lt, const double &Lz, const double &angle, 
    const double &lambda_s, const double &lambda_tau ) const
{
  const double dLndLt = G_ch * Lt * sin(angle) * sin(angle) / (lambda_s * lambda_tau);
  const double ddLnddLt = G_ch * Lz * Lz * sin(angle) * sin(angle) 
    * cos(angle) * cos(angle) / (lambda_s * lambda_s * lambda_s);
  const double stretch = G_ch * lambda_s / lambda_tau;
  return mass 
    * (ddWkddLn(stretch)*dLndLt * dLndLt + dWkdLn(stretch) * ddLnddLt) / rho_s;
}


double Model_wall::get_dwdLt_m( const double &mass,
           const double &lambda_s, const double &lambda_tau ) const
{
  const double stretch = G_mh * lambda_s / lambda_tau;
  return mass * dWmdLn(stretch) * G_mh / (rho_s * lambda_tau);
}


double Model_wall::get_ddwddLt_m( const double &mass,
           const double &lambda_s, const double &lambda_tau ) const
{
  const double stretch = G_mh * lambda_s / lambda_tau;
  return mass * ddWmddLn(stretch) * G_mh * G_mh / (rho_s * lambda_tau * lambda_tau);
}


double Model_wall::get_mc_tau(const int &s, const int &tau, const int &ii,
    const double &dt, const double &wt) const
{
  return Dm_c[tau][ii] * q_i_c((s-tau)*dt) * wt;
}


double Model_wall::get_mm_tau(const int &s, const int &tau, const double &dt,
            const double &wt) const
{
  return Dm_m[tau] * q_i_m((s-tau)*dt) * wt;
}


double Model_wall::get_dwdLt_e( const double &Lt, const double &Lz ) const
{
  return M_eh * dWedLn(Lt, Lz) * G_et / rho_s;
}


double Model_wall::get_ddwddLt_e( const double &Lt, const double &Lz ) const
{
  return M_eh * ddWeddLn(Lt, Lz) * G_et * G_et / rho_s;
}


double Model_wall::get_C_t(const double &in_tau_w) const
{
  const double temp = C_basal - T_S0 * ( in_tau_w / tau_wh - 1.0 );
  if(temp >= 0.0) return temp;
  else return 0.0;
}


double Model_wall::get_dC_t( const double &in_Q_M, const double &in_a_t) const
{
  return 12.0 * T_S0 * mu * in_Q_M 
    / (pi * tau_wh * in_a_t*in_a_t*in_a_t*in_a_t);
}


double Model_wall::get_T_act( const double &mmass, 
    const double &inLm_act,
    const double &J, const double &in_C_t ) const
{
  const double temp1 = T_M * mmass / (rho_s * J);
  const double temp2 = 1.0 - exp(-1.0 * in_C_t * in_C_t);
  const double temp3 = (Lambda_M - inLm_act) / (Lambda_M - Lambda_0);
  const double temp = temp1 * temp2 * inLm_act * (1.0 - temp3*temp3); 

  if(temp < 0.0) return 0.0;
  else return temp;
}


double Model_wall::get_dT_act( const double &mmass, const double &inLm_act,
    const double &in_a_t, const double &L_z, const double in_a_act,
    const double &J, const double &in_C_t,
    const double &in_dC_t ) const
{
  const double dTact_1 = T_M * mmass / (rho_s * J);
  const double dTact_2 = 1.0 - exp(-1.0 * in_C_t * in_C_t);
  const double dTact_3 = (Lambda_M - inLm_act) / (Lambda_M - Lambda_0);
  const double dTact_4 = 1.0 - dTact_3 * dTact_3;

  return dTact_1 * ((-1.0 * L_z)/(J * a_M)) * inLm_act * dTact_2 * dTact_4
    + dTact_1 * dTact_2 * dTact_4 / in_a_act
    + dTact_1 * inLm_act * (1.0-dTact_2) * 2.0 * in_C_t * in_dC_t * dTact_4
    + dTact_1 * inLm_act * dTact_2 * 2.0 * dTact_3 
    / ((Lambda_M - Lambda_0) * in_a_act);
}


double Model_wall::l2error_a( const double &in_a, const int &tstep ) const
{
  const double a = in_a;
  const double b = Da[tstep];
  return sqrt((a-b)*(a-b)/(b*b));
}



void Model_wall::update_m_c( const int &tstep, 
    const double &Lt, const double &Lz,
    const double &dwdLt, const double &dwdLz, const double &Mc,
    const double &C_t, double &error, double &error_bottom )
{
  const double T_t_c = (rho_s * Lt * dwdLt) / Mc;
  const double T_z_c = (rho_s * Lz * dwdLz) / Mc;
  const double alpha_ck[4] = {0.0, 0.5*pi, 
    Dalpha[tstep], 2.0*pi - Dalpha[tstep]};

  double sigma_ck[4];
  SYS_T::get_Lk(sigma_ck, T_t_c, T_z_c, alpha_ck, 4);

  double nc_stress[4];
  for(int ii=0; ii<4; ++ii) nc_stress[ii] = (sigma_ck[ii] / sigma_ch) - 1.0;

  const double dn_C = (C_t / C_basal) - 1.0;

  double m_c[4];

  for(int ii=0; ii<4; ++ii)
  {
    m_c[ii] = m_basal_ck[ii] * (K_c1 * nc_stress[ii] + K_c2 * dn_C + 1.0);
  
    if(m_c[ii]<0.0) m_c[ii] = 0.0;
  }

  error = 0.0; error_bottom = 0.0;
  for(int ii=0; ii<4; ++ii)
  {
    error += (m_c[ii] - Dm_c[tstep][ii]) * (m_c[ii] - Dm_c[tstep][ii]);
    error_bottom += m_c[ii] * m_c[ii];

    Dm_c[tstep][ii] = m_c[ii];
  }
}


void Model_wall::update_m_m( const int &tstep, const double &Lt, 
    const double &Lz, const double &dwdLt, const double &Tact, 
    const double &Mm, const double &C_t,
    double &error, double &error_bottom )
{
  const double T_t_m = (rho_s * Lt * Lz / Mm) * ((dwdLt/Lz) + Tact);

  const double nm_stress = (T_t_m / sigma_mh) - 1.0;

  const double dn_C = (C_t / C_basal) - 1.0;

  double m_m = m_basal_m * (K_m1 * nm_stress + K_m2 * dn_C + 1.0);

  if(m_m < 0.0) m_m = 0.0;

  error = (m_m - Dm_m[tstep]) * (m_m - Dm_m[tstep]);
  error_bottom = m_m * m_m;

  Dm_m[tstep] = m_m;
}
// EOF
