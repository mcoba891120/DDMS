! Copyright 2011-2021 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program. If not, see <http://www.gnu.org/licenses/>.
!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief  phenomenological crystal plasticity formulation using a powerlaw fitting
!--------------------------------------------------------------------------------------------------
submodule(phase:plastic) phenopowerlaw

  type :: tParameters
    character(len=10) :: &
      kinetics 
    real(pReal) :: &
      dot_gamma_0_sl = 1.0_pReal, &                                                                 !< reference shear strain rate for slip
      n_sl           = 1.0_pReal, &                                                                 !< stress exponent for slip
      c_1            = 1.0_pReal, &
      c_2            = 1.0_pReal, &
      c_3            = 1.0_pReal, &
      c_4            = 1.0_pReal, &
      h_0_sl_sl      = 1.0_pReal, &                                                                 !< reference hardening slip - slip
      a_sl           = 1.0_pReal, &
      !!! global params
      T              = 1.0_pReal, &
      rate           = 1.0_pReal, &
      rl             = 1.0_pReal, &
      thickness      = 1.0_pReal, &
      V                         , &
      !!! c_G
      Tm             = 1.0_pReal, &
      theta          = 1.0_pReal, &
      u0             = 1.0_pReal, &
      !!! c_prec_6111
      b              = 1.0_pReal, &
      M              = 1.0_pReal, &
      rtrans         = 1.0_pReal, &
      rcl            = 1.0_pReal, &
      beta           = 1.0_pReal, &
      !!! c_gb
      grainSize      = 1.0_pReal, &
      ky             = 1.0_pReal, &
      !!! c_YS_temper
      c1             = 1.0_pReal, &
      RR             = 1.0_pReal, &
      DG             = 1.0_pReal, &
      e0             = 1.0_pReal, &
      q              = 1.0_pReal, &
      p              = 1.0_pReal, &
      n_             = 1.0_pReal, &
      !!! output
      f_o                       , &
      No_mean                   , &
      !!! hardening lwa: ssd_v1 + gnd_v2
      alpha          = 1.0_pReal, &
      G              = 1.0_pReal, &
      !!! c_ssd_v1
      k1             = 1.0_pReal, &
      k2                        , &
      k3             = 1.0_pReal, &
      k20            = 1.0_pReal, &
      Crmg           = 1.0_pReal, &
      Zs             = 1.0_pReal, &
      m0             = 1.0_pReal, &
      !!! c_gnd_v1
      k1g            = 1.0_pReal, &
      k2g                       , &
      k2g0           = 1.0_pReal, &
      fr_o           = 1.0_pReal, &
      Zg             = 1.0_pReal, &
      phi                       , &
      lmb                       , &
      lmb_star                  , &
      !!! c_gnd_v2
      ksat           = 1.0_pReal, &
      ko             = 1.0_pReal, &
      rhog_sat                  , &
      a_gnd                     , &
      a_n                       , &
      !!! precipitation kinetics region start
      Cp             = 1.0_pReal, &
      A0             = 1.0_pReal, &
      j0             = 1.0_pReal, &
      Qd             = 1.0_pReal, &
      D0             = 1.0_pReal, &
      Cs             = 1.0_pReal, &
      Qs             = 1.0_pReal, &
      sigma          = 1.0_pReal, &
      Vm             = 1.0_pReal, &
      r_min          = 1.0_pReal, &
      r_max          = 1.0_pReal, &
      C0             = 1.0_pReal, &
      D                         , &
      Ce                        , &
      A                         , &
      aTol_rho       = 1.0_pReal, &
      numerical_inc  = 1e-10_pReal
      !!! precipitation kinetics region end
    real(pReal),               allocatable, dimension(:) :: &
      xi_inf_sl, &                                                                                  !< maximum critical shear stress for slip
      h_int, &                                                                                      !< per family hardening activity (optional)
      tem_areas, &
      !!! c_ss_6111
      cc, &
      wt, &
      aw, &
      rho_ssd_0, &
      rho_gnd_0, &
      r_i, Ci, &
      tau_slip_cur
    real(pReal),               allocatable, dimension(:,:) :: &
      h_sl_sl                                                                                    !< slip resistance from slip activity
    real(pReal),               allocatable, dimension(:,:,:) :: &
      P_sl, &
      nonSchmid_pos, &
      nonSchmid_neg
    integer :: &
      sum_N_sl, &                                                                                   !< total number of active slip system
      classes
    logical :: &
      nonSchmidActive = .false.
    character(len=pStringLen), allocatable, dimension(:) :: &
      output
  end type tParameters

  type :: tPhenopowerlawState
    real(pReal), pointer, dimension(:,:) :: &
      xi_slip, &
      gamma_slip, &
      rho_ssd, &
      rho_gnd, &
      Nv_i, &
      f_all, &
      Req_mean, &
      Cbar, &
      pp, &
      ss, &
      gb, &
      crss
  end type tPhenopowerlawState

!--------------------------------------------------------------------------------------------------
! containers for parameters and state
  type(tParameters),         allocatable, dimension(:) :: param
  type(tPhenopowerlawState), allocatable, dimension(:) :: &
    dotState, &
    state, &
    deltaState

contains

!--------------------------------------------------------------------------------------------------
!> @brief Perform module initialization.
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function plastic_phenopowerlaw_init() result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: &
    ph, i, o, s, &
    Nmembers, &
    sizeState, sizeDotState, sizeDeltaState, &
    startIndex, endIndex
  integer,     dimension(:), allocatable :: &
    N_sl
  real(pReal), dimension(:), allocatable :: &
    xi_0_sl, &                                                                                      !< initial critical shear stress for slip
    a                                                                                               !< non-Schmid coefficients
  character(len=pStringLen) :: &
    extmsg = ''
  class(tNode), pointer :: &
    phases, &
    phase, &
    mech, &
    pl


  myPlasticity = plastic_active('phenopowerlaw')
  if(count(myPlasticity) == 0) return

  print'(/,a)', ' <<<+-  phase:mechanical:plastic:phenopowerlaw_MOD init  -+>>>'
  print'(a,i0)', ' # phases: ',count(myPlasticity); flush(IO_STDOUT)

  phases => config_material%get('phase')
  allocate(param(phases%length))
  allocate(state(phases%length))
  allocate(dotState(phases%length))
  allocate(deltaState(phases%length))

  do ph = 1, phases%length
    if(.not. myPlasticity(ph)) cycle

    associate(prm => param(ph), dot => dotState(ph), stt => state(ph), dlt => deltaState(ph))

    phase => phases%get(ph)
    mech  => phase%get('mechanical')
    pl  => mech%get('plastic')

!--------------------------------------------------------------------------------------------------
! slip related parameters
    N_sl         = pl%get_as1dInt('N_sl',defaultVal=emptyIntArray)
    prm%sum_N_sl = sum(abs(N_sl))
    slipActive: if (prm%sum_N_sl > 0) then
      prm%P_sl = lattice_SchmidMatrix_slip(N_sl,phase%get_asString('lattice'),&
                                           phase%get_asFloat('c/a',defaultVal=0.0_pReal))

      if(phase%get_asString('lattice') == 'cI') then
        a = pl%get_as1dFloat('a_nonSchmid',defaultVal=emptyRealArray)
        if(size(a) > 0) prm%nonSchmidActive = .true.
        prm%nonSchmid_pos  = lattice_nonSchmidMatrix(N_sl,a,+1)
        prm%nonSchmid_neg  = lattice_nonSchmidMatrix(N_sl,a,-1)
      else
        prm%nonSchmid_pos  = prm%P_sl
        prm%nonSchmid_neg  = prm%P_sl
      endif
      prm%h_sl_sl   = lattice_interaction_SlipBySlip(N_sl, &
                                                     pl%get_as1dFloat('h_sl-sl'), &
                                                     phase%get_asString('lattice'))

      xi_0_sl             = pl%get_as1dFloat('xi_0_sl',   requiredSize=size(N_sl))
      prm%xi_inf_sl       = pl%get_as1dFloat('xi_inf_sl', requiredSize=size(N_sl))
      prm%h_int           = pl%get_as1dFloat('h_int',     requiredSize=size(N_sl), &
                                            defaultVal=[(0.0_pReal,i=1,size(N_sl))])
      !!! rho_Nslip region start
      prm%tem_areas       = pl%get_as1dFloat('tem_areas')
      prm%rho_ssd_0       = pl%get_as1dFloat('rho_ssd_0', requiredSize=size(N_sl))
      prm%rho_gnd_0       = pl%get_as1dFloat('rho_gnd_0', requiredSize=size(N_sl))
      !!! rho_Nslip region end
      prm%dot_gamma_0_sl  = pl%get_asFloat('dot_gamma_0_sl')
      prm%n_sl            = pl%get_asFloat('n_sl')
      prm%a_sl            = pl%get_asFloat('a_sl')
      prm%h_0_sl_sl       = pl%get_asFloat('h_0_sl-sl')

      !!! global params
      prm%kinetics        = pl%get_asString('kinetics')
      prm%T               = pl%get_asFloat('T')
      prm%rate            = pl%get_asFloat('rate')
      prm%rl              = pl%get_asFloat('rl')
      prm%thickness       = pl%get_asFloat('thickness')
      prm%V               = (prm%thickness+prm%rl)*sum(prm%tem_areas)
      !!! c_G
      prm%Tm              = pl%get_asFloat('Tm')
      prm%theta           = pl%get_asFloat('theta')
      prm%u0              = pl%get_asFloat('u0')
      !!! c_prec_6111
      prm%b               = pl%get_asFloat('b')
      prm%M               = pl%get_asFloat('M')
      prm%rtrans          = pl%get_asFloat('rtrans')
      prm%rcl             = pl%get_asFloat('rcl')
      prm%beta            = pl%get_asFloat('beta')
      !!! c_ss_6111
      prm%wt              = pl%get_as1dFloat('wt')
      prm%aw              = pl%get_as1dFloat('aw')
      prm%cc              = [0.0, 0.0]
      !!! c_gb
      prm%grainSize       = pl%get_asFloat('grainSize')
      prm%ky              = pl%get_asFloat('ky')
      !!! c_YS_temper
      prm%c1              = pl%get_asFloat('c1')
      prm%RR              = pl%get_asFloat('RR')
      prm%DG              = pl%get_asFloat('DG')
      prm%e0              = pl%get_asFloat('e0')
      prm%q               = pl%get_asFloat('q')
      prm%p               = pl%get_asFloat('p')
      prm%n_              = pl%get_asFloat('n_')
      !!! hardening lwa: ssd_v1 + gnd_v2
      prm%alpha           = pl%get_asFloat('alpha')
      !!! c_ssd_v1
      prm%k1              = pl%get_asFloat('k1')
      prm%k3              = pl%get_asFloat('k3')
      prm%k20             = pl%get_asFloat('k20')
      prm%Crmg            = pl%get_asFloat('Crmg')
      prm%Zs              = pl%get_asFloat('Zs')
      prm%m0              = pl%get_asFloat('m0')
      !!! c_gnd_v1
      prm%k1g             = pl%get_asFloat('k1g')
      prm%k2g0            = pl%get_asFloat('k2g0')
      prm%fr_o            = pl%get_asFloat('fr_o')
      prm%Zg              = pl%get_asFloat('Zg')
      !!! c_gnd_v2
      prm%ksat            = pl%get_asFloat('ksat')
      prm%ko              = pl%get_asFloat('ko')
      prm%a_n             = pl%get_asFloat('a_n')
      prm%a_gnd           = pl%get_asFloat('a_gnd', defaultVal=-999.0_pReal)
      !!! precipitation kinetics region start
      prm%Cp              = pl%get_asFloat('Cp')
      prm%A0              = pl%get_asFloat('A0')
      prm%j0              = pl%get_asFloat('j0')
      prm%Qd              = pl%get_asFloat('Qd')
      prm%D0              = pl%get_asFloat('D0')
      prm%Cs              = pl%get_asFloat('Cs')
      prm%Qs              = pl%get_asFloat('Qs')
      prm%sigma           = pl%get_asFloat('sigma')
      prm%Vm              = pl%get_asFloat('Vm')
      prm%r_min           = pl%get_asFloat('r_min')
      prm%r_max           = pl%get_asFloat('r_max')
      prm%C0              = pl%get_asFloat('C0')
      prm%D               = prm%D0*exp(-prm%Qd/(prm%RR*prm%T))
      prm%Ce              = prm%Cs*exp(-prm%Qs/(prm%RR*prm%T))
      prm%A               = 0.5*prm%rl/pl%get_asFloat('Req_mean')
      ! prm%aTol_rho        = pl%get_asFloat('aTol_rho')
      prm%numerical_inc   = pl%get_asFloat('numerical_inc')
      prm%classes         = pl%get_asInt('classes')
      prm%r_i              = linspace(prm%r_min, prm%r_max, prm%classes)
      prm%Ci              = prm%Ce*exp(2*prm%sigma*prm%Vm/(prm%r_i*prm%RR*prm%T))
      !!! precipitation kinetics region end

      ! expand: family => system
      xi_0_sl             = math_expand(xi_0_sl,      N_sl)
      prm%xi_inf_sl       = math_expand(prm%xi_inf_sl,N_sl)
      prm%h_int           = math_expand(prm%h_int,    N_sl)
      !!! rho_Nslip region start 
      prm%rho_ssd_0       = math_expand(prm%rho_ssd_0,N_sl)
      prm%rho_gnd_0       = math_expand(prm%rho_gnd_0,N_sl)
      !!! rho_Nslip region end

      ! sanity checks
      if (    prm%dot_gamma_0_sl  <= 0.0_pReal)      extmsg = trim(extmsg)//' dot_gamma_0_sl'
      if (    prm%a_sl            <= 0.0_pReal)      extmsg = trim(extmsg)//' a_sl'
      if (    prm%n_sl            <= 0.0_pReal)      extmsg = trim(extmsg)//' n_sl'
      if (any(xi_0_sl             <= 0.0_pReal))     extmsg = trim(extmsg)//' xi_0_sl'
      if (any(prm%xi_inf_sl       <= 0.0_pReal))     extmsg = trim(extmsg)//' xi_inf_sl'

    else slipActive
      xi_0_sl = emptyRealArray
      allocate(prm%xi_inf_sl,prm%h_int,source=emptyRealArray)
      allocate(prm%h_sl_sl(0,0))
    endif slipActive

!--------------------------------------------------------------------------------------------------
!  output pararameters

#if defined (__GFORTRAN__)
    prm%output = output_as1dString(pl)
#else
    prm%output = pl%get_as1dString('output',defaultVal=emptyStringArray)
#endif

!--------------------------------------------------------------------------------------------------
! allocate state arrays
    Nmembers = count(material_phaseID == ph)
    sizeDotState = size(['xi_sl   ','gamma_sl']) * prm%sum_N_sl &
                 + size(['rho_ssd ','rho_gnd ']) * prm%sum_N_sl        !!! rho_Nslip region start/end
    !!! precipitation kinetics region start/end
    sizeDeltaState = size(['Nv_i    ']) * prm%classes &
                   + size(['xi_sl   ']) &
                   + size(['f_all   ','Req_mean','Cbar    ']) &
                   + size(['pp      ','ss      ','gb      ']) &
                   + size(['crss    ']) 
    !!! precipitation kinetics region start/end

    sizeState = sizeDotState+sizeDeltaState

    call phase_allocateState(plasticState(ph),Nmembers,sizeState,sizeDotState,sizeDeltaState)

!--------------------------------------------------------------------------------------------------
! state aliases and initialization
!!! Nmembers == number of grid points
!!! shape(stt%xi_slip) == (Nslip, number of grid points)
    startIndex = 1
    endIndex   = prm%sum_N_sl
    stt%xi_slip => plasticState(ph)%state   (startIndex:endIndex,:)
    stt%xi_slip =  spread(xi_0_sl, 2, Nmembers)
    dot%xi_slip => plasticState(ph)%dotState(startIndex:endIndex,:)
    dlt%xi_slip => plasticState(ph)%deltaState(startIndex:endIndex,:)
    plasticState(ph)%atol(startIndex:endIndex) = pl%get_asFloat('atol_xi',defaultVal=1.0_pReal)
    if(any(plasticState(ph)%atol(startIndex:endIndex) < 0.0_pReal)) extmsg = trim(extmsg)//' atol_xi'

    startIndex = endIndex + 1
    endIndex   = endIndex + prm%sum_N_sl
    stt%gamma_slip => plasticState(ph)%state   (startIndex:endIndex,:)
    dot%gamma_slip => plasticState(ph)%dotState(startIndex:endIndex,:)
    plasticState(ph)%atol(startIndex:endIndex) = pl%get_asFloat('atol_gamma',defaultVal=1.0e-6_pReal)
    if(any(plasticState(ph)%atol(startIndex:endIndex) < 0.0_pReal)) extmsg = trim(extmsg)//' atol_gamma'

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + prm%sum_N_sl
    stt%rho_ssd=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%rho_ssd= spread(prm%rho_ssd_0,2,Nmembers)
    dot%rho_ssd=>plasticState(ph)%dotState(startIndex:endIndex,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho 

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + prm%sum_N_sl
    stt%rho_gnd=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%rho_gnd= spread(prm%rho_gnd_0,2,Nmembers)
    dot%rho_gnd=>plasticState(ph)%dotState(startIndex:endIndex,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

!--------------------------------------------------------------------------------------------------
    o = plasticState(ph)%offsetDeltaState
    s = prm%sum_N_sl    ! offset for dlt%%xi_slip

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + prm%classes
    stt%Nv_i=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%Nv_i= spread(pl%get_as1dFloat('Nv_i'),2,Nmembers)
    dlt%Nv_i=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%f_all=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%f_all= pl%get_asFloat('f_all')
    dlt%f_all=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%Req_mean=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%Req_mean= pl%get_asFloat('Req_mean')
    dlt%Req_mean=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%pp=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%pp = 0.0_pReal
    dlt%pp=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%ss=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%ss = 0.0_pReal
    dlt%ss=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%gb=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%gb = 0.0_pReal
    dlt%gb=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%crss=>plasticState(ph)%state(startIndex:endIndex,:)
    stt%crss = xi_0_sl(1)
    dlt%crss=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    startIndex = endIndex + 1_pInt
    endIndex   = endIndex + 1_pInt
    stt%Cbar=>plasticState(ph)%state(startIndex:endIndex,:)
    call c_ss_6111(ph,pl%get_asFloat('f_all'),prm%wt,prm%aw, &
                   prm%cc,stt%ss(1,1))
    stt%Cbar = prm%cc(1)
    dlt%Cbar=>plasticState(ph)%deltaState(startIndex-o+s:endIndex-o+s,:)
    plasticState(ph)%atol(startIndex:endIndex) = prm%aTol_rho

    call c_TEM(ph,prm%rtrans,prm%rcl,prm%A,stt%Nv_i(:,1),prm%r_i,stt%Req_mean(1,1),prm%V, &
               prm%f_o,prm%No_mean,prm%phi,prm%lmb,prm%lmb_star)
    ! >-------------------------
    ! ! > AA7075 17hr_298K
    ! prm%f_o = 0.0239
    ! >-------------------------
    call c_G(prm%T,prm%u0,prm%Tm,prm%theta, &
             prm%G)
    ! call c_prec_6111_avg(prm%rtrans,prm%b,stt%f_all(1,1),stt%Req_mean(1,1),prm%G,prm%beta,prm%M, &
    !                  stt%pp(1,1))
    call c_prec_6111(ph,prm%rtrans,prm%b,stt%f_all(1,1),stt%Req_mean(1,1),prm%G,prm%beta,prm%M,prm%r_i,stt%Nv_i(:,1), &
                     stt%pp(1,1))
    call c_ss_6111(ph,stt%f_all(1,1),prm%wt,prm%aw, &
                   prm%cc,stt%ss(1,1))
    call c_gb(prm%grainSize,prm%ky, &
              stt%gb(1,1))
    call c_YS_temper(ph,prm%T,prm%rate,prm%u0,stt%pp(1,1),stt%ss(1,1),stt%gb(1,1), &
                     prm%G,prm%c1,prm%RR,prm%DG,prm%e0,prm%M,prm%q,prm%p, &
                     stt%crss(1,1))
    call c_ssd_v1(ph,prm%T,prm%b,prm%rate,prm%u0,prm%RR,prm%wt,prm%cc,prm%G,prm%M, &
                  prm%k1,prm%k3,prm%Zs,prm%m0,prm%alpha,prm%k20,prm%Crmg, &
                  prm%k2)
    call c_gnd_v1(prm%T,prm%rate,prm%RR,prm%k2g0,prm%f_o,prm%fr_o,prm%Zg,prm%m0, &
                  prm%k2g)
    ! call c_gnd_v2(prm%b,prm%No_mean,prm%lmb,prm%f_o,prm%ko,prm%ksat,prm%a_n, &
    !               prm%a_gnd,prm%rhog_sat)

    ! spread to all fourier points
    stt%pp(1,:) = stt%pp(1,1)
    stt%ss(1,:) = stt%ss(1,1)
    stt%gb(1,:) = stt%gb(1,1)
    stt%crss(1,:) = stt%crss(1,1)    

    print '(/,a,a,a)', ' <<<+-  YS/WH model:', prm%kinetics, ' init  -+>>>'
    print '(a15,f10.2,a5)', 'sigma_p: ' , stt%pp(1,1)*1e-6            , 'MPa'
    print '(a15,f10.2,a5)', 'sigma_ss: ', stt%ss(1,1)*1e-6            , 'MPa'
    print '(a15,f10.2,a5)', 'sigma_gb: ', stt%gb(1,1)*1e-6            , 'MPa'
    print '(a15,f10.2,a5)', 'crss: '    , stt%crss(1,1)*1e-6          , 'MPa'
    print '(a15,f10.2,a5)', 'YS: '      , stt%crss(1,1)*prm%M*1e-6, 'MPa'
    print '(a15,f10.2,a5)', 'f_all: '   , stt%f_all(1,1)*100.0        , '%'
    print '(a15,f10.2,a5)', 'f_o: '     , prm%f_o*100.0               , '%'
    print '(a15,f10.2,a5)', 'Nv: '      , sum(stt%Nv_i(:,1))/1e18     , 'um-3'
    print '(a15,f10.2,a5)', 'No_mean: ' , prm%No_mean                 , '-'
    print '(a15,f10.2,a5)', 'phi: '     , prm%phi                     , '-'
    print '(a15,f10.2,a5)', 'lmb: '     , prm%lmb*1e9                 , 'nm'
    print '(a15,f10.2,a5)', 'lmb_star: ', prm%lmb_star*1e9            , 'nm'
    print '(a15,f10.2,a5)', 'k2: '      , prm%k2                      , '-'
    print '(a15,f10.2,a5)', 'k2g: '     , prm%k2g                     , '-'
    print '(a15,f10.4,a5)', 'a_gnd: '   , prm%a_gnd                   , '-'
    print '(a15,e10.2,a5)', 'rhog_sat: ', prm%rhog_sat                , 'm-2'
    end associate

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
    if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg)//'(phenopowerlaw)')

  enddo
end function plastic_phenopowerlaw_init


!--------------------------------------------------------------------------------------------------
!> @brief Calculate plastic velocity gradient and its tangent.
!> @details asummes that deformation by dislocation glide affects twinned and untwinned volume
!  equally (Taylor assumption). Twinning happens only in untwinned volume
!--------------------------------------------------------------------------------------------------
pure module subroutine phenopowerlaw_LpAndItsTangent(Lp,dLp_dMp,Mp,ph,en)

  real(pReal), dimension(3,3),     intent(out) :: &
    Lp                                                                                              !< plastic velocity gradient
  real(pReal), dimension(3,3,3,3), intent(out) :: &
    dLp_dMp                                                                                         !< derivative of Lp with respect to the Mandel stress

  real(pReal), dimension(3,3), intent(in) :: &
    Mp                                                                                              !< Mandel stress
  integer,               intent(in) :: &
    ph, &
    en

  integer :: &
    i,k,l,m,n
  real(pReal), dimension(param(ph)%sum_N_sl) :: &
    tau_slip_pos, tau_slip_neg, &   ! region start/end
    gdot_slip_pos,gdot_slip_neg, &
    dgdot_dtauslip_pos,dgdot_dtauslip_neg

  Lp = 0.0_pReal
  dLp_dMp = 0.0_pReal

  associate(prm => param(ph))

  call kinetics_slip(Mp,ph,en,tau_slip_pos,tau_slip_neg,gdot_slip_pos,gdot_slip_neg,dgdot_dtauslip_pos,dgdot_dtauslip_neg)
  slipSystems: do i = 1, prm%sum_N_sl
    Lp = Lp + (gdot_slip_pos(i)+gdot_slip_neg(i))*prm%P_sl(1:3,1:3,i)
    forall (k=1:3,l=1:3,m=1:3,n=1:3) &
      dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                       + dgdot_dtauslip_pos(i) * prm%P_sl(k,l,i) * prm%nonSchmid_pos(m,n,i) &
                       + dgdot_dtauslip_neg(i) * prm%P_sl(k,l,i) * prm%nonSchmid_neg(m,n,i)
  enddo slipSystems

  end associate

end subroutine phenopowerlaw_LpAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief Calculate the rate of change of microstructure.
!--------------------------------------------------------------------------------------------------
module subroutine phenopowerlaw_dotState(Mp,ph,en,Cauchy)

  real(pReal), dimension(3,3),  intent(in) :: &
    Mp,Cauchy                                                            !< Mandel stress
  integer,                      intent(in) :: &
    ph, &
    en
  integer :: i
  real(pReal) :: VM
  real(pReal), dimension(param(ph)%sum_N_sl) :: &
    tau_slip_pos, tau_slip_neg, &   ! region start/end
    gdot_slip_pos,gdot_slip_neg, &
    dtau_drho, &
    drho_ssd_deps, &
    drho_gnd_deps, &
    h, tau_p, tau_d, exponent

  associate(prm => param(ph), stt => state(ph), &
  dot => dotState(ph))

!--------------------------------------------------------------------------------------------------
! shear rates
  call kinetics_slip(Mp,ph,en,tau_slip_pos,tau_slip_neg,gdot_slip_pos,gdot_slip_neg)
  dot%gamma_slip(:,en) = abs(gdot_slip_pos+gdot_slip_neg)
!--------------------------------------------------------------------------------------------------
! hardening
  !!! rho_Nslip region start
  ! calculate slip resistances on N slip systems, gdot
  ! shape(prm%h_sl_sl) == (Nslip, Nslip)
  ! WARNING: whole program becoms much slower if having weird value in dotState
  ! WARNING: crystallite/integrateStateFPI() will run much more iters

  dtau_drho     = prm%alpha*prm%G*prm%b / (2.0*sqrt(stt%rho_ssd(:,en)+stt%rho_gnd(:,en)))            ! shape == (12,1)
  drho_ssd_deps = prm%k1*sqrt(stt%rho_ssd(:,en))-prm%k2*stt%rho_ssd(:,en)                          ! shape == (12,1)
  drho_gnd_deps = prm%k1g/prm%lmb_star - prm%k2g*stt%rho_gnd(:,en)                                    ! c_gnd_v1
  ! drho_gnd_deps = (1-(stt%rho_gnd(:,en)/prm%rhog_sat)**prm%a_gnd) / (prm%b*prm%ko*prm%lmb)    ! shape == (12,1), c_gnd_v2
  tau_p = stt%pp(1,en)/prm%M
  tau_d = abs(tau_slip_pos)-stt%crss(1,en)
  exponent = merge((tau_p**prm%n_+tau_d**prm%n_)**(1.0/prm%n_-1.0)*tau_d**(prm%n_-1.0), &
                    1.0, tau_d>0.0)
  h = exponent*dtau_drho*(drho_ssd_deps+drho_gnd_deps)

  dot%rho_ssd(:,en) = dot%gamma_slip(:,en)*drho_ssd_deps
  dot%rho_gnd(:,en) = dot%gamma_slip(:,en)*drho_gnd_deps
  dot%xi_slip(:,en) = matmul(prm%h_sl_sl,dot%gamma_slip(:,en)*h)

  ! if (en==1) then
  !   print *, '##########'
  !   print *, 'tau_slip_pos'
  !   print *, tau_slip_pos
  !   print *, 'exponent'
  !   print *, exponent
  !   print *, 'h'
  !   print *, h
  ! endif

  end associate

end subroutine phenopowerlaw_dotState

!--------------------------------------------------------------------------------------------------
!> @brief Calculate (instantaneous) incremental change of microstructure.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_phenopowerlaw_deltaState(Mp,ph,en,Delta_t,Cauchy)
  real(pReal), dimension(3,3),  intent(in) :: &
    Mp,Cauchy                                                                                              !< Mandel stress
  integer,                      intent(in) :: &
    ph, &
    en
  real(pReal), intent(in) :: &
    Delta_t
  real(pReal) :: &
    G_het, j, rstar, Delta_r, ap0, crss_old, dcrss, VM
  real(pReal), dimension(param(ph)%classes) :: &
    v, vw, ve, ae, aw, ap, Np, Ne, Nw
  logical, dimension(param(ph)%classes) :: &
    mask

  associate(prm => param(ph), stt => state(ph), dlt => deltaState(ph))

  dlt%crss(1,en) = 0.0  ! stable or if Delta_t==0
  nonstable: if (prm%kinetics=='nonstable') then
  !--------------------------------------------------------------------------------------------------
    precipitationKinetics: if (Delta_t>0.0_pReal) then
      ! >--------------------------------------------------
      ! nucleation
      G_het = prm%A0**3/(prm%RR*prm%T*log(stt%Cbar(1,en)/prm%Ce))**2
      j = prm%j0*exp(-G_het/(prm%RR*prm%T))*exp(-prm%Qd/(prm%RR*prm%T))
      rstar = 2*prm%sigma*prm%Vm/(prm%RR*prm%T)/log(stt%Cbar(1,en)/prm%Ce)
      mask = prm%r_i<=(rstar+0.05*rstar)
      if (any(mask)) then 
        stt%Nv_i(count(mask),en) = stt%Nv_i(count(mask),en) + j*Delta_t
      end if
      
      ! >--------------------------------------------------
      ! mass balance
      stt%Req_mean(1,en) = dot_product(stt%Nv_i(:,en),prm%r_i)/sum(stt%Nv_i(:,en))
      ! stt%f_all(1,en) = sum(atan(1.0_pReal)*4*prm%r_i**2*2*prm%A*stt%Req_mean(1,en)*stt%Nv_i(:,en))
      stt%f_all(1,en) = sum(4.0/3.0*atan(1.0)*4.0*prm%r_i**3*stt%Nv_i(:,en))                ! spherical
      ! stt%Cbar(1,en) = max(prm%Ce+prm%numerical_inc, & 
      !                      prm%C0-(prm%Cp-stt%Cbar(1,en))*stt%f_all(1,en))
      stt%Cbar(1,en) = max(prm%Ce+prm%numerical_inc, & 
                           (prm%C0-prm%Cp*stt%f_all(1,en))/(1-stt%f_all(1,en)))

      ! >--------------------------------------------------
      ! growth rate
      ! vw = (stt%Cbar(1,en)-prm%Ci)/(prm%Cp-prm%Ci)*prm%D/prm%r_i
      ! ve = [vw(2:), prm%numerical_inc]
      v = (stt%Cbar(1,en)-prm%Ci)/(prm%Cp-prm%Ci)*prm%D/prm%r_i
      ve = [v(2:), prm%numerical_inc]
      vw = v

      ! >--------------------------------------------------
      ! udpate dot%Nv_i
      Delta_r = (prm%r_max-prm%r_min)/prm%classes
      ap0 = Delta_r/Delta_t
      ae = 0.0_pReal
      aw = 0.0_pReal
      ap = 0.0_pReal

      where (ve>0 .and. vw>0)
        aw = vw 
        ap = ap0 + ve 
      else where (ve>0 .and. vw<0)
        ap = ap0 + ve - vw
      else where (ve<0 .and. vw>0)
        ae = -ve 
        aw = vw
        ap = ap0
      else where (ve<0 .and. vw<0)
        ae = -ve 
        ap = ap0 - vw
      end where

      Np = stt%Nv_i(:,en)
      Ne = [stt%Nv_i(2:,en), 0.0_pReal]
      Nw = [0.0_pReal, stt%Nv_i(:prm%classes-1,en)]
      stt%Nv_i(:,en) = (ae*Ne + aw*Nw + ap0*Np)/ap

      ! update parameters in YS/WH model
      ! record YS slip resistance
      crss_old = stt%crss(1,en)

      call c_TEM(ph,prm%rtrans,prm%rcl,prm%A,stt%Nv_i(:,en),prm%r_i,stt%Req_mean(1,en),prm%V, &
                prm%f_o,prm%No_mean,prm%phi,prm%lmb,prm%lmb_star)
      call c_G(prm%T,prm%u0,prm%Tm,prm%theta, &
              prm%G)
      ! call c_prec_6111_avg(prm%rtrans,prm%b,stt%f_all(1,1),stt%Req_mean(1,1),prm%G,prm%beta,prm%M, &
      !          stt%pp(1,1))
      call c_prec_6111(ph,prm%rtrans,prm%b,stt%f_all(1,en),stt%Req_mean(1,en),prm%G,prm%beta,prm%M,prm%r_i,stt%Nv_i(:,en), &
                      stt%pp(1,en))
      call c_ss_6111(ph,stt%f_all(1,en),prm%wt,prm%aw, &
                    prm%cc,stt%ss(1,en))
      call c_gb(prm%grainSize,prm%ky, &
                stt%gb(1,en))
      call c_YS_temper(ph,prm%T,prm%rate,prm%u0,stt%pp(1,en),stt%ss(1,en),stt%gb(1,en), &
                      prm%G,prm%c1,prm%RR,prm%DG,prm%e0,prm%M,prm%q,prm%p, &
                      stt%crss(1,en))
      call c_ssd_v1(ph,prm%T,prm%b,prm%rate,prm%u0,prm%RR,prm%wt,prm%cc,prm%G,prm%M, &
                    prm%k1,prm%k3,prm%Zs,prm%m0,prm%alpha,prm%k20,prm%Crmg, &
                    prm%k2)
      call c_gnd_v1(prm%T,prm%rate,prm%RR,prm%k2g0,prm%f_o,prm%fr_o,prm%Zg,prm%m0, &
                    prm%k2g)
      call c_gnd_v2(prm%b,prm%No_mean,prm%lmb,prm%f_o,prm%ko,prm%ksat,prm%a_n, &
                    prm%a_gnd,prm%rhog_sat)

      dcrss = stt%crss(1,en) - crss_old
      ! >--------------------------------------------------
      ! WARNING: consider update through dlt
      ! update YS slip resistance
      ! stt%xi_slip(:,en) = stt%xi_slip(:,en) + dcrss
      ! >--------------------------------------------------
      dlt%xi_slip(:,en) = dcrss
      ! >--------------------------------------------------

      if (en==1) then 
        print '(/,a)', ' <<<+-  YS/WH model report  -+>>>'
        print '(a15,f10.2,a5)', 'sigma_p: ' , stt%pp(1,1)*1e-6            , 'MPa'
        print '(a15,f10.2,a5)', 'sigma_ss: ', stt%ss(1,1)*1e-6            , 'MPa'
        print '(a15,f10.2,a5)', 'sigma_gb: ', stt%gb(1,1)*1e-6            , 'MPa'
        print '(a15,f10.2,a5)', 'crss: '    , stt%crss(1,1)*1e-6          , 'MPa'
        print '(a15,e10.2,a5)', 'dcrss: '   , dcrss*1e-6                  , 'MPa'
        print '(a15,f10.2,a5)', 'YS: '      , stt%crss(1,1)*prm%M*1e-6    , 'MPa'
        print '(a15,f10.2,a5)', 'f_all: '   , stt%f_all(1,1)*100.0        , '%'
        print '(a15,f10.2,a5)', 'f_o: '     , prm%f_o*100.0               , '%'
        print '(a15,f10.2,a5)', 'Nv: '      , sum(stt%Nv_i(:,1))/1e18     , 'um-3'
        print '(a15,f10.2,a5)', 'No_mean: ' , prm%No_mean                 , '-'
        print '(a15,f10.2,a5)', 'phi: '     , prm%phi                     , '-'
        print '(a15,f10.2,a5)', 'lmb: '     , prm%lmb*1e9                 , 'nm'
        print '(a15,f10.2,a5)', 'lmb_star: ', prm%lmb_star*1e9            , 'nm'
        print '(a15,f10.2,a5)', 'k2: '      , prm%k2                      , '-'
        print '(a15,f10.2,a5)', 'k2g: '     , prm%k2g                     , '-'
        print '(a15,f10.4,a5)', 'a_gnd: '   , prm%a_gnd                   , '-'
        print '(a15,e10.2,a5)', 'rhog_sat: ', prm%rhog_sat                , 'm-2'
      end if 

    end if precipitationKinetics
  end if nonstable
!--------------------------------------------------------------------------------------------------

  end associate
end subroutine plastic_phenopowerlaw_deltaState


!--------------------------------------------------------------------------------------------------
!> @brief Write results to HDF5 output file.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_phenopowerlaw_results(ph,group)

  integer,          intent(in) :: ph
  character(len=*), intent(in) :: group

  integer :: o

  associate(prm => param(ph), stt => state(ph))
  outputsLoop: do o = 1,size(prm%output)
    select case(trim(prm%output(o)))

      case('xi_sl')
        if(prm%sum_N_sl>0) call results_writeDataset(stt%xi_slip,group,trim(prm%output(o)), &
                                                     'resistance against plastic slip','Pa')
      case('gamma_sl')
        if(prm%sum_N_sl>0) call results_writeDataset(stt%gamma_slip,group,trim(prm%output(o)), &
                                                     'plastic shear','1')
      case('rho_ssd')
        call results_writeDataset(stt%rho_ssd,group,trim(prm%output(o)), &
                                                     'ssd','m-3')
      case('rho_gnd')
        call results_writeDataset(stt%rho_gnd,group,trim(prm%output(o)), &
                                                     'gnd','m-3')
      case('Nv_i')
        call results_writeDataset(stt%Nv_i,group,trim(prm%output(o)), &
                                                     'number density','m-3')
      case('Req_mean')
        call results_writeDataset(stt%Req_mean,group,trim(prm%output(o)), &
                                                     'mean radius','m')
      case('f_all')
        call results_writeDataset(stt%f_all,group,trim(prm%output(o)), &
                                                     'volume fraction','1')
      case('Cbar')
        call results_writeDataset(stt%Cbar,group,trim(prm%output(o)), &
                                                     'mean concentration of Mg','%')
      case('pp')
        call results_writeDataset(stt%pp,group,trim(prm%output(o)), &
                                                     'precipitation hardening','MPa')
      case('ss')
        call results_writeDataset(stt%ss,group,trim(prm%output(o)), &
                                                     'solid solution hardening','MPa')
      case('gb')
        call results_writeDataset(stt%gb,group,trim(prm%output(o)), &
                                                     'Hall-Petch hardening','MPa')
      case('crss')
        call results_writeDataset(stt%crss,group,trim(prm%output(o)), &
                                                     'YS slip resistance','MPa')

    end select
  enddo outputsLoop
  end associate

end subroutine plastic_phenopowerlaw_results


!--------------------------------------------------------------------------------------------------
!> @brief Calculate shear rates on slip systems and their derivatives with respect to resolved
!         stress.
!> @details Derivatives are calculated only optionally.
! NOTE: Against the common convention, the result (i.e. intent(out)) variables are the last to
! have the optional arguments at the end.
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics_slip(Mp,ph,en, &
                              tau_slip_pos, tau_slip_neg, & ! region start/end
                              gdot_slip_pos,gdot_slip_neg,dgdot_dtau_slip_pos,dgdot_dtau_slip_neg)

  real(pReal), dimension(3,3),  intent(in) :: &
    Mp                                                                                              !< Mandel stress
  integer,                      intent(in) :: &
    ph, &
    en

  real(pReal),                  intent(out), dimension(param(ph)%sum_N_sl) :: &
    gdot_slip_pos, &
    gdot_slip_neg, &
    ! region start
    tau_slip_pos, &
    tau_slip_neg
    ! region end
  real(pReal),                  intent(out), optional, dimension(param(ph)%sum_N_sl) :: &
    dgdot_dtau_slip_pos, &
    dgdot_dtau_slip_neg

  ! >------------------------------------------------------------------------------
  ! real(pReal), dimension(param(ph)%sum_N_sl) :: &
  !   tau_slip_pos, &
  !   tau_slip_neg
  ! >------------------------------------------------------------------------------
  integer :: i

  associate(prm => param(ph), stt => state(ph))

  do i = 1, prm%sum_N_sl
    tau_slip_pos(i) =       math_tensordot(Mp,prm%nonSchmid_pos(1:3,1:3,i))
    tau_slip_neg(i) = merge(math_tensordot(Mp,prm%nonSchmid_neg(1:3,1:3,i)), &
                            0.0_pReal, prm%nonSchmidActive)
  enddo

  ! >------------------------------------------------------------------------------
  ! where(dNeq0(tau_slip_pos))
  !   gdot_slip_pos = prm%dot_gamma_0_sl * merge(0.5_pReal,1.0_pReal, prm%nonSchmidActive) &          ! 1/2 if non-Schmid active
  !                 * sign(abs(tau_slip_pos/stt%xi_slip(:,en))**prm%n_sl,  tau_slip_pos)
  ! else where
  !   gdot_slip_pos = 0.0_pReal
  ! end where
  ! >------------------------------------------------------------------------------
  where(dNeq0(tau_slip_pos))
    gdot_slip_pos = prm%dot_gamma_0_sl * merge(0.5_pReal,1.0_pReal, prm%nonSchmidActive) &          ! 1/2 if non-Schmid active
                  * sign(abs(tau_slip_pos/stt%xi_slip(:,en))**prm%n_sl, &
                         tau_slip_pos)
  else where
    gdot_slip_pos = 0.0_pReal
  end where
  ! >------------------------------------------------------------------------------

  where(dNeq0(tau_slip_neg))
    gdot_slip_neg = prm%dot_gamma_0_sl * 0.5_pReal &                                                ! only used if non-Schmid active, always 1/2
                  * sign(abs(tau_slip_neg/stt%xi_slip(:,en))**prm%n_sl,  tau_slip_neg)
  else where
    gdot_slip_neg = 0.0_pReal
  end where

  if (present(dgdot_dtau_slip_pos)) then
    where(dNeq0(gdot_slip_pos))
      dgdot_dtau_slip_pos = gdot_slip_pos*prm%n_sl/tau_slip_pos
    else where
      dgdot_dtau_slip_pos = 0.0_pReal
    end where
  endif
  if (present(dgdot_dtau_slip_neg)) then
    where(dNeq0(gdot_slip_neg))
      dgdot_dtau_slip_neg = gdot_slip_neg*prm%n_sl/tau_slip_neg
    else where
      dgdot_dtau_slip_neg = 0.0_pReal
    end where
  endif
  end associate

end subroutine kinetics_slip

function linspace(from, to, points) result(array)
    real(pReal) :: from, to
    integer :: points, i
    real(pReal), dimension(points) :: array

    if (points == 0) return
    if (points == 1) then
        array(1) = from
        return
    end if

    do i=1, points
        array(i) = from+(to-from)*(i-1)/(points-1)
    end do
end function linspace

subroutine c_TEM(ph,rtrans,rcl,A,Nv_i,r_i,Req_mean,V,f_o,No_mean,phi,lmb,lmb_star)
  integer, intent(in) :: &
   ph
  real(pReal), intent(in) :: &
    rtrans,rcl,A,Req_mean,V
  real(pReal), dimension(param(ph)%classes), intent(in) :: &
    Nv_i,r_i
  real(pReal), intent(out) :: &
    f_o,No_mean,phi,lmb,lmb_star
  real(pReal), dimension(param(ph)%classes) :: &
    Nvo_i,phi_i

  where (r_i>rtrans)
    Nvo_i = Nv_i
  else where 
    Nvo_i = 0.0
  end where

  f_o = sum(4.0/3.0*atan(1.0)*4.0*r_i**3*Nvo_i)                ! spherical
  ! f_o = sum(atan(1.0)*4.0*r_i**2*2.0*A*Req_mean*Nvo_i)         ! needle

  No_mean = sum(Nvo_i*V)/real(size(param(ph)%tem_areas))/3.0   ! gnd_v2
  ! lmb = (1.0/sum(Nvo_i))**(1.0/3.0)
  lmb = 1.0/(8*sum(r_i**2*Nvo_i))

  where (r_i>rcl)
    phi_i = 1.0
  else where (r_i<rtrans)
    phi_i = 0.0
  else where
    phi_i = (r_i-rtrans)/(rcl-rtrans)
  end where
  phi = sum(phi_i*Nv_i)/sum(Nv_i)
  if (phi>0.0) then
    lmb_star = lmb/phi
  else
    lmb_star = 1.0e+200_pReal   ! inf
  end if

end subroutine c_TEM

subroutine c_G(T,u0,Tm,theta,G)
  real(pReal), intent(in) :: &
    T, u0, Tm, theta 
  real(pReal), intent(out) :: &
    G 

  G = u0*(1-T/Tm*exp(theta*(1-Tm/T)))
end subroutine c_G

subroutine c_prec_6111_avg(rtrans,b,f_all,Req_mean,G,beta,M,pp)
  real(pReal), intent(in) :: &
    rtrans,b,f_all,Req_mean,G,beta,M
  real(pReal), intent(out) :: &
    pp
  real(pReal) :: &
    L,Lf,F
  ! >--------------------------------------------------
  ! > Larry et al. average
  ! >--------------------------------------------------
    L = sqrt(2*atan(1.0_pReal)*4/f_all)*Req_mean
    
    if (Req_mean<sqrt(3.0)*rtrans/2) then 
      Lf = sqrt(sqrt(3.0)*rtrans/(2*Req_mean))*L-2*Req_mean
    else
      Lf = L-2*Req_mean
    end if 
    
    if (Req_mean<rtrans) then 
      F = 2*beta*G*b**2*(Req_mean/rtrans)
    else
      F = 2*beta*G*b**2
    end if
    
    pp = M*F/Lf/b 
end subroutine c_prec_6111_avg

subroutine c_prec_6111(ph,rtrans,b,f_all,Req_mean,G,beta,M,r_i,Nv_i,pp)
  integer, intent(in) :: &
    ph
  real(pReal), intent(in) :: &
    rtrans,b,f_all,Req_mean,G,beta,M
  real(pReal), dimension(param(ph)%classes), intent(in) :: &
    r_i,Nv_i
  real(pReal), intent(out) :: &
    pp
  real(pReal) :: &
    Lf_bar, F_bar
  real(pReal), dimension(param(ph)%classes) :: &
    Lf,F
  ! >--------------------------------------------------
  ! > Larry et al. distribution
  ! >--------------------------------------------------
    ! where (r_i < sqrt(3.0)*rtrans/2.0)
    !   Lf = sqrt(sqrt(3.0)*rtrans/2.0/r_i) * sqrt(2*atan(1.0_pReal)*4/f_all)*r_i - 2*r_i
    ! else where 
    !   Lf = sqrt(2*atan(1.0_pReal)*4/f_all)*r_i - 2*r_i
    ! end where

    ! where (r_i < rtrans)
    !   F = 2*beta*G*b**2*(r_i/rtrans)
    ! else where
    !   F = spread(2*beta*G*b**2, 1, size(r_i))
    ! end where

    ! Lf_bar = sum(Nv_i*Lf)/sum(Nv_i)
    ! F_bar = sum(Nv_i*F)/sum(Nv_i)
    ! pp = M*F_bar/Lf_bar/b

  ! >--------------------------------------------------
  ! > Myhr et al. distribution
  ! >--------------------------------------------------
    where (r_i < rtrans)
      F = 2*beta*G*b**2*(r_i/rtrans)
    else where
      F = spread(2*beta*G*b**2, 1, size(r_i))
    end where
    F_bar = sum(Nv_i*F)/sum(Nv_i)
    pp = M/(b*Req_mean)/sqrt(2*beta*G*b**2)*sqrt(3*f_all/(2*atan(1.0_pReal)*4))*F_bar**(3.0/2.0)  

end subroutine c_prec_6111

subroutine c_ss_6111(ph,f_all,wt,aw,cc,ss)
  integer, intent(in) :: &
    ph
  real(pReal), intent(in) :: &
    f_all 
  real(pReal), dimension(size(param(ph)%wt)), intent(in) :: &
    wt,aw
  real(pReal), intent(out) :: &
    ss 
  real(pReal), dimension(2), intent(out) :: &
    cc
  real(pReal), dimension(2) :: &
    cc_at
  real(pReal), dimension(3) :: &
    K
  real(pReal), dimension(size(param(ph)%wt)) :: &
    an,at 

  an = wt/aw
  at = an/sum(an)
!-------------------------------------------------------------------------------------
! AA6111
  cc_at = max(0.0, [at(3)-0.42*f_all, at(2)-0.33*f_all])
  cc = cc_at*sum(an)*[aw(3),aw(2)]
  K = [29.0e6, 66.3e6, 46.4e6]
  ss = dot_product(K, [cc, wt(4)]**(2.0/3.0))
!-------------------------------------------------------------------------------------
! ! AA7075
!   cc_at = max(0.0, [at(3)-0.33*f_all*100, at(2)-0.66*f_all*100])
!   cc = cc_at*sum(an)/100*[aw(3),aw(2)]
!   K = [18.6e6, 2.9e6, 13.8e6]
!   ss = dot_product(K, [cc, 0.0])
!-------------------------------------------------------------------------------------
end subroutine c_ss_6111

subroutine c_gb(grainSize,ky,gb)
  real(pReal), intent(in) :: &
    grainSize,ky 
  real(pReal), intent(out) :: &
    gb 

  gb = ky*1e6/sqrt(grainSize)
end subroutine c_gb

subroutine c_YS_temper(ph,T,rate,u0,pp,ss,gb,G,c1,RR,DG,e0,M,q,p,crss,xidot_slip)
  integer, intent(in) :: &
    ph
  real(pReal), intent(in) :: &
    T,rate,u0,pp,ss,gb,G,c1,RR,DG,e0,M,q,p
  real(pReal), intent(out) :: &
    crss
  real(pReal) :: &
    YS,YS_temper 
  real(pReal), dimension(param(ph)%sum_N_sl), intent(out), optional :: &
    xidot_slip

  if (present(xidot_slip)) then  
    xidot_slip = xidot_slip*c1*G/u0*(1.0-(T*RR/DG*log(e0/rate))**(1/q))**(1/p)
  else
    YS = 10.0e6+ss+pp+gb
    YS_temper = YS/c1*G/u0*(1.0-(T*RR/DG*log(e0/rate))**(1/q))**(1/p)
    crss = YS_temper/M
  endif  
end subroutine c_YS_temper

subroutine c_ssd_v1(ph,T,b,rate,u0,RR,wt,cc,G,M,k1,k3,Zs,m0,alpha,k20,Crmg,k2)
  integer, intent(in) :: &
    ph
  real(pReal), intent(in) :: &
    T,b,rate,u0,RR,G,M,k1,k3,Zs,m0,alpha,k20,Crmg
  real(pReal), dimension(2), intent(in) :: &
    cc
  real(pReal), dimension(size(param(ph)%wt)), intent(in) :: &
    wt
  real(pReal), intent(out) :: &
    k2
  real(pReal) :: &
    Cmg,k2_RT,Z

  Cmg = max(1.0e-2, cc(1)+0.5*(cc(2)-0.33*(wt(5)+wt(6))))
  Z = rate*exp(68.8*1000.0/RR/T)

! >--------------------------------------------------
! AA6111, naming: s1a
  k2_RT = k1*alpha*M*G*b/k3/(Cmg**(3.0/4.0))          ! dynamic recovery rate at room temperature
  k2 = k2_RT*(1+(Zs/Z)**m0) 
!-------------------------------------------------------------------------------------
! AA6111, naming: s1b
  ! k2 = k20*(G/u0)*(Crmg/Cmg)**(3.0/4.0)*(1.0+(Zs/Z)**m0)
!-------------------------------------------------------------------------------------
! ! AA7075
!   Cmg = max(1.0e-2, cc(1)+0.3*cc(2))
!   k2_RT = k1*alpha*M*G*b/k3/(Cmg**(3.0/4.0))
!   k2 = k2_RT*G/u0*(1+(Zs/Z)**m0)
!-------------------------------------------------------------------------------------
end subroutine c_ssd_v1

subroutine c_gnd_v1(T,rate,RR,k2g0,f_o,fr_o,Zg,m0,k2g)
  real(pReal), intent(in) :: &
    T,rate,RR,k2g0,f_o,fr_o,Zg,m0
  real(pReal), intent(out) :: &
    k2g
  real(pReal) :: &
    Z

  Z = rate*exp(68.8*1000.0/RR/T)
  k2g = k2g0*(f_o/fr_o)*(1.0+(Zg/Z)**m0)
end subroutine c_gnd_v1

subroutine c_gnd_v2(b,No_mean,lmb,f_o,ko,ksat,a_n,a_gnd,rhog_sat)
  real(pReal), intent(in) :: &
    b,No_mean,lmb,f_o,ko,ksat,a_n 
  real(pReal), intent(out) :: &
    a_gnd,rhog_sat
  
  rhog_sat = ksat/(f_o*b*ko*lmb)
  if (a_gnd<0.0) then
    a_gnd = (4.0/No_mean)**a_n
  endif
  ! if (No_mean/=0.0) then 
  !   rhog_sat = ksat/(f_o*b*ko*lmb)
  !   if (No_mean>=30.0) then 
  !     a_gnd = 0.001
  !   else if (No_mean>=20.0 .and. No_mean<30.0) then 
  !     a_gnd = 0.02 
  !   else if (No_mean>=10.0 .and. No_mean<20.0) then 
  !     a_gnd = 0.04
  !   else
  !     a_gnd = 10.0
  !   end if 
  ! else
  !   rhog_sat = 100.0
  !   a_gnd = 0.0
  ! end if     
    
end subroutine c_gnd_v2

end submodule phenopowerlaw







