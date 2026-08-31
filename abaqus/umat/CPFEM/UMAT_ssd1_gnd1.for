      SUBROUTINE UMAT(stress,statev,ddsdde,sse,spd,scd,
     1 rpl, ddsddt, drplde, drpldt,
     2 stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,
     3 ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt,
     4 celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc)

C     ND: equal to the total number of slip systems in all sets
      PARAMETER (ND=12)
  
      include 'aba_param.inc'
      CHARACTER*8 CMNAME
      EXTERNAL F

      dimension stress(ntens),statev(nstatv),
     1 ddsdde(ntens,ntens),ddsddt(ntens),drplde(ntens),
     2 stran(ntens),dstran(ntens),time(2),predef(1),dpred(1),
     3 props(nprops),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3)

      DIMENSION ISPDIR(3), ISPNOR(3), NSLIP(3), 
     2          SLPDIR(3,ND), SLPNOR(3,ND), SLPDEF(6,ND), 
     3          SLPSPN(3,ND), DSPDIR(3,ND), DSPNOR(3,ND), 
     4          DLOCAL(6,6), D(6,6), ROTD(6,6), ROTATE(3,3), 
     5          FSLIP(ND), DFDXSP(ND), DDEMSD(6,ND), 
     6          H(ND,ND), DDGDDE(ND,6), 
     7          DSTRES(6), DELATS(6), DSPIN(3), DVGRAD(3,3),
     8          DGAMMA(ND), DTAUSP(ND), DGSLIP(ND), 
     9          WORKST(ND,ND), INDX(ND), TERM(3,3), TRM0(3,3), ITRM(3)  

      DIMENSION FSLIP1(ND), STRES1(6), GAMMA1(ND), TAUSP1(ND), 
     2          GSLP1(ND), SPNOR1(3,ND), SPDIR1(3,ND), DDSDE1(6,6),
     3          DSOLD(6), DGAMOD(ND), DTAUOD(ND), DGSPOD(ND), 
     4          DSPNRO(3,ND), DSPDRO(3,ND), DHDGDG(ND,ND), RESIMA(12)

      REAL*8    RHO_S, RHO_G, DRHO_S, DRHO_G, DRHOOD_S, DRHOOD_G
                
C-----  Solution dependent state variable STATEV:
C            Denote the number of total slip systems by NSLPTL, which 
C            will be calculated in this code.
C       NSLPTL: 12
C       Array STATEV:
C       1          - NSLPTL    :  current strength in slip systems
C       NSLPTL+1   - 2*NSLPTL  :  shear strain in slip systems
C       2*NSLPTL+1 - 3*NSLPTL  :  resolved shear stress in slip systems
C
C       3*NSLPTL+1 - 6*NSLPTL  :  current components of normals to slip
C                                 slip planes
C       6*NSLPTL+1 - 9*NSLPTL  :  current components of slip directions
C
CFIX    9*NSLPTL+1 - 10*NSLPTL :  total cumulative shear strain on each 
CFIX                              slip system (sum of the absolute 
CFIX                              values of shear strains in each slip 
CFIX                              system individually)
CFIX
CFIX    10*NSLPTL+1            :  total cumulative shear strain on all 
C                                 slip systems (sum of the absolute 
C                                 values of shear strains in all slip 
C                                 systems)
C
CFIX    10*NSLPTL+2            :  This STATEV is currently not used.
C                                                                                               
C       NSTATV-3               :  number of slip systems in the 1st set
C       NSTATV-2               :  Total cumulative Rho_s
C       NSTATV-1               :  Total cumulative Rho_g
C       NSTATV                 :  total number of slip systems in all 
C                                 sets (The same as NSTATV-3. Only one 
C                                 slip system can be used in this UMAT.)
C
C
C-----  Material constants PROPS:
C
C       PROPS(1) - PROPS(8) -- elastic constants for a general elastic
C                               anisotropic material
C
C            isotropic   : PROPS(i)=0  for  i>2
C                          PROPS(1) -- Young's modulus
C                          PROPS(2) -- Poisson's ratio
C
C            cubic       : PROPS(i)=0  for i>3
C                          PROPS(1) -- c11
C                          PROPS(2) -- c12
C                          PROPS(3) -- c44
C
C       PROPS(9) - PROPS(16) -- parameters characterizing all slip 
C                               systems to be activated in a cubic 
C                               crystal (only one set of slip system)
C
C            PROPS(9) - PROPS(11) -- normal to a typical slip plane in
C                                     the first set of slip systems, 
C                                     e.g. (1 1 1)
C                                     (They must be real numbers, e.g. 
C                                      1. 1. 1., not 1 1 1)
C            PROPS(12) - PROPS(14) -- a typical slip direction in the 
C                                     first set of slip systems, e.g. 
C                                     [1 1 0]
C                                     (They must be real numbers, e.g. 
C                                      1. 1. 0., not 1 1 0)
C
C       PROPS(17) - PROPS(32) -- parameters characterizing the initial 
C                                orientation of a single crystal in 
C                                global system
C            The directions in global system and directions in local 
C            cubic crystal system of two nonparallel vectors are needed
C            to determine the crystal orientation.
C
C            PROPS(17) - PROPS(19) -- [p1 p2 p3], direction of first 
C                                     vector in local cubic crystal 
C                                     system, e.g. [1 1 0]
C                                     (They must be real numbers, e.g. 
C                                      1. 1. 0., not 1 1 0 !)
C            PROPS(20) - PROPS(22) -- [P1 P2 P3], direction of first 
C                                     vector in global system, e.g. 
C                                     [2. 1. 0.]
C                                     (It does not have to be a unit 
C                                      vector)
C
C            PROPS(25) - PROPS(27) -- direction of second vector in 
C                                     local cubic crystal system (real 
C                                     numbers)
C            PROPS(28) - PROPS(30) -- direction of second vector in 
C                                     global system
C
C
C       PROPS(33) - PROPS(34) -- parameters characterizing the visco-
C                                plastic constitutive law (shear 
C                                strain-rate vs. resolved shear 
C                                stress), e.g. a power-law relation
C                                (for one set of slip system)
C
C       PROPS(40) -- Initial value for dislocation density (Can not be zero)
C
C       PROPS(41) - PROPS(48)-- parameters characterizing the self-
C                               and latent-hardening laws of slip system
C     
C                     tau0, alpha(0.3), G(24400 Mpa), b(0.286e-6 mm)
C               k1: the dislocation accumulation coefficient
C                               k2: the coefficient connecting the equivalent
C                                   Mg concentration and the dynamic recovery
C                                L: the characteristic distance between precipitates
C                          Rho_sat: the density of GND at saturation
C
C       PROPS(49)- PROPS(50)-- latent-hardening parameters for 
C                              the first set of slip systems and
C                              interaction with other sets of 
C                              slip systems
C
C       PROPS(51)           -- parameter used for calculating the density of GND
C                              The value: a=10 is usually be used.
C
C======================================================================
C-----  Implicit integration parameter: THETA
      THETA = 0.5

C-----  Finite deformation ?
C-----  NLGEOM = 0,   small deformation theory
C       otherwise, theory of finite rotation and finite strain, Users 
C       must declare "NLGEOM" in the input file, at the *STEP card

      NLGEOM = 1

C-----  Iteration?
C-----  ITRATN = 0, no iteration
C       otherwise, iteration (solving increments of stresses and 
C       solution dependent state variables)

      ITRATN = 1
      ITRMAX = 60
      GAMERR = 1e-8

C-----  Total number of sets of slip systems: NSET = 1
C       In this UMAT, only 1 can be used.
      NSET = 1
C======================================================================
C-----  Elastic matrix in local cubic crystal system: DLOCAL
      DO J=1,6
         DO I=1,6
            DLOCAL(I,J)=0.
         END DO
      END DO

      IF (PROPS(3).EQ.0.) THEN

C-----  Isotropic material
         GSHEAR=PROPS(1)/2./(1.+PROPS(2))
         E11=2.*GSHEAR*(1.-PROPS(2))/(1.-2.*PROPS(2))
         E12=2.*GSHEAR*PROPS(2)/(1.-2.*PROPS(2))

         DO J=1,3
            DLOCAL(J,J)=E11

            DO I=1,3
               IF (I.NE.J) DLOCAL(I,J)=E12
            END DO

            DLOCAL(J+3,J+3)=GSHEAR
         END DO

      ELSE

C-----  Cubic material
         DO J=1,3
            DLOCAL(J,J)=PROPS(1)

            DO I=1,3
               IF (I.NE.J) DLOCAL(I,J)=PROPS(2)
            END DO

            DLOCAL(J+3,J+3)=PROPS(3)
         END DO

      END IF

C-----  Rotation matrix: ROTATE(3,3), i.e. direction cosines of [100], [010]
C     and [001] of a cubic crystal in global system
C     {ROTATE}*{L} = {G}

      CALL ROTATION (PROPS(17), ROTATE)

C-----  Rotation matrix: ROTD(6,6) to transform local elastic matrix DLOCAL 
C     to global elastic matrix D

      DO J=1,3
         J1=1+J/3
         J2=2+J/2

         DO I=1,3
            I1=1+I/3
            I2=2+I/2

            ROTD(I,J)=ROTATE(I,J)**2
            ROTD(I,J+3)=2.*ROTATE(I,J1)*ROTATE(I,J2)
            ROTD(I+3,J)=ROTATE(I1,J)*ROTATE(I2,J)
            ROTD(I+3,J+3)=ROTATE(I1,J1)*ROTATE(I2,J2)+
     2                    ROTATE(I1,J2)*ROTATE(I2,J1)

         END DO
      END DO

C-----  Elastic matrix in global system: D
C     {D} = {ROTD} * {DLOCAL} * {ROTD}transpose
C
      DO J=1,6
         DO I=1,6
            D(I,J)=0.
         END DO
      END DO

      DO J=1,6
         DO I=1,J

            DO K=1,6
               DO L=1,6
                  D(I,J)=D(I,J)+DLOCAL(K,L)*ROTD(I,K)*ROTD(J,L)
               END DO
            END DO

            D(J,I)=D(I,J)

         END DO
      END DO
C======================================================================
C-----  Initial the number of iterations
      NITRTN=-1

      DO I=1,NTENS
         DSOLD(I)=0.
      END DO

C-----  Initial the values of 12 slip systems for iteration
      DO J=1,ND
         DGAMOD(J)=0.
         DTAUOD(J)=0.
         DGSPOD(J)=0.
         DO I=1,3
            DSPNRO(I,J)=0.
            DSPDRO(I,J)=0.
         END DO
      END DO
C-----  Initial the values of dislocation density for iteration
      DRHOOD_S=0. 
      DRHOOD_G=0.

C-----  Increment of spin associated with the material element: DSPIN
C     (only needed for finite rotation)
C
      IF (NLGEOM.NE.0) THEN
         DO J=1,3
            DO I=1,3
               TERM(I,J)=DROT(J,I)
               TRM0(I,J)=DROT(J,I)
            END DO

            TERM(J,J)=TERM(J,J)+1.D0
            TRM0(J,J)=TRM0(J,J)-1.D0
         END DO

         CALL LUDCMP (TERM, 3, 3, ITRM, DDCMP)

         DO J=1,3
            CALL LUBKSB (TERM, 3, 3, ITRM, TRM0(1,J))
         END DO

         DSPIN(1)=TRM0(2,1)-TRM0(1,2)
         DSPIN(2)=TRM0(1,3)-TRM0(3,1)
         DSPIN(3)=TRM0(3,2)-TRM0(2,3)

      END IF

C-----  Increment of dilatational strain: DEV
      DEV=0.D0
      DO I=1,NDI
         DEV=DEV+DSTRAN(I)
      END DO

C======================================================================
C-----  Iteration starts (only when iteration method is used)
1000  CONTINUE

C-----  Parameter NITRTN: number of iterations
C       NITRTN = 0 --- no-iteration solution
C
      NITRTN=NITRTN+1

C-----  Check whether the current stress state is the initial state
      IF (STATEV(1).EQ.0.) THEN

C-----  Initial state
C
C-----  Generating the following parameters and variables at initial 
C     state:
C          Total number of slip systems in all the sets NSLPTL
C          Number of slip systems in each set NSLIP
C          Unit vectors in initial slip directions SLPDIR
C          Unit normals to initial slip planes SLPNOR
C
         NSLPTL=0
         DO I=1,NSET
            ISPNOR(1)=NINT(PROPS(1+8*I))
            ISPNOR(2)=NINT(PROPS(2+8*I))
            ISPNOR(3)=NINT(PROPS(3+8*I))

            ISPDIR(1)=NINT(PROPS(4+8*I))
            ISPDIR(2)=NINT(PROPS(5+8*I))
            ISPDIR(3)=NINT(PROPS(6+8*I))

            CALL SLIPSYS (ISPDIR, ISPNOR, NSLIP(I), SLPDIR(1,NSLPTL+1), 
     2                    SLPNOR(1,NSLPTL+1), ROTATE)

            NSLPTL=NSLPTL+NSLIP(I)
         END DO

         IF (ND.LT.NSLPTL) THEN
            WRITE (6,*) 
     2 '***ERROR - parameter ND chosen by the present user is less than
     3             the total number of slip systems NSLPTL'
            STOP
         END IF

C-----  Slip deformation tensor: SLPDEF (Schmid factors)
         DO J=1,NSLPTL
            SLPDEF(1,J)=SLPDIR(1,J)*SLPNOR(1,J)
            SLPDEF(2,J)=SLPDIR(2,J)*SLPNOR(2,J)
            SLPDEF(3,J)=SLPDIR(3,J)*SLPNOR(3,J)
            SLPDEF(4,J)=SLPDIR(1,J)*SLPNOR(2,J)+SLPDIR(2,J)*SLPNOR(1,J)
            SLPDEF(5,J)=SLPDIR(1,J)*SLPNOR(3,J)+SLPDIR(3,J)*SLPNOR(1,J)
            SLPDEF(6,J)=SLPDIR(2,J)*SLPNOR(3,J)+SLPDIR(3,J)*SLPNOR(2,J)
         END DO

C-----  Initial value of state variables: unit normal to a slip plane 
C       and unit vector in a slip direction
C       NSTATV = Depvar (126)
         STATEV(NSTATV)=FLOAT(NSLPTL)
         DO I=1,NSET
            STATEV(NSTATV-4+I)=FLOAT(NSLIP(I)) ! NSLIP = 12
         END DO
         IDNOR=3*NSLPTL                        ! IDNOR = 36
         IDDIR=6*NSLPTL                        ! IDDIR = 72

C       Get the information about slip systems

         DO J=1,NSLPTL
            DO I=1,3
               IDNOR=IDNOR+1
               STATEV(IDNOR)=SLPNOR(I,J)

               IDDIR=IDDIR+1
               STATEV(IDDIR)=SLPDIR(I,J)
            END DO
         END DO

C-----  Initial value of the current strength for all slip systems
C
C       PROPS(41): tau0 (yield stress)
C       To get current shear stress : g_alpha 
C       from -> GSLP0 = tau0 + alpha*G*b*sqrt(rho)
C
         CALL GSLPINIT (STATEV(1), NSLIP, NSLPTL, NSET, PROPS(41))

         DO I=1,NSLPTL

C       Initial value of shear strain in each slip system
            STATEV(NSLPTL+I)=0.

C       Initial value of cumulative shear strain in each slip system
            STATEV(9*NSLPTL+I)=0.

         END DO

C       Initial value of cumulative shear strain of all slip systems
         STATEV(10*NSLPTL+1)=0.

C       Initial value of dislocation density

C        STATEV(122): all dislocation density (will not used)
C        STATEV(10*NSLPTL+2)=PROPS(40)

C        STATEV(124): the average densities of statistically stored
         STATEV(NSTATV-2)=PROPS(40)

C        STATEV(125): the average densities of GND
         STATEV(NSTATV-1)=PROPS(40)

C-----  Initial value of the resolved shear stress in slip systems
         DO I=1,NSLPTL
            TERM1=0.

            DO J=1,NTENS
               IF (J.LE.NDI) THEN
                  TERM1=TERM1+SLPDEF(J,I)*STRESS(J)
               ELSE
                  TERM1=TERM1+SLPDEF(J-NDI+3,I)*STRESS(J)
               END IF
            END DO
C-----  Calculate the resolved shear stress in slip systems
            STATEV(2*NSLPTL+I)=TERM1
         END DO

C-----  When the current stress state is not the initial state
     
      ELSE

C-----  Current stress state
C
C-----  Copying from the array of state variables STATEV the following
C       parameters and variables at current stress state:
C       Total number of slip systems in all the sets NSLPTL (12)
C       Number of slip systems in each set NSLIP 
C       Current slip directions SLPDIR
C       Normals to current slip planes SLPNOR
C
         NSLPTL=NINT(STATEV(NSTATV))
C        NSET = 1 : Number of slip systems sets
         DO I=1,NSET
C           Number of slip systems in 1st set: 12
            NSLIP(I)=NINT(STATEV(NSTATV-4+I))
         END DO

         IDNOR=3*NSLPTL
         IDDIR=6*NSLPTL
C
         DO J=1,NSLPTL
            DO I=1,3
               IDNOR=IDNOR+1
               SLPNOR(I,J)=STATEV(IDNOR)

               IDDIR=IDDIR+1
               SLPDIR(I,J)=STATEV(IDDIR)
            END DO
         END DO

C-----  Slip deformation tensor: SLPDEF (Schmid factors)
         DO J=1,NSLPTL
            SLPDEF(1,J)=SLPDIR(1,J)*SLPNOR(1,J)
            SLPDEF(2,J)=SLPDIR(2,J)*SLPNOR(2,J)
            SLPDEF(3,J)=SLPDIR(3,J)*SLPNOR(3,J)
            SLPDEF(4,J)=SLPDIR(1,J)*SLPNOR(2,J)+SLPDIR(2,J)*SLPNOR(1,J)
            SLPDEF(5,J)=SLPDIR(1,J)*SLPNOR(3,J)+SLPDIR(3,J)*SLPNOR(1,J)
            SLPDEF(6,J)=SLPDIR(2,J)*SLPNOR(3,J)+SLPDIR(3,J)*SLPNOR(2,J)
         END DO

      END IF

C-----  Slip spin tensor: SLPSPN (only needed for finite rotation)
C       NLGEOM is set as 1 in this code
      IF (NLGEOM.NE.0) THEN
         DO J=1,NSLPTL
            SLPSPN(1,J)=0.5*(SLPDIR(1,J)*SLPNOR(2,J)-
     2                       SLPDIR(2,J)*SLPNOR(1,J))
            SLPSPN(2,J)=0.5*(SLPDIR(3,J)*SLPNOR(1,J)-
     2                       SLPDIR(1,J)*SLPNOR(3,J))
            SLPSPN(3,J)=0.5*(SLPDIR(2,J)*SLPNOR(3,J)-
     2                       SLPDIR(3,J)*SLPNOR(2,J))
         END DO
      END IF

C======================================================================

C-----  Double dot product of elastic moduli tensor with the slip 
C     deformation tensor (Schmid factors) plus, only for finite 
C     rotation, the dot product of slip spin tensor with the stress: 
C     DDEMSD
C
      DO J=1,NSLPTL
         DO I=1,6
            DDEMSD(I,J)=0.
            DO K=1,6
               DDEMSD(I,J)=DDEMSD(I,J)+D(K,I)*SLPDEF(K,J)
            END DO
         END DO
      END DO

      IF (NLGEOM.NE.0) THEN
         DO J=1,NSLPTL

            DDEMSD(4,J)=DDEMSD(4,J)-SLPSPN(1,J)*STRESS(1)
            DDEMSD(5,J)=DDEMSD(5,J)+SLPSPN(2,J)*STRESS(1)

            IF (NDI.GT.1) THEN
               DDEMSD(4,J)=DDEMSD(4,J)+SLPSPN(1,J)*STRESS(2)
               DDEMSD(6,J)=DDEMSD(6,J)-SLPSPN(3,J)*STRESS(2)
            END IF

            IF (NDI.GT.2) THEN
               DDEMSD(5,J)=DDEMSD(5,J)-SLPSPN(2,J)*STRESS(3)
               DDEMSD(6,J)=DDEMSD(6,J)+SLPSPN(3,J)*STRESS(3)
            END IF

            IF (NSHR.GE.1) THEN
               DDEMSD(1,J)=DDEMSD(1,J)+SLPSPN(1,J)*STRESS(NDI+1)
               DDEMSD(2,J)=DDEMSD(2,J)-SLPSPN(1,J)*STRESS(NDI+1)
               DDEMSD(5,J)=DDEMSD(5,J)-SLPSPN(3,J)*STRESS(NDI+1)
               DDEMSD(6,J)=DDEMSD(6,J)+SLPSPN(2,J)*STRESS(NDI+1)
            END IF

            IF (NSHR.GE.2) THEN
               DDEMSD(1,J)=DDEMSD(1,J)-SLPSPN(2,J)*STRESS(NDI+2)
               DDEMSD(3,J)=DDEMSD(3,J)+SLPSPN(2,J)*STRESS(NDI+2)
               DDEMSD(4,J)=DDEMSD(4,J)+SLPSPN(3,J)*STRESS(NDI+2)
               DDEMSD(6,J)=DDEMSD(6,J)-SLPSPN(1,J)*STRESS(NDI+2)
            END IF

            IF (NSHR.EQ.3) THEN
               DDEMSD(2,J)=DDEMSD(2,J)+SLPSPN(3,J)*STRESS(NDI+3)
               DDEMSD(3,J)=DDEMSD(3,J)-SLPSPN(3,J)*STRESS(NDI+3)
               DDEMSD(4,J)=DDEMSD(4,J)-SLPSPN(2,J)*STRESS(NDI+3)
               DDEMSD(5,J)=DDEMSD(5,J)+SLPSPN(1,J)*STRESS(NDI+3)
            END IF

         END DO
      END IF

C-----  Shear strain-rate in a slip system at the start of increment: 
C     FSLIP, and its derivative: DFDXSP
C 
    
      ID=1
      DO I=1,NSET
         IF (I.GT.1) ID=ID+NSLIP(I-1)

C        STATEV(NSLPTL+ID): shear strain in each slip system
C      STATEV(2*NSLPTL+ID): RSS in each slip system
C               STATEV(ID): current strength in each system
         CALL STRAINRATE (STATEV(NSLPTL+ID), STATEV(2*NSLPTL+ID), 
     2                    STATEV(ID), NSLIP(I), FSLIP(ID), DFDXSP(ID), 
     3                    PROPS(25+8*I))

C        Parameters for flow rule(PROPS(25+8*I)): m & gamma_0
C        FSLIP(Shear strain-rate): gamma_0*sign(tau/g)*|tau/g|**m
      END DO

C-----  Self- and latent-hardening laws
CFIXA  
       CALL LATENTHARDEN (NSLIP, NSLPTL, NSET, H(1,1), PROPS(41),
     2                    ND, STATEV(NSTATV-2), STATEV(NSTATV-1))
CFIXB

C-----  LU decomposition to solve the increment of shear strain in a 
C     slip system
C
      TERM1=THETA*DTIME
      DO I=1,NSLPTL
         TAUSLP=STATEV(2*NSLPTL+I)
         GSLIP=STATEV(I)
         X=TAUSLP/GSLIP
         TERM2=TERM1*DFDXSP(I)/GSLIP
         TERM3=TERM1*X*DFDXSP(I)/GSLIP

         DO J=1,NSLPTL
            TERM4=0.
            DO K=1,6
               TERM4=TERM4+DDEMSD(K,I)*SLPDEF(K,J)
            END DO

            WORKST(I,J)=TERM2*TERM4+H(I,J)*TERM3*DSIGN(1.D0,FSLIP(J))
      
            IF (NITRTN.GT.0) WORKST(I,J)=WORKST(I,J)+TERM3*DHDGDG(I,J)
         
      END DO

         WORKST(I,I)=WORKST(I,I)+1.
      END DO

      CALL LUDCMP (WORKST, NSLPTL, ND, INDX, DDCMP)

C-----  Increment of shear strain in a slip system: DGAMMA
      TERM1=THETA*DTIME
      DO I=1,NSLPTL

         IF (NITRTN.EQ.0) THEN
            TAUSLP=STATEV(2*NSLPTL+I)
            GSLIP=STATEV(I)
            X=TAUSLP/GSLIP
            TERM2=TERM1*DFDXSP(I)/GSLIP

            DGAMMA(I)=0.
            DO J=1,NDI
               DGAMMA(I)=DGAMMA(I)+DDEMSD(J,I)*DSTRAN(J)
            END DO

            IF (NSHR.GT.0) THEN
               DO J=1,NSHR
                  DGAMMA(I)=DGAMMA(I)+DDEMSD(J+3,I)*DSTRAN(J+NDI)
               END DO
            END IF

            DGAMMA(I)=DGAMMA(I)*TERM2+FSLIP(I)*DTIME

         ELSE
C           Here calculate 'R'
            DGAMMA(I)=TERM1*(FSLIP(I)-FSLIP1(I))+FSLIP1(I)*DTIME
     2                -DGAMOD(I)

         END IF

      END DO

      CALL LUBKSB (WORKST, NSLPTL, ND, INDX, DGAMMA)

      DO I=1,NSLPTL
         DGAMMA(I)=DGAMMA(I)+DGAMOD(I)
      END DO

C-----  Update the shear strain in a slip system: STATEV(NSLPTL+1) - 
C     STATEV(2*NSLPTL)
C     DGAMOD: previous DGAMMA
      DO I=1,NSLPTL
         STATEV(NSLPTL+I)=STATEV(NSLPTL+I)+DGAMMA(I)-DGAMOD(I)
      END DO     

C-----  Increment of current strength in a slip system: DGSLIP
      DO I=1,NSLPTL
         DGSLIP(I)=0.
         DO J=1,NSLPTL
            DGSLIP(I)=DGSLIP(I)+H(I,J)*ABS(DGAMMA(J))
         END DO
      END DO

C-----  Update the current strength in a slip system: STATEV(1) - 
C     STATEV(NSLPTL)
C
      DO I=1,NSLPTL
         STATEV(I)=STATEV(I)+DGSLIP(I)-DGSPOD(I)
      END DO

C**********************************************************************
C-----  Increment of current dislocation density:
C     RHO_S, RHO_G, DRHO_S, DRHO_G, DRHOOD_S, DRHOOD_G
C     
      TEMP_DGATOL=0.
      DO I=1,NSLPTL
        TEMP_DGATOL=TEMP_DGATOL+ABS(DGAMMA(I))   
      END DO

      RHO_S=STATEV(NSTATV-2)
      RHO_G=STATEV(NSTATV-1)

      DRHO_S=(PROPS(45)*SQRT(RHO_S)-PROPS(46)*RHO_S)*TEMP_DGATOL
      DRHO_G=(1.-(RHO_G/PROPS(48))**PROPS(51))/PROPS(44)/PROPS(47)*TEMP_DGATOL

      STATEV(NSTATV-2)=STATEV(NSTATV-2)+DRHO_S-DRHOOD_S
      STATEV(NSTATV-1)=STATEV(NSTATV-1)+DRHO_G-DRHOOD_G
    
C**********************************************************************

C-----  Increment of strain associated with lattice stretching: DELATS
      DO J=1,6
         DELATS(J)=0.
      END DO

      DO J=1,3
         IF (J.LE.NDI) DELATS(J)=DSTRAN(J)
         DO I=1,NSLPTL
            DELATS(J)=DELATS(J)-SLPDEF(J,I)*DGAMMA(I)
         END DO
      END DO

      DO J=1,3
         IF (J.LE.NSHR) DELATS(J+3)=DSTRAN(J+NDI)
         DO I=1,NSLPTL
            DELATS(J+3)=DELATS(J+3)-SLPDEF(J+3,I)*DGAMMA(I)
         END DO
      END DO

C-----  Increment of deformation gradient associated with lattice 
C     stretching in the current state, i.e. the velocity gradient 
C     (associated with lattice stretching) times the increment of time:
C     DVGRAD (only needed for finite rotation)
C
      IF (NLGEOM.NE.0) THEN
         DO J=1,3
            DO I=1,3
               IF (I.EQ.J) THEN
                  DVGRAD(I,J)=DELATS(I)
               ELSE
                  DVGRAD(I,J)=DELATS(I+J+1)
               END IF
            END DO
         END DO

         DO J=1,3
            DO I=1,J
               IF (J.GT.I) THEN
                  IJ2=I+J-2
                  IF (MOD(IJ2,2).EQ.1) THEN
                     TERM1=1.
                  ELSE
                     TERM1=-1.
                  END IF

                  DVGRAD(I,J)=DVGRAD(I,J)+TERM1*DSPIN(IJ2)
                  DVGRAD(J,I)=DVGRAD(J,I)-TERM1*DSPIN(IJ2)

                  DO K=1,NSLPTL
                     DVGRAD(I,J)=DVGRAD(I,J)-TERM1*DGAMMA(K)*
     2                                       SLPSPN(IJ2,K)
                     DVGRAD(J,I)=DVGRAD(J,I)+TERM1*DGAMMA(K)*
     2                                       SLPSPN(IJ2,K)
                  END DO
               END IF
            END DO
         END DO
      END IF

C-----  Increment of resolved shear stress in a slip system: DTAUSP
      DO I=1,NSLPTL
         DTAUSP(I)=0.
         DO J=1,6
            DTAUSP(I)=DTAUSP(I)+DDEMSD(J,I)*DELATS(J)
         END DO
      END DO

C-----  Update the resolved shear stress in a slip system: 
C     STATEV(2*NSLPTL+1) - STATEV(3*NSLPTL)
C
      DO I=1,NSLPTL
         STATEV(2*NSLPTL+I)=STATEV(2*NSLPTL+I)+DTAUSP(I)-DTAUOD(I)
      END DO

C-----  Increment of stress: DSTRES
      IF (NLGEOM.EQ.0) THEN
         DO I=1,NTENS
            DSTRES(I)=0.
         END DO
      ELSE
         DO I=1,NTENS
            DSTRES(I)=-STRESS(I)*DEV
         END DO
      END IF

      DO I=1,NDI
         DO J=1,NDI
            DSTRES(I)=DSTRES(I)+D(I,J)*DSTRAN(J)
         END DO

         IF (NSHR.GT.0) THEN
            DO J=1,NSHR
               DSTRES(I)=DSTRES(I)+D(I,J+3)*DSTRAN(J+NDI)
            END DO
         END IF

         DO J=1,NSLPTL
            DSTRES(I)=DSTRES(I)-DDEMSD(I,J)*DGAMMA(J)
         END DO
      END DO

      IF (NSHR.GT.0) THEN
         DO I=1,NSHR

            DO J=1,NDI
               DSTRES(I+NDI)=DSTRES(I+NDI)+D(I+3,J)*DSTRAN(J)
            END DO

            DO J=1,NSHR
               DSTRES(I+NDI)=DSTRES(I+NDI)+D(I+3,J+3)*DSTRAN(J+NDI)
            END DO

            DO J=1,NSLPTL
               DSTRES(I+NDI)=DSTRES(I+NDI)-DDEMSD(I+3,J)*DGAMMA(J)
            END DO

         END DO
      END IF

C-----  Update the stress: STRESS
      DO I=1,NTENS
         STRESS(I)=STRESS(I)+DSTRES(I)-DSOLD(I)
      END DO

C-----  Increment of normal to a slip plane and a slip direction (only 
C     needed for finite rotation)
C
      IF (NLGEOM.NE.0) THEN
         DO J=1,NSLPTL
            DO I=1,3
               DSPNOR(I,J)=0.
               DSPDIR(I,J)=0.

               DO K=1,3
                  DSPNOR(I,J)=DSPNOR(I,J)-SLPNOR(K,J)*DVGRAD(K,I)
                  DSPDIR(I,J)=DSPDIR(I,J)+SLPDIR(K,J)*DVGRAD(I,K)
               END DO

            END DO
         END DO

C-----  Update the normal to a slip plane and a slip direction (only 
C     needed for finite rotation)
C
         IDNOR=3*NSLPTL
         IDDIR=6*NSLPTL
         DO J=1,NSLPTL
            DO I=1,3
               IDNOR=IDNOR+1
               STATEV(IDNOR)=STATEV(IDNOR)+DSPNOR(I,J)-DSPNRO(I,J)

               IDDIR=IDDIR+1
               STATEV(IDDIR)=STATEV(IDDIR)+DSPDIR(I,J)-DSPDRO(I,J)
            END DO
         END DO

      END IF

C-----  Derivative of shear strain increment in a slip system w.r.t. 
C     strain increment: DDGDDE
C
      TERM1=THETA*DTIME
      DO I=1,NTENS
         DO J=1,NSLPTL
            TAUSLP=STATEV(2*NSLPTL+J)
            GSLIP=STATEV(J)
            X=TAUSLP/GSLIP
            TERM2=TERM1*DFDXSP(J)/GSLIP
            IF (I.LE.NDI) THEN
               DDGDDE(J,I)=TERM2*DDEMSD(I,J)
            ELSE
               DDGDDE(J,I)=TERM2*DDEMSD(I-NDI+3,J)
            END IF
         END DO

         CALL LUBKSB (WORKST, NSLPTL, ND, INDX, DDGDDE(1,I))

      END DO

C-----  Derivative of stress increment w.r.t. strain increment, i.e. 
C     Jacobian matrix
C
C-----  Jacobian matrix: elastic part
      DO J=1,NTENS
         DO I=1,NTENS
            DDSDDE(I,J)=0.
         END DO
      END DO

      DO J=1,NDI
         DO I=1,NDI
            DDSDDE(I,J)=D(I,J)
            IF (NLGEOM.NE.0) DDSDDE(I,J)=DDSDDE(I,J)-STRESS(I)
         END DO
      END DO

      IF (NSHR.GT.0) THEN
         DO J=1,NSHR
            DO I=1,NSHR
               DDSDDE(I+NDI,J+NDI)=D(I+3,J+3)
            END DO

            DO I=1,NDI
               DDSDDE(I,J+NDI)=D(I,J+3)
               DDSDDE(J+NDI,I)=D(J+3,I)
               IF (NLGEOM.NE.0)
     2            DDSDDE(J+NDI,I)=DDSDDE(J+NDI,I)-STRESS(J+NDI)
            END DO
         END DO
      END IF

C-----  Jacobian matrix: plastic part (slip)
      DO J=1,NDI
         DO I=1,NDI
            DO K=1,NSLPTL
               DDSDDE(I,J)=DDSDDE(I,J)-DDEMSD(I,K)*DDGDDE(K,J)
            END DO
         END DO
      END DO

      IF (NSHR.GT.0) THEN
         DO J=1,NSHR

            DO I=1,NSHR
               DO K=1,NSLPTL
                  DDSDDE(I+NDI,J+NDI)=DDSDDE(I+NDI,J+NDI)-
     2                                DDEMSD(I+3,K)*DDGDDE(K,J+NDI)
               END DO
            END DO

            DO I=1,NDI
               DO K=1,NSLPTL
                  DDSDDE(I,J+NDI)=DDSDDE(I,J+NDI)-
     2                            DDEMSD(I,K)*DDGDDE(K,J+NDI)
                  DDSDDE(J+NDI,I)=DDSDDE(J+NDI,I)-
     2                            DDEMSD(J+3,K)*DDGDDE(K,I)
               END DO
            END DO

         END DO
      END IF

      IF (ITRATN.NE.0) THEN
         DO J=1,NTENS
            DO I=1,NTENS
               DDSDDE(I,J)=DDSDDE(I,J)/(1.+DEV)
            END DO
         END DO
      END IF

C-----  Iteration ?
      IF (ITRATN.NE.0) THEN

C-----  Save solutions (without iteration):
C            Shear strain-rate in a slip system FSLIP1
C            Current strength in a slip system GSLP1
C            Shear strain in a slip system GAMMA1
C            Resolved shear stress in a slip system TAUSP1
C            Dislocation density RHO_S1 | RHO_G1
C            Normal to a slip plane SPNOR1
C            Slip direction SPDIR1
C            Stress STRES1
C            Jacobian matrix DDSDE1
C
         IF (NITRTN.EQ.0) THEN

            IDNOR=3*NSLPTL
            IDDIR=6*NSLPTL
            DO J=1,NSLPTL
               FSLIP1(J)=FSLIP(J)
               GSLP1(J)=STATEV(J)
               GAMMA1(J)=STATEV(NSLPTL+J)
               TAUSP1(J)=STATEV(2*NSLPTL+J)
               DO I=1,3
                  IDNOR=IDNOR+1
                  SPNOR1(I,J)=STATEV(IDNOR)

                  IDDIR=IDDIR+1
                  SPDIR1(I,J)=STATEV(IDDIR)
               END DO
            END DO

            DO J=1,NTENS
               STRES1(J)=STRESS(J)
               DO I=1,NTENS
                  DDSDE1(I,J)=DDSDDE(I,J)
               END DO
            END DO

            RHO_S1=STATEV(NSTATV-2)
            RHO_G1=STATEV(NSTATV-1)

         END IF

C-----  Increments of stress DSOLD, and solution dependent state 
C     variables DGAMOD, DTAUOD, DGSPOD, DSPNRO, DSPDRO, 
C     DRHOOD_S, DRHOOD_G (for the next iteration)
C
         DO I=1,NTENS
            DSOLD(I)=DSTRES(I)
         END DO

         DO J=1,NSLPTL
            DGAMOD(J)=DGAMMA(J)
            DTAUOD(J)=DTAUSP(J)
            DGSPOD(J)=DGSLIP(J)
            DO I=1,3
               DSPNRO(I,J)=DSPNOR(I,J)
               DSPDRO(I,J)=DSPDIR(I,J)
            END DO
         END DO

         DRHOOD_S=DRHO_S
         DRHOOD_G=DRHO_G

C-----  Check if the iteration solution converges
     
         IDBACK=0
         ID=0
         DO I=1,NSET
            DO J=1,NSLIP(I)
               ID=ID+1
               X=STATEV(2*NSLPTL+ID)/STATEV(ID)
         RESIMA(ID)=0.
               RESIDU=THETA*DTIME*F(X,PROPS(25+8*I))
     2                +DTIME*(1.0-THETA)*FSLIP1(ID)-DGAMMA(ID) 
         RESIMA(ID)=RESIDU
               IF (ABS(RESIDU).GT.GAMERR) IDBACK=1
            END DO
         END DO

         IF (IDBACK.NE.0.AND.NITRTN.LT.ITRMAX) THEN
C-----  Iteration: arrays for iteration
CFIXA
            CALL ITERATION (NSLPTL,NSET,NSLIP,ND,PROPS(41),DGAMOD,
     2                      DHDGDG,STATEV(NSTATV-2),STATEV(NSTATV-1))
CFIXB
            GO TO 1000
     
         ELSE IF (NITRTN.GE.ITRMAX) THEN
C-----  Solution not converge within maximum number of iteration (the 
C     solution without iteration will be used)
C
            WRITE(*,*) "NITRTN.GE.ITRMAX"
      DO J=1,NTENS
               STRESS(J)=STRES1(J)
               DO I=1,NTENS
                  DDSDDE(I,J)=DDSDE1(I,J)
               END DO
            END DO

            IDNOR=3*NSLPTL
            IDDIR=6*NSLPTL
            DO J=1,NSLPTL
               STATEV(J)=GSLP1(J)
               STATEV(NSLPTL+J)=GAMMA1(J)
               STATEV(2*NSLPTL+J)=TAUSP1(J)

               DO I=1,3
                  IDNOR=IDNOR+1
                  STATEV(IDNOR)=SPNOR1(I,J)

                  IDDIR=IDDIR+1
                  STATEV(IDDIR)=SPDIR1(I,J)
               END DO
            END DO
      
            STATEV(NSTATV-2)=RHO_S1
            STATEV(NSTATV-1)=RHO_G1
      
         END IF

      END IF

C-----  Total cumulative shear strains on all slip systems (sum of the 
C       absolute values of shear strains in all slip systems)
CFIX--  Total cumulative shear strains on each slip system (sum of the 
CFIX    absolute values of shear strains in each individual slip system)
C
      DO I=1,NSLPTL
CFIXA
         STATEV(10*NSLPTL+1)=STATEV(10*NSLPTL+1)+ABS(DGAMMA(I))
         STATEV(9*NSLPTL+I)=STATEV(9*NSLPTL+I)+ABS(DGAMMA(I))
CFIXB 
      END DO
    
      RETURN
      END

C======================================================================

CFIXA
      SUBROUTINE ITERATION (NSLPTL, NSET, NSLIP, ND, PROP, DGAMOD, 
     2                      DHDGDG, RHOS, RHOG)
CFIXB

C-----  This subroutine generates arrays for the Newton-Rhapson 
C     iteration method.

C-----  Users who want to use their own self- and latent-hardening law 
C     may change the function subprograms DHSELF (self hardening) and 
C     DHLATN (latent hardening).  The parameters characterizing these 
C     hardening laws are passed into DHSELF and DHLATN through array 
C     PROP.


C-----  Function subprograms:
C
C       DHSELF -- User-supplied function of the derivative of self-
C                 hardening moduli
C
C       DHLATN -- User-supplied function of the derivative of latent-
C                 hardening moduli

C-----  Variables:
C
C     GAMMA  -- shear strain in all slip systems at the start of time 
C               step  (INPUT)
C     TAUSLP -- resolved shear stress in all slip systems (INPUT)
C     GSLIP  -- current strength (INPUT)
CFIX  GMSLTL -- total cumulative shear strains on each individual slip system 
CFIX            (INPUT)
C     GAMTOL -- total cumulative shear strains over all slip systems 
C               (INPUT)
C     NSLPTL -- total number of slip systems in all the sets (INPUT)
C     NSET   -- number of sets of slip systems (INPUT)
C     NSLIP  -- number of slip systems in each set (INPUT)
C     ND     -- leading dimension of arrays defined in subroutine UMAT 
C               (INPUT) 
C
C     PROP   -- material constants characterizing the self- and latent-
C               hardening law (INPUT)
C
C-----  Arrays for iteration:
C
C       DGAMOD (INPUT)
C
C       DHDGDG (OUTPUT)
C

C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      EXTERNAL DHSELF, DHLATN
CFIXA
      DIMENSION NSLIP(NSET), PROP(16,NSET), 
     2          DGAMOD(NSLPTL), DHDGDG(ND,NSLPTL)
      REAL*8 RHOS,RHOG
CFIXB
    
      ISELF=0
      DO I=1,NSET
         ISET=I
         DO J=1,NSLIP(I)
            ISELF=ISELF+1

            DO KDERIV=1,NSLPTL
               DHDGDG(ISELF,KDERIV)=0.

               DO LATENT=1,NSLPTL
                  IF (LATENT.EQ.ISELF) THEN
CFIXA
                     DHDG=DHSELF(PROP(1,I),RHOS,RHOG)
CFIXB        
                  ELSE
CFIXA
                     DHDG=DHLATN(NSET,NSLIP,PROP(1,I),ISET,
     2                           LATENT,RHOS,RHOG)
CFIXB
                  END IF

                  DHDGDG(ISELF,KDERIV)=DHDGDG(ISELF,KDERIV)+
     2                                 DHDG*ABS(DGAMOD(LATENT))
               END DO

            END DO
         END DO
      END DO

      RETURN
      END


C-----------------------------------


C-----  Use single precision on cray
CFIXA
           REAL*8 FUNCTION DHSELF(PROP,RHOS,RHOG)
CFIXB
C-----  User-supplied function of the derivative of self-hardening
C     moduli

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
CFIXA
           DIMENSION PROP(16)

           REAL*8 RHOS,RHOG

           TERM1=PROP(2)*PROP(3)*PROP(4)/2
           TERM2=RHOS+RHOG
           TERM3=PROP(5)*SQRT(RHOS)-PROP(6)*RHOS
           TERM4=PROP(12)-PROP(13)*RHOG
           TERM5=(PROP(5)/2/SQRT(RHOS)-PROP(6))*TERM3
           TERM6=PROP(13)*TERM4

           TEMP1=-0.5*TERM2**(-1.5)*(TERM3+TERM4)**2
           TEMP2=TERM2**(-0.5)*(TERM5-TERM6)

           DHSELF=TERM1*(TEMP1+TEMP2)
CFIXB
           RETURN
           END


C-----------------------------------


C-----  Use single precision on cray
CFIXA
           REAL*8 FUNCTION DHLATN(NSET,NSLIP,PROP,ISET,
     2                            LATENT,RHOS,RHOG)
CFIXB

C-----  User-supplied function of the derivative of latent-hardening 
C     moduli

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
CFIXA
           DIMENSION NSLIP(NSET),PROP(16)

           REAL*8 RHOS,RHOG
CFIXB

           ILOWER=0
           IUPPER=NSLIP(1)
           IF (ISET.GT.1) THEN
              DO K=2,ISET
                 ILOWER=ILOWER+NSLIP(K-1)
                 IUPPER=IUPPER+NSLIP(K)
              END DO
           END IF

           IF (LATENT.GT.ILOWER.AND.LATENT.LE.IUPPER) THEN
              Q=PROP(9)
           ELSE
              Q=PROP(10)
           END IF
CFIXA

           TERM1=PROP(2)*PROP(3)*PROP(4)/2
           TERM2=RHOS+RHOG
           TERM3=PROP(5)*SQRT(RHOS)-PROP(6)*RHOS
           TERM4=PROP(12)-PROP(13)*RHOG
           TERM5=(PROP(5)/2/SQRT(RHOS)-PROP(6))*TERM3
           TERM6=PROP(13)*TERM4

           TEMP1=-0.5*TERM2**(-1.5)*(TERM3+TERM4)**2
           TEMP2=TERM2**(-0.5)*(TERM5-TERM6)

           DHLATN=TERM1*(TEMP1+TEMP2)*Q

CFIXB
           RETURN
           END

C======================================================================

      SUBROUTINE LATENTHARDEN (NSLIP, NSLPTL, NSET, H, PROP, ND, 
     2                         RHOS, RHOG)
CFIXB

C-----  This subroutine calculates the current self- and latent-
C     hardening moduli for all slip systems in a rate-dependent single 
C     crystal.

C-----  The hardening coefficients are assumed the same for all slip 
C     systems in each set, though they could be different from set to 
C     set, e.g. <110>{111} and <110>{100}.

C-----  Users who want to use their own self- and latent-hardening law 
C     may change the function subprograms HSELF (self hardening) and 
C     HLATNT (latent hardening).  The parameters characterizing these 
C     hardening laws are passed into HSELF and HLATNT through array 
C     PROP.

C-----  Function subprograms:
C
C       HSELF  -- User-supplied self-hardening function in a slip 
C                 system
C
C       HLATNT -- User-supplied latent-hardening function

C-----  Variables:
C
C     NSLIP  -- number of slip systems in each set (INPUT)
C     NSLPTL -- total number of slip systems in all the sets (INPUT)
C     NSET   -- number of sets of slip systems (INPUT)
C
C     H      -- current value of self- and latent-hardening moduli 
C               (OUTPUT)
C               H(i,i) -- self-hardening modulus of the ith slip system
C                         (no sum over i)
C               H(i,j) -- latent-hardening molulus of the ith slip 
C                         system due to a slip in the jth slip system 
C                         (i not equal j)
C
C     PROP   -- material constants characterizing the self- and latent-
C               hardening law (INPUT)
C
C     ND     -- leading dimension of arrays defined in subroutine UMAT 
C               (INPUT) 


C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      EXTERNAL HSELF, HLATNT
CFIXA
      DIMENSION NSLIP(NSET), PROP(16,NSET), H(ND,NSLPTL) 
      REAL*8 RHOS,RHOG
CFIXB

      ISELF=0
      DO I=1,NSET
         ISET=I
         DO J=1,NSLIP(I)
            ISELF=ISELF+1

            DO LATENT=1,NSLPTL
               IF (LATENT.EQ.ISELF) THEN
CFIXA
                  H(LATENT,ISELF)=HSELF(PROP(1,I),RHOS,RHOG)
CFIXB
               ELSE
CFIXA
                  H(LATENT,ISELF)=HLATNT(NSET,NSLIP,PROP(1,I),
     2                                   ISET,LATENT,RHOS,RHOG)
CFIXB
               END IF
            END DO

         END DO
      END DO

      RETURN
      END

C-----------------------------------
C-----  Use single precision on cray
CFIXA
           REAL*8 FUNCTION HSELF(PROP,RHOS,RHOG)

C-----     User-supplied self-hardening function in a slip system

C-----  Use single precision on cray

           IMPLICIT REAL*8 (A-H,O-Z)
           DIMENSION PROP(16)
           REAL*8 RHOS,RHOG

           TERM1=PROP(2)*PROP(3)*PROP(4)/SQRT(RHOS+RHOG)
           TERM2=PROP(5)*SQRT(RHOS)-PROP(6)*RHOS
           TERM3=PROP(12)-PROP(13)*RHOG
    
           HSELF=TERM1*0.5*(TERM2+TERM3)

           RETURN
           END
CFIXB

C-----------------------------------
C-----  Use single precision on cray
CFIXA
           REAL*8 FUNCTION HLATNT(NSET,NSLIP,PROP,
     2                            ISET,LATENT,RHOS,RHOG)
CFIXB
C-----     User-supplied latent-hardening function

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
CFIXA
           DIMENSION NSLIP(NSET), PROP(16)
           REAL*8 RHOS,RHOG
CFIXB
           ILOWER=0
           IUPPER=NSLIP(1)
           IF (ISET.GT.1) THEN
              DO K=2,ISET
                 ILOWER=ILOWER+NSLIP(K-1)
                 IUPPER=IUPPER+NSLIP(K)
              END DO
           END IF

           IF (LATENT.GT.ILOWER.AND.LATENT.LE.IUPPER) THEN
              Q=PROP(9)
           ELSE
              Q=PROP(10)
           END IF
CFIXA
           TERM1=PROP(2)*PROP(3)*PROP(4)/SQRT(RHOS+RHOG)
           TERM2=PROP(5)*SQRT(RHOS)-PROP(6)*RHOS
           TERM3=PROP(12)-PROP(13)*RHOG

           HLATNT=TERM1*0.5*(TERM2+TERM3)*Q
CFIXB
           RETURN
           END

C======================================================================

      SUBROUTINE STRAINRATE (GAMMA, TAUSLP, GSLIP, NSLIP, FSLIP, 
     2                       DFDXSP, PROP)

C-----  This subroutine calculates the shear strain-rate in each slip 
C     system for a rate-dependent single crystal.  The POWER LAW 
C     relation between shear strain-rate and resolved shear stress 
C     proposed by Hutchinson, Pan and Rice, is used here.

C-----  The power law exponents are assumed the same for all slip 
C     systems in each set, though they could be different from set to 
C     set, e.g. <110>{111} and <110>{100}.  The strain-rate coefficient
C     in front of the power law form are also assumed the same for all 
C     slip systems in each set. 

C-----  Users who want to use their own constitutive relation may 
C     change the function subprograms F and its derivative DFDX, 
C     where F is the strain hardening law, dGAMMA/dt = F(X), 
C     X=TAUSLP/GSLIP.  The parameters characterizing F are passed into 
C     F and DFDX through array PROP.

C-----  Function subprograms:
C
C       F    -- User-supplied function subprogram which gives shear 
C               strain-rate for each slip system based on current 
C               values of resolved shear stress and current strength
C
C       DFDX -- User-supplied function subprogram dF/dX, where x is the
C               ratio of resolved shear stress over current strength

C-----  Variables:
C
C     GAMMA  -- shear strain in each slip system at the start of time 
C               step  (INPUT)
C     TAUSLP -- resolved shear stress in each slip system (INPUT)
C     GSLIP  -- current strength (INPUT)
C     NSLIP  -- number of slip systems in this set (INPUT)
C
C     FSLIP  -- current value of F for each slip system (OUTPUT)
C     DFDXSP -- current value of DFDX for each slip system (OUTPUT)
C
C     PROP   -- material constants characterizing the strain hardening 
C               law (INPUT)
C
C               For the current power law strain hardening law 
C               PROP(1) -- power law hardening exponent
C               PROP(1) = infinity corresponds to a rate-independent 
C               material
C               PROP(2) -- coefficient in front of power law hardening


C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      EXTERNAL F, DFDX
      DIMENSION GAMMA(NSLIP), TAUSLP(NSLIP), GSLIP(NSLIP), 
     2          FSLIP(NSLIP), DFDXSP(NSLIP), PROP(8)

      DO I=1,NSLIP
         X=TAUSLP(I)/GSLIP(I)
         FSLIP(I)=F(X,PROP)
         DFDXSP(I)=DFDX(X,PROP)
      END DO

      RETURN
      END


C-----------------------------------


C-----  Use single precision on cray
C
           REAL*8 FUNCTION F(X,PROP)

C-----     User-supplied function subprogram which gives shear 
C        strain-rate for each slip system based on current values of 
C        resolved shear stress and current strength
C
C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
           DIMENSION PROP(8)

           F=PROP(2)*(ABS(X))**PROP(1)*DSIGN(1.D0,X)

           RETURN
           END


C-----------------------------------


C-----  Use single precision on cray
C
           REAL*8 FUNCTION DFDX(X,PROP)

C-----     User-supplied function subprogram dF/dX, where x is the 
C        ratio of resolved shear stress over current strength

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
           DIMENSION PROP(8)

           DFDX=PROP(1)*PROP(2)*(ABS(X))**(PROP(1)-1.)

           RETURN
           END


C======================================================================

      SUBROUTINE GSLPINIT (GSLIP0, NSLIP, NSLPTL, NSET, PROP)

C-----  This subroutine calculates the initial value of current 
C     strength for each slip system in a rate-dependent single crystal.
C     Two sets of initial values, proposed by Asaro, Pierce et al, and 
C     by Bassani, respectively, are used here.  Both sets assume that 
C     the initial values for all slip systems are the same (initially 
C     isotropic).

C-----  These initial values are assumed the same for all slip systems 
C     in each set, though they could be different from set to set, e.g.
C     <110>{111} and <110>{100}.

C-----  Users who want to use their own initial values may change the 
C     function subprogram GSLP0.  The parameters characterizing these 
C     initial values are passed into GSLP0 through array PROP.

C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      EXTERNAL GSLP0
      DIMENSION GSLIP0(NSLPTL), NSLIP(NSET), PROP(8,NSET)

C-----  Function subprograms:
C
C       GSLP0 -- User-supplied function subprogram given the initial 
C                value of current strength at initial state

C-----  Variables:
C
C     GSLIP0 -- initial value of current strength (OUTPUT)
C
C     NSLIP  -- number (12) of slip systems in each set (INPUT)
C     NSLPTL -- total number (12) of slip systems in all the sets (INPUT)
C     NSET   -- number (1) of sets of slip systems (INPUT)
C
C     PROP   -- material constants characterizing the initial value of 
C               current strength (INPUT)

      ID=0
      DO I=1,NSET
         ISET=I
         DO J=1,NSLIP(I)
            ID=ID+1
            GSLIP0(ID)=GSLP0(NSLPTL,NSET,NSLIP,PROP(1,I),ID,ISET)
         END DO
      END DO
      
      RETURN
      END
C----------------------------------
C-----  Use single precision on cray
C
           REAL*8 FUNCTION GSLP0(NSLPTL,NSET,NSLIP,PROP,ISLIP,ISET)

C-----  User-supplied function subprogram given the initial value of
C       current strength at initial state

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
           DIMENSION NSLIP(NSET), PROP(8)

C          TERM1=PROP(2)*PROP(3)*PROP(4)
C          TERM2=SQRT(PROP(8))
C          GSLP0=TERM1*TERM2+PROP(1)
C          GSLP0 = tau0 + alpha*G*b*sqrt(rho)

           GSLP0=PROP(1)

           RETURN
           END

C======================================================================

      SUBROUTINE SLIPSYS (ISPDIR, ISPNOR, NSLIP, SLPDIR, SLPNOR, 
     2                    ROTATE)

C-----  This subroutine generates all slip systems in the same set for 
C     a CUBIC crystal.  For other crystals (e.g., HCP, Tetragonal, 
C     Orthotropic, ...), it has to be modified to include the effect of
C     crystal aspect ratio.

C-----  Denote s as a slip direction and m as normal to a slip plane.  
C     In a cubic crystal, (s,-m), (-s,m) and (-s,-m) are NOT considered
C     independent of (s,m).

C-----  Subroutines:  LINE1 and LINE

C-----  Variables:
C
C     ISPDIR -- a typical slip direction in this set of slip systems 
C               (integer)  (INPUT)
C     ISPNOR -- a typical normal to slip plane in this set of slip 
C               systems (integer)  (INPUT)
C     NSLIP  -- number of independent slip systems in this set 
C               (OUTPUT)
C     SLPDIR -- unit vectors of all slip directions  (OUTPUT)
C     SLPNOR -- unit normals to all slip planes  (OUTPUT)
C     ROTATE -- rotation matrix (INPUT)
C          ROTATE(i,1) -- direction cosines of [100] in global system
C          ROTATE(i,2) -- direction cosines of [010] in global system
C          ROTATE(i,3) -- direction cosines of [001] in global system
C
C     NSPDIR -- number of all possible slip directions in this set
C     NSPNOR -- number of all possible slip planes in this set
C     IWKDIR -- all possible slip directions (integer)
C     IWKNOR -- all possible slip planes (integer)


C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION ISPDIR(3), ISPNOR(3), SLPDIR(3,50), SLPNOR(3,50), 
     *          ROTATE(3,3), IWKDIR(3,24), IWKNOR(3,24), TERM(3)
!!    There might be bugs that there are more than 24 slip systems? 
      NSLIP=0
      NSPDIR=0
      NSPNOR=0

C-----  Generating all possible slip directions in this set
C
C       Denote the slip direction by [lmn].  I1 is the minimum of the 
C       absolute value of l, m and n, I3 is the maximum and I2 is the 
C       mode, e.g. (1 -3 2), I1=1, I2=2 and I3=3.  I1<=I2<=I3.

      I1=MIN(IABS(ISPDIR(1)),IABS(ISPDIR(2)),IABS(ISPDIR(3)))
      I3=MAX(IABS(ISPDIR(1)),IABS(ISPDIR(2)),IABS(ISPDIR(3)))
      I2=IABS(ISPDIR(1))+IABS(ISPDIR(2))+IABS(ISPDIR(3))-I1-I3

      RMODIR=SQRT(FLOAT(I1*I1+I2*I2+I3*I3))

C     I1=I2=I3=0
      IF (I3.EQ.0) THEN 
         WRITE (6,*) '***ERROR - slip direction is [000]'
         STOP

C     I1=I2=0, I3>0   ---   [001] type
      ELSE IF (I2.EQ.0) THEN
         NSPDIR=3
         DO J=1,3
            DO I=1,3
               IWKDIR(I,J)=0
               IF (I.EQ.J) IWKDIR(I,J)=I3
            END DO
         END DO

C     I1=0, I3>=I2>0
      ELSE IF (I1.EQ.0) THEN

C        I1=0, I3=I2>0   ---   [011] type
         IF (I2.EQ.I3) THEN
            NSPDIR=6
            DO J=1,6
               DO I=1,3
                  IWKDIR(I,J)=I2
                  IF (I.EQ.J.OR.J-I.EQ.3) IWKDIR(I,J)=0
                  IWKDIR(1,6)=-I2
                  IWKDIR(2,4)=-I2
                  IWKDIR(3,5)=-I2
               END DO
            END DO

C        I1=0, I3>I2>0   ---   [012] type
         ELSE
            NSPDIR=12
            CALL LINE1 (I2, I3, IWKDIR(1,1), 1)
            CALL LINE1 (I3, I2, IWKDIR(1,3), 1)
            CALL LINE1 (I2, I3, IWKDIR(1,5), 2)
            CALL LINE1 (I3, I2, IWKDIR(1,7), 2)
            CALL LINE1 (I2, I3, IWKDIR(1,9), 3)
            CALL LINE1 (I3, I2, IWKDIR(1,11), 3)

         END IF

C     I1=I2=I3>0   ---   [111] type
      ELSE IF (I1.EQ.I3) THEN
         NSPDIR=4
         CALL LINE (I1, I1, I1, IWKDIR)

C     I3>I2=I1>0   ---   [112] type
      ELSE IF (I1.EQ.I2) THEN
         NSPDIR=12
         CALL LINE (I1, I1, I3, IWKDIR(1,1))
         CALL LINE (I1, I3, I1, IWKDIR(1,5))
         CALL LINE (I3, I1, I1, IWKDIR(1,9))

C     I3=I2>I1>0   ---   [122] type
      ELSE IF (I2.EQ.I3) THEN
         NSPDIR=12
         CALL LINE (I1, I2, I2, IWKDIR(1,1))
         CALL LINE (I2, I1, I2, IWKDIR(1,5))
         CALL LINE (I2, I2, I1, IWKDIR(1,9))

C     I3>I2>I1>0   ---   [123] type
      ELSE
         NSPDIR=24
         CALL LINE (I1, I2, I3, IWKDIR(1,1))
         CALL LINE (I3, I1, I2, IWKDIR(1,5))
         CALL LINE (I2, I3, I1, IWKDIR(1,9))
         CALL LINE (I1, I3, I2, IWKDIR(1,13))
         CALL LINE (I2, I1, I3, IWKDIR(1,17))
         CALL LINE (I3, I2, I1, IWKDIR(1,21))

      END IF

C-----  Generating all possible slip planes in this set
C
C       Denote the normal to slip plane by (pqr).  J1 is the minimum of
C     the absolute value of p, q and r, J3 is the maximum and J2 is the
C     mode, e.g. (1 -2 1), J1=1, J2=1 and J3=2.  J1<=J2<=J3.

      J1=MIN(IABS(ISPNOR(1)),IABS(ISPNOR(2)),IABS(ISPNOR(3)))
      J3=MAX(IABS(ISPNOR(1)),IABS(ISPNOR(2)),IABS(ISPNOR(3)))
      J2=IABS(ISPNOR(1))+IABS(ISPNOR(2))+IABS(ISPNOR(3))-J1-J3

      RMONOR=SQRT(FLOAT(J1*J1+J2*J2+J3*J3))

      IF (J3.EQ.0) THEN 
         WRITE (6,*) '***ERROR - slip plane is [000]'
         STOP

C     (001) type
      ELSE IF (J2.EQ.0) THEN
         NSPNOR=3
         DO J=1,3
            DO I=1,3
               IWKNOR(I,J)=0
               IF (I.EQ.J) IWKNOR(I,J)=J3
            END DO
         END DO

      ELSE IF (J1.EQ.0) THEN

C     (011) type
         IF (J2.EQ.J3) THEN
            NSPNOR=6
            DO J=1,6
               DO I=1,3
                  IWKNOR(I,J)=J2
                  IF (I.EQ.J.OR.J-I.EQ.3) IWKNOR(I,J)=0
                  IWKNOR(1,6)=-J2
                  IWKNOR(2,4)=-J2
                  IWKNOR(3,5)=-J2
               END DO
            END DO

C     (012) type
         ELSE
            NSPNOR=12
            CALL LINE1 (J2, J3, IWKNOR(1,1), 1)
            CALL LINE1 (J3, J2, IWKNOR(1,3), 1)
            CALL LINE1 (J2, J3, IWKNOR(1,5), 2)
            CALL LINE1 (J3, J2, IWKNOR(1,7), 2)
            CALL LINE1 (J2, J3, IWKNOR(1,9), 3)
            CALL LINE1 (J3, J2, IWKNOR(1,11), 3)

         END IF

C     (111) type
      ELSE IF (J1.EQ.J3) THEN
         NSPNOR=4
         CALL LINE (J1, J1, J1, IWKNOR)

C     (112) type
      ELSE IF (J1.EQ.J2) THEN
         NSPNOR=12
         CALL LINE (J1, J1, J3, IWKNOR(1,1))
         CALL LINE (J1, J3, J1, IWKNOR(1,5))
         CALL LINE (J3, J1, J1, IWKNOR(1,9))

C     (122) type
      ELSE IF (J2.EQ.J3) THEN
         NSPNOR=12
         CALL LINE (J1, J2, J2, IWKNOR(1,1))
         CALL LINE (J2, J1, J2, IWKNOR(1,5))
         CALL LINE (J2, J2, J1, IWKNOR(1,9))

C     (123) type
      ELSE
         NSPNOR=24
         CALL LINE (J1, J2, J3, IWKNOR(1,1))
         CALL LINE (J3, J1, J2, IWKNOR(1,5))
         CALL LINE (J2, J3, J1, IWKNOR(1,9))
         CALL LINE (J1, J3, J2, IWKNOR(1,13))
         CALL LINE (J2, J1, J3, IWKNOR(1,17))
         CALL LINE (J3, J2, J1, IWKNOR(1,21))

      END IF

C-----  Generating all slip systems in this set
C
C-----  Unit vectors in slip directions: SLPDIR, and unit normals to 
C     slip planes: SLPNOR in local cubic crystal system
C

      DO J=1,NSPNOR
         DO I=1,NSPDIR

            IDOT=0
            DO K=1,3
               IDOT=IDOT+IWKDIR(K,I)*IWKNOR(K,J)
            END DO

            IF (IDOT.EQ.0) THEN
               NSLIP=NSLIP+1
               DO K=1,3
                  SLPDIR(K,NSLIP)=IWKDIR(K,I)/RMODIR
                  SLPNOR(K,NSLIP)=IWKNOR(K,J)/RMONOR
               END DO

            END IF

         END DO
      END DO

      IF (NSLIP.EQ.0) THEN
         WRITE (6,*) 
     *      'There is no slip direction normal to the slip planes!'
         STOP

      ELSE

C-----  Unit vectors in slip directions: SLPDIR, and unit normals to 
C     slip planes: SLPNOR in global system
C
         DO J=1,NSLIP
            DO I=1,3
               TERM(I)=0.
               DO K=1,3
                  TERM(I)=TERM(I)+ROTATE(I,K)*SLPDIR(K,J)
               END DO
            END DO
            DO I=1,3
               SLPDIR(I,J)=TERM(I)
            END DO

            DO I=1,3
               TERM(I)=0.
               DO K=1,3
                  TERM(I)=TERM(I)+ROTATE(I,K)*SLPNOR(K,J)
               END DO
            END DO
            DO I=1,3
               SLPNOR(I,J)=TERM(I)
            END DO
         END DO

      END IF

      RETURN
      END


C----------------------------------


           SUBROUTINE LINE (I1, I2, I3, IARRAY)

C-----  Generating all possible slip directions <lmn> (or slip planes 
C     {lmn}) for a cubic crystal, where l,m,n are not zeros.

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
           DIMENSION IARRAY(3,4)

           DO J=1,4
              IARRAY(1,J)=I1
              IARRAY(2,J)=I2
              IARRAY(3,J)=I3
           END DO

           DO I=1,3
              DO J=1,4
                 IF (J.EQ.I+1) IARRAY(I,J)=-IARRAY(I,J)
              END DO
           END DO

           RETURN
           END


C-----------------------------------


           SUBROUTINE LINE1 (J1, J2, IARRAY, ID)

C-----  Generating all possible slip directions <0mn> (or slip planes 
C     {0mn}) for a cubic crystal, where m,n are not zeros and m does 
C     not equal n.

C-----  Use single precision on cray
C
           IMPLICIT REAL*8 (A-H,O-Z)
           DIMENSION IARRAY(3,2)

           IARRAY(ID,1)=0
           IARRAY(ID,2)=0

           ID1=ID+1
           IF (ID1.GT.3) ID1=ID1-3
           IARRAY(ID1,1)=J1
           IARRAY(ID1,2)=J1

           ID2=ID+2
           IF (ID2.GT.3) ID2=ID2-3
           IARRAY(ID2,1)=J2
           IARRAY(ID2,2)=-J2
  
           RETURN
           END


C======================================================================

      SUBROUTINE ROTATION (PROP, ROTATE)

C-----  This subroutine calculates the rotation matrix, i.e. the 
C     direction cosines of cubic crystal [100], [010] and [001] 
C     directions in global system

C-----  The rotation matrix is stored in the array ROTATE.

C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION PROP(16), ROTATE(3,3), TERM1(3,3), TERM2(3,3), INDX(3) 

C-----  Subroutines:
C
C       CROSS  -- cross product of two vectors
C
C       LUDCMP -- LU decomposition
C
C       LUBKSB -- linear equation solver based on LU decomposition 
C                 method (must call LUDCMP first)


C-----  PROP -- constants characterizing the crystal orientation 
C               (INPUT)
C
C            PROP(1) - PROP(3) -- direction of the first vector in 
C                                 local cubic crystal system
C            PROP(4) - PROP(6) -- direction of the first vector in 
C                                 global system
C
C            PROP(9) - PROP(11)-- direction of the second vector in 
C                                 local cubic crystal system
C            PROP(12)- PROP(14)-- direction of the second vector in 
C                                 global system
C
C-----  ROTATE -- rotation matrix (OUTPUT):
C
C            ROTATE(i,1) -- direction cosines of direction [1 0 0] in 
C                           local cubic crystal system
C            ROTATE(i,2) -- direction cosines of direction [0 1 0] in 
C                           local cubic crystal system
C            ROTATE(i,3) -- direction cosines of direction [0 0 1] in 
C                           local cubic crystal system

C-----  local matrix: TERM1
      CALL CROSS (PROP(1), PROP(9), TERM1, ANGLE1)

C-----  LU decomposition of TERM1
      CALL LUDCMP (TERM1, 3, 3, INDX, DCMP)

C-----  inverse matrix of TERM1: TERM2
      DO J=1,3
         DO I=1,3
            IF (I.EQ.J) THEN
               TERM2(I,J)=1.
            ELSE
               TERM2(I,J)=0.
            END IF
         END DO
      END DO

      DO J=1,3
         CALL LUBKSB (TERM1, 3, 3, INDX, TERM2(1,J))
      END DO

C-----  global matrix: TERM1
      CALL CROSS (PROP(4), PROP(12), TERM1, ANGLE2)

C-----  Check: the angle between first and second vector in local and 
C     global systems must be the same.  The relative difference must be
C     less than 0.1%.
C
      IF (ABS(ANGLE1/ANGLE2-1.).GT.0.001) THEN 
         WRITE (6,*) 
     2      '***ERROR - angles between two vectors are not the same'
         STOP
      END IF

C-----  rotation matrix: ROTATE
      DO J=1,3
         DO I=1,3
            ROTATE(I,J)=0.
            DO K=1,3
               ROTATE(I,J)=ROTATE(I,J)+TERM1(I,K)*TERM2(K,J)
            END DO
         END DO
      END DO

      RETURN
      END

C======================================================================

      SUBROUTINE CROSS (A, B, C, ANGLE)

C-----  (1) normalize vectors A and B to unit vectors
C       (2) store A, B and A*B (cross product) in C

C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(3), B(3), C(3,3)

      SUM1=SQRT(A(1)**2+A(2)**2+A(3)**2)
      SUM2=SQRT(B(1)**2+B(2)**2+B(3)**2)

      IF (SUM1.EQ.0.) THEN
         WRITE (6,*) '***ERROR - first vector is zero'
         STOP
      ELSE
         DO I=1,3
            C(I,1)=A(I)/SUM1
         END DO
      END IF

      IF (SUM2.EQ.0.) THEN
         WRITE (6,*) '***ERROR - second vector is zero'
         STOP
      ELSE
         DO I=1,3
            C(I,2)=B(I)/SUM2
         END DO
      END IF

      ANGLE=0.
      DO I=1,3
         ANGLE=ANGLE+C(I,1)*C(I,2)
      END DO
      ANGLE=ACOS(ANGLE)

      C(1,3)=C(2,1)*C(3,2)-C(3,1)*C(2,2)
      C(2,3)=C(3,1)*C(1,2)-C(1,1)*C(3,2)
      C(3,3)=C(1,1)*C(2,2)-C(2,1)*C(1,2)
      SUM3=SQRT(C(1,3)**2+C(2,3)**2+C(3,3)**2)
      IF (SUM3.LT.1.E-8) THEN
         WRITE (6,*) 
     2       '***ERROR - first and second vectors are parallel'
          STOP
       END IF

      RETURN
      END

C======================================================================

      SUBROUTINE LUDCMP (A, N, NP, INDX, D)

C-----  LU decomposition

C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (NMAX=200, TINY=1.0E-20)
      DIMENSION A(NP,NP), INDX(N), VV(NMAX)

      D=1.
      DO I=1,N
         AAMAX=0.

         DO J=1,N
            IF (ABS(A(I,J)).GT.AAMAX) AAMAX=ABS(A(I,J))
         END DO

         IF (AAMAX.EQ.0.) PAUSE 'Singular matrix.'
         VV(I)=1./AAMAX
      END DO

      DO J=1,N
         DO I=1,J-1
            SUM=A(I,J)

            DO K=1,I-1
               SUM=SUM-A(I,K)*A(K,J)
            END DO

            A(I,J)=SUM
         END DO
         AAMAX=0.

         DO I=J,N
            SUM=A(I,J)

            DO K=1,J-1
               SUM=SUM-A(I,K)*A(K,J)
            END DO

            A(I,J)=SUM
            DUM=VV(I)*ABS(SUM)
            IF (DUM.GE.AAMAX) THEN
               IMAX=I
               AAMAX=DUM
            END IF
         END DO

         IF (J.NE.IMAX) THEN
            DO K=1,N
               DUM=A(IMAX,K)
               A(IMAX,K)=A(J,K)
               A(J,K)=DUM
            END DO

            D=-D
            VV(IMAX)=VV(J)
         END IF

         INDX(J)=IMAX
         IF (A(J,J).EQ.0.) A(J,J)=TINY
         IF (J.NE.N) THEN
            DUM=1./A(J,J)
            DO I=J+1,N
               A(I,J)=A(I,J)*DUM
            END DO
         END IF

      END DO

      RETURN
      END


C======================================================================

      SUBROUTINE LUBKSB (A, N, NP, INDX, B)

C-----  Linear equation solver based on LU decomposition

C-----  Use single precision on cray
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(NP,NP), INDX(N), B(N)

      II=0
      DO I=1,N
         LL=INDX(I)
         SUM=B(LL)
         B(LL)=B(I)

         IF (II.NE.0) THEN
            DO J=II,I-1
               SUM=SUM-A(I,J)*B(J)
            END DO
         ELSE IF (SUM.NE.0.) THEN
            II=I
         END IF

         B(I)=SUM
      END DO

      DO I=N,1,-1
         SUM=B(I)

         IF (I.LT.N) THEN
            DO J=I+1,N
               SUM=SUM-A(I,J)*B(J)
            END DO
         END IF

         B(I)=SUM/A(I,I)
      END DO

      RETURN
      END
C======================================================================
      SUBROUTINE OUTPUT (STATEV,Ikinc,RESIMA,NITRTN)
    
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION STATEV(126), RESIMA(12)
    
    
      OPEN(10, FILE='G:/Leo/CAE/CPFEM/2020/Myhr/test/STATEV.txt', STATUS='UNKNOWN',ACCESS='APPEND')
      WRITE(10,*) "====== STATEV value ======"
      WRITE(10,*) "Inc:", Ikinc, "ITERATION:", NITRTN
      WRITE(10,*) "RESIDU:"
      WRITE(10,*) (RESIMA(I),I=1,12)
      WRITE(10,*) "Current stress:"
      WRITE(10,*) (STATEV(I),I=1,12)
      WRITE(10,*) "Shear strain:"
      WRITE(10,*) (STATEV(12+I),I=1,12)
      WRITE(10,*) "Tau:"
      WRITE(10,*) (STATEV(24+I),I=1,12)
      WRITE(10,*) "Cumulative shear strain(each):"
      WRITE(10,*) (STATEV(108+I),I=1,12)
      WRITE(10,*) "Cumulative shear strain(all):", STATEV(121)
      WRITE(10,*) "Rhos:", STATEV(124)
      WRITE(10,*) "Rhog:", STATEV(125)
      WRITE(10,*) "========== END =========="
      CLOSE(10)
      
      RETURN
      END