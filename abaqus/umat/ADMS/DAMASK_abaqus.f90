! Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
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
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Koen Janssens, Paul Scherrer Institut
!> @author Arun Prakash, Fraunhofer IWM
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief interfaces DAMASK with Abaqus/Standard
!> @details put the included file abaqus_v6.env in either your home or model directory, 
!> it is a minimum Abaqus environment file  containing all changes necessary to use the 
!> DAMASK subroutine (see Abaqus documentation for more information on the use of abaqus_v6.env)
!> @details  Abaqus subroutines used:
!> @details   - UMAT
!> @details   - DFLUX
!--------------------------------------------------------------------------------------------------
#define Abaqus



! module DAMASK_interface

!  implicit none
!  private
!  character(len=4), dimension(2),  parameter, public :: INPUTFILEEXTENSION = ['.pes','.inp']
!  character(len=4),                parameter, public :: LOGFILEEXTENSION   =  '.log'
 
!  public :: &
!   DAMASK_interface_init, &
!   getSolverJobName

! contains

! !--------------------------------------------------------------------------------------------------
! !> @brief reports and sets working directory
! !--------------------------------------------------------------------------------------------------
! subroutine DAMASK_interface_init
! #if __INTEL_COMPILER >= 1800
!  use, intrinsic :: iso_fortran_env, only: &
!    compiler_version, &
!    compiler_options
! #endif
! !  use ifport, only: &
! !    CHDIR
 
!  implicit none
!  integer, dimension(8) :: &
!    dateAndTime
!  integer :: lenOutDir,ierr
!  character(len=256) :: wdl

!  write(6,'(/,a)') ' <<<+-  DAMASK_abaqus init -+>>>'

!  write(6,'(/,a)') ' Roters et al., Computational Materials Science 158:420–478, 2019'
!  write(6,'(a)')   ' https://doi.org/10.1016/j.commatsci.2018.04.030'

!  write(6,'(/,a)') ' Version: '//DAMASKVERSION

! ! ! https://github.com/jeffhammond/HPCInfo/blob/master/docs/Preprocessor-Macros.md
! ! #if __INTEL_COMPILER >= 1800
! !  write(6,'(/,a)') ' Compiled with: '//compiler_version()
! !  write(6,'(a)')   ' Compiler options: '//compiler_options()
! ! #else
! !  write(6,'(/,a,i4.4,a,i8.8)') ' Compiled with Intel fortran version :', __INTEL_COMPILER,&
! !                                                       ', build date :', __INTEL_COMPILER_BUILD_DATE
! ! #endif

! !  write(6,'(/,a)') ' Compiled on: '//__DATE__//' at '//__TIME__

! !  call date_and_time(values = dateAndTime)
! !  write(6,'(/,a,2(i2.2,a),i4.4)') ' Date: ',dateAndTime(3),'/',dateAndTime(2),'/', dateAndTime(1)
! !  write(6,'(a,2(i2.2,a),i2.2)')   ' Time: ',dateAndTime(5),':', dateAndTime(6),':', dateAndTime(7)

! !  call getoutdir(wd, lenOutDir)
! !  ierr = CHDIR(wd)
! !  if (ierr /= 0) then
! !    write(6,'(a20,a,a16)') ' working directory "',trim(wd),'" does not exist'
! !    call quit(1)
! !  endif

! end subroutine DAMASK_interface_init


! !--------------------------------------------------------------------------------------------------
! !> @brief using Abaqus/Standard function to get solver job name
! !--------------------------------------------------------------------------------------------------
! character(1024) function getSolverJobName()
 
!  implicit none
!  integer :: lenJobName

!  getSolverJobName=''
!  call getJobName(getSolverJobName, lenJobName)

! end function getSolverJobName


! end module DAMASK_interface
 





!--------------------------------------------------------------------------------------------------
!> @brief This is the Abaqus std user subroutine for defining material behavior
!--------------------------------------------------------------------------------------------------
! #include "commercialFEM_fileList.f90"  ! resolve via compiler -I include path
#include "petsc/finclude/petscsys.h"

subroutine UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,&
                RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,&
                TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,&
                NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,CELENT,&
                DFGRD0,DFGRD1,NOEL,NPT,KSLAY,KSPT,KSTEP,KINC)
  use PETScsys
  use prec
  use parallelization
  use DAMASK_interface
  use IO
  use config
  use math
  use CPFEM2
  use material
  use spectral_utilities
  use grid_mechanical_spectral_basic
  use grid_mechanical_spectral_polarisation
  use grid_mechanical_FEM
  use grid_damage_spectral
  use grid_thermal_spectral
  use results

  implicit none

!--------------------------------------------------------------------------------------------------
! umat variables
  integer(pInt),                       intent(in) :: &
    nDi, &                                                                                           !< Number of direct stress components at this point
    nShr, &                                                                                          !< Number of engineering shear stress components at this point
    nTens, &                                                                                         !< Size of the stress or strain component array (NDI + NSHR)
    nStatV, &                                                                                        !< Number of solution-dependent state variables
    nProps, &                                                                                        !< User-defined number of material constants
    noEl, &                                                                                          !< element number
    nPt,&                                                                                            !< integration point number
    kSlay, &                                                                                         !< layer number (shell elements etc.)
    kSpt, &                                                                                          !< section point within the current layer
    kStep, &                                                                                         !< step number
    kInc                                                                                             !< increment number
  character(len=80),                   intent(in) :: &
    cmname                                                                                           !< uses-specified material name, left justified
  real(pReal),                         intent(in) :: &
    DTIME, &
    TEMP, &
    DTEMP, &
    CELENT
  real(pReal), dimension(1),           intent(in) :: & 
    PREDEF, & 
    DPRED
  real(pReal), dimension(2),           intent(in) :: &
    TIME                                                                                             !< step time/total time at beginning of the current increment
  real(pReal), dimension(3),           intent(in) :: &
    COORDS
  real(pReal), dimension(nTens),       intent(in) :: &
    STRAN, &                                                                                         !< total strains at beginning of the increment
    DSTRAN                                                                                           !< strain increments
  real(pReal), dimension(nProps),      intent(in) :: &
    PROPS
  real(pReal), dimension(3,3),         intent(in) :: &
    DROT, &                                                                                          !< rotation increment matrix
    DFGRD0, &                                                                                        !< F at beginning of increment
    DFGRD1                                                                                           !< F at end of increment
  real(pReal),                         intent(inout) :: &                                                             
    PNEWDT, &                                                                                        !< ratio of suggested new time increment
    SSE, &                                                                                           !< specific elastic strain engergy
    SPD, &                                                                                           !< specific plastic dissipation
    SCD, &                                                                                           !< specific creep dissipation
    RPL, &                                                                                           !< volumetric heat generation per unit time at the end of the increment 
    DRPLDT                                                                                           !< varation of RPL with respect to the temperature
  real(pReal), dimension(nTens),       intent(inout) :: &
    STRESS                                                                                           !< stress tensor at the beginning of the increment, needs to be updated
  real(pReal), dimension(nStatV),      intent(inout) :: &
    STATEV                                                                                           !< solution-dependent state variables
  real(pReal), dimension(nTens),       intent(out) :: &
    DDSDDT, &
    DRPLDE
  real(pReal), dimension(nTens,nTens), intent(out) :: &
    DDSDDE                                                                                          !< Jacobian matrix of the constitutive model



! !--------------------------------------------------------------------------------------------------
! ! DAMASK_grid variables
!   type :: tLoadCase
!     type(rotation)           :: rot                                                                 !< rotation of BC
!     type(tBoundaryCondition) :: stress, &                                                           !< stress BC
!                                 deformation                                                         !< deformation BC (dot_F, F, or L)
!     real(pReal) ::              t, &                                                                !< length of increment
!                                 r                                                                   !< ratio of geometric progression
!     integer ::                  N, &                                                                !< number of increments
!                                 f_out, &                                                            !< frequency of result writes
!                                 f_restart                                                           !< frequency of restart writes
!     logical ::                  estimate_rate                                                       !< follow trajectory of former loadcase
!   end type tLoadCase

!   integer(kind(FIELD_UNDEFINED_ID)), allocatable :: ID(:)

! !--------------------------------------------------------------------------------------------------
! ! loop variables, convergence etc.
!   real(pReal), dimension(3,3), parameter :: &
!     ones  = 1.0_pReal, &
!     zeros = 0.0_pReal
!   integer, parameter :: &
!     subStepFactor = 2                                                                               !< for each substep, divide the last time increment by 2.0
!   real(pReal) :: &
!     T_0 = 300.0_pReal, &
!     time = 0.0_pReal, &                                                                             !< elapsed time
!     time0 = 0.0_pReal, &                                                                            !< begin of interval
!     timeinc = 1.0_pReal, &                                                                          !< current time interval
!     timeIncOld = 0.0_pReal, &                                                                       !< previous time interval
!     remainingLoadCaseTime = 0.0_pReal                                                               !< remaining time of current load case
!   logical :: &
!     guess, &                                                                                        !< guess along former trajectory
!     stagIterate, &
!     cutBack = .false.,&
!     signal
!   integer :: &
!     i, j, m, field, &
!     errorID = 0, &
!     ierr,&
!     cutBackLevel = 0, &                                                                             !< cut back level \f$ t = \frac{t_{inc}}{2^l} \f$
!     stepFraction = 0, &                                                                             !< fraction of current time interval
!     l = 0, &                                                                                        !< current load case
!     inc, &                                                                                          !< current increment in current load case
!     totalIncsCounter = 0, &                                                                         !< total # of increments
!     statUnit = 0, &                                                                                 !< file unit for statistics output
!     stagIter, &
!     nActiveFields = 0, &
!     maxCutBack, &                                                                                   !< max number of cut backs
!     stagItMax                                                                                       !< max number of field level staggered iterations
!   character(len=pStringLen) :: &
!     incInfo

!   type(tLoadCase), allocatable, dimension(:) :: loadCases                                           !< array of all load cases
!   type(tSolutionState), allocatable, dimension(:) :: solres
!   procedure(grid_mechanical_spectral_basic_init), pointer :: &
!     mechanical_init
!   procedure(grid_mechanical_spectral_basic_forward), pointer :: &
!     mechanical_forward
!   procedure(grid_mechanical_spectral_basic_solution), pointer :: &
!     mechanical_solution
!   procedure(grid_mechanical_spectral_basic_updateCoords), pointer :: &
!     mechanical_updateCoords
!   procedure(grid_mechanical_spectral_basic_restartWrite), pointer :: &
!     mechanical_restartWrite

!   external :: &
!     quit
!   class (tNode), pointer :: &
!     num_grid, &
!     config_load, &
!     load_steps, &
!     load_step, &
!     solver, &
!     initial_conditions, &
!     thermal, &
!     step_bc, &
!     step_mech, &
!     step_discretization

!   real(pReal) :: temperature                                                                         ! temp by Abaqus is intent(in)
!   real(pReal), dimension(6) ::   stress_h
!   real(pReal), dimension(6,6) :: ddsdde_h
!   integer(pInt) :: computationMode, i, cp_en
!   logical :: cutBack
 
!--------------------------------------------------------------------------------------------------
! debuging
  write(6,'(/,a)') ' <<<+-  UMAT running -+>>>'

!--------------------------------------------------------------------------------------------------
! init DAMASK (all modules)

  call CPFEM_initAll
  print'(/,a)',   ' <<<+-  DAMASK_grid init  -+>>>'; flush(IO_STDOUT)

  print*, 'P. Shanthraj et al., Handbook of Mechanics of Materials, 2019'
  print*, 'https://doi.org/10.1007/978-981-10-6855-3_80'



end subroutine UMAT


! !--------------------------------------------------------------------------------------------------
! !> @brief calls the exit function of Abaqus/Standard
! !--------------------------------------------------------------------------------------------------
! subroutine quit(DAMASK_error)
!  use prec, only: &
!    pInt
 
!  implicit none
!  integer(pInt) :: DAMASK_error

!  flush(6)
!  call xit

! end subroutine quit
