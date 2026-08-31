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
!> @brief dummy homogenization homogenization scheme for 1 constituent per material point
!--------------------------------------------------------------------------------------------------
submodule(homogenization:mechanical) mechanical_pass

  ! region start
  type :: tParameters
    character(len=pStringLen), allocatable, dimension(:) :: &
      output
  end type tParameters

  type(tparameters), dimension(:), allocatable :: &
    param
  ! region end

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all necessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
module subroutine pass_init
  integer :: &
    ho, &
    Nmembers

  ! region start
  class (tNode), pointer :: &
    material_homogenization, &
    homog, &
    homogMech
  ! region end

  print'(/,a)', ' <<<+-  homogenization:mechanical:pass_MOD init  -+>>>'

  print'(a,i0)', ' # homogenizations: ',count(homogenization_type == HOMOGENIZATION_NONE_ID)
  flush(IO_STDOUT)

  ! region start
  material_homogenization => config_material%get('homogenization')
  allocate(param(material_homogenization%length))
  ! region end

  do ho = 1, size(homogenization_type)
    if(homogenization_type(ho) /= HOMOGENIZATION_NONE_ID) cycle

    if(homogenization_Nconstituents(ho) /= 1) &
      call IO_error(211,ext_msg='N_constituents (pass)')

    ! region start
    homog => material_homogenization%get(ho)
    homogMech => homog%get('mechanical')
    associate(prm => param(ho))
      
#if defined (__GFORTRAN__)
    prm%output = output_as1dString(homogMech)
#else
    prm%output = homogMech%get_as1dString('output',defaultVal=emptyStringArray)
#endif
    
    end associate 
    ! region end

    Nmembers = count(material_homogenizationID == ho)
    homogState(ho)%sizeState = 0
    allocate(homogState(ho)%state0(0,Nmembers))
    allocate(homogState(ho)%state (0,Nmembers))

  enddo

end subroutine pass_init

! region start
module subroutine pass_results(ho, group)
  integer,          intent(in)              :: ho
  character(len=*), intent(in)              :: group

  integer                                   :: o
  real(pReal), dimension (:,:), allocatable :: dPdF

  associate(prm => param(ho))
  outputsLoop: do o = 1,size(prm%output)
    select case(trim(prm%output(o)))

      case('dPdF')
        allocate(dPdF(81, discretization_nIPs*discretization_Nelems))
        dPdF = reshape(homogenization_dPdF, [81, discretization_nIPs*discretization_Nelems])
        call results_writeDataset(dPdF, group, trim(prm%output(o)), 'dPdF, stiffness Jacobian matrix', 'Pa')

    end select
  enddo outputsLoop
    
  end associate

end subroutine pass_results
! region end

end submodule mechanical_pass
