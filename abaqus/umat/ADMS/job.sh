#!/bin/bash
#SBATCH --job-name="ADMS"
#SBATCH --partition=cpu-2g
#SBATCH --ntasks=1
###SBATCH --time=1-0:0
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
#SBATCH --chdir=.
###SBATCH --test-only

sbatch_pre.sh
module load opt gcc compiler/2021.2.0 abaqus/2019
unset SLURM_GTIDS

# export DAMASK path; set DAMASK_BUILD_DIR before running this script
damask="${DAMASK_BUILD_DIR}"
# export LD_LIBRARY_PATH=${damask}/src:${hdf5}/lib:$LD_LIBRARY_PATH

user="DAMASK_abaqus"
inp="SEcomplex"

if [ "$1" == "makemod" ] ; then
	# compile dependencies
	for mod in ${damask}/src/*.f90
	do 
		mod_name=$(basename "${mod}")
		echo "<<< compiling ${mod_name}... >>>"		
		
		if [ ${mod_name} == "commercialFEM_fileList.f90" ] ; then 
			continue
		fi
		
		gfortran -c -fPIC -ffree-form -O3 -cpp -fopenmp \
				-ffp-contract=fast -fno-range-check \
				-fimplicit-none -std=gnu -fdefault-real-8 \
				-ffree-line-length-none -DDAMASKVERSION=\"0\" \
				-o ${damask}/lib/${mod_name}.o ${mod} \
				-I${damask}/include \
				-I${PETSC_DIR}/include \
				-I${PETSC_DIR}/${PETSC_ARCH}/include \
				-J${damask}/include
	done

fi

if [ "$1" == "" ] || [ "$1" == "make" ] ; then
	# compile umat.f90 -> umat.o
	echo "<<< compiling ${user}.f90... >>>"		
	gfortran -c -fPIC -ffree-form -O3 -cpp -fopenmp \
			-ffp-contract=fast -fno-range-check \
			-fimplicit-none -std=gnu -fdefault-real-8 \
			-ffree-line-length-none -DDAMASKVERSION=\"0\" \
			-o ${user}.o ${user}.f90 \
			-I${damask}/src \
			-I${PETSC_DIR}/include \
			-I${PETSC_DIR}/${PETSC_ARCH}/include
			# -J${damask}/lib
fi

if [ "$1" == "" ] || [ "$1" == "run" ] ; then
	echo "<<< running job ${inp} with user ${user}.o... >>>"
	abaqus job=${inp} input=${inp}.inp user=${user}.o \
		cpus=$SLURM_NTASKS mp_mode=thread scratch=. interactive
fi

# if [ "$1" == "" ] || [ "$1" == "output" ] ; then
# 	echo "<<< postprocessing... >>>"
# 	abaqus viewer noGUI=output.py
# fi

sbatch_post.sh