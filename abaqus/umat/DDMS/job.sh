#!/bin/bash
#SBATCH --job-name="SEcomplex"
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

# export libtorch path; set LIBTORCH_DIR / LIBTORCHSCATTER_DIR / LIBTORCHSPARSE_DIR before running
libtorch="${LIBTORCH_DIR}"
libtorchscatter="${LIBTORCHSCATTER_DIR}"
libtorchsparse="${LIBTORCHSPARSE_DIR}"
export LD_LIBRARY_PATH=${libtorch}/lib:${libtorchscatter}/lib64:${libtorchsparse}/lib64:$LD_LIBRARY_PATH

user="vumat"
inp="SEcomplex"

if [ "$1" == "abqmake" ] ; then
	# compile abaqus share library
	echo "<<< compiling ${user}.cpp to abaqus share library... >>>"
	abaqus make library=${user}.cpp -dir ./abqlib
fi

if [ "$1" == "" ] || [ "$1" == "make" ] ; then
	# compile umat.cpp -> umat.o
	echo "<<< compiling ${user}.cpp... >>>"
	g++ -c -O3 -std=c++17 -fPIC \
		-ltorch_cpu -lc10 -lpthread \
		-D_GLIBCXX_USE_CXX11_ABI=0 \
		-o ${user}.o ${user}.cpp\
		-I${libtorch}/include/torch/csrc/api/include \
		-I${libtorch}/include \
		-I${libtorchscatter}/include \
		-I${libtorchsparse}/include \
		-L${libtorch}/lib \
		-L${libtorchscatter}/lib64 \
		-L${libtorchsparse}/lib64
fi

if [ "$1" == "" ] || [ "$1" == "run" ] ; then
	echo "<<< running job ${inp} with user ${user}.o... >>>"
	abaqus job=${inp} input=${inp}.inp user=${user}.o \
		cpus=$SLURM_NTASKS mp_mode=thread scratch=. interactive
fi

if [ "$1" == "" ] || [ "$1" == "output" ] ; then
	echo "<<< postprocessing... >>>"
	abaqus viewer noGUI=output.py
fi

sbatch_post.sh