## CPFEM About
- Crystal plasticity (CP) theory is used to describe material behavior in plastic deformation. In our current cases, we focus on aluminum alloys because of the superior **lightweight** and **high-strength** properties, which are desired in modern electrical car.
- We have established a numerical framework using **finite element method (FEM)** as the solver to material plastic behavior, and this framework is implemented in Abaqus user-defined material subroutine (UMAT). The crystal plasticity UMAT is written in Fortran originally by [prof. 黃永剛](http://www.columbia.edu/~jk2079/Kysar_Research_Laboratory/Single_Crystal_UMAT.html). [2]
- Our CPFEM model has successfully captured the uniaxial tensile stress-strain response of AA6111 aluminum alloy (Al-Mg-Si), including a yield strength model considering **grain size (Hall-Petch)**, **solid solution**, and **precipitation** strengthening; and a work hardening model accounting for **dislocation** hardening. [1,5,6]

## CPFEM workflow [1]
- obtain experiments data from 
	- transmission electron microscopy (TEM) to get microstructure parameters such as precipitation size
	- electron backscatter diffraction (EBSD) to get material texture
- reconstruct a statistically equivalent representative volume element (RVE) with respect to the experimental material texture (done by DREAM.3d)
- assign the microstructure parameters to our constitutive model to calculate critical strengthening parameters such as solid solution, precipitation strength contribution, etc. (done by python)
- start uniaxial tensile simulation on the RVE with the critical parameters (done by Abaqus UMAT)
- postprocessing to extract the predicted stress-strain response (done by python)

## CPFEM TODO
- To establish a comprehensive temperature-dependent contitutive model for both AA6111(Al-Mg-Si) and AA7075(Al-Mg-Zn) aluminum alloys.

## CPFFT About
- Due to the limitation of computational resources, the simulation is done in mesoscale (tens of grains) with the assumption that it is representative to the macroscale material behaviors.
- [DAMASK](https://damask.mpie.de/) is a material simulation package for modeling crystal plasticity in which a **spectral solver** is well developed to solve material plastic behavior with fast fourier transform (FFT). [4]
- With high computational efficiency, full-field simulation for a macroscale material component is feasible. [3]


## CPFFT TODO
- To implement our cutomized contitutive model in DAMASK spectral solver.
- To couple every material point in macroscale with mesoscale RVE.

---
## References
0. [codes](https://github.com/KyleChien/crystal-plasticity)
> the link is set as private repository in github, pls contact me if u need :)
1. thesis from Yi Liang Cheng, the core of the CPFEM model
> [Dislocation density enhaced crystal plasticity model for precipitation hardening of aluminum alloys](https://drive.google.com/open?id=1b6c3Luhv0Y3L8qBRNjwyo-PF2-yu8Wg2&authuser=kylechien0%40caece.net&usp=drive_fs)
2. explanation to crystal plasticity theory and tutorial for wirtting Abaqus UMAT code in fortran
> [實現自己的材料庫－Abaqus UMAT於計算力學的應用](https://drive.google.com/file/d/1n5xPHgHCBaBREtN2vPnCQT0_Wksaqb2y/view?usp=sharing)
3. coupling mesoscale and macroscale; CPFFT related; our goal at current state
> [DAMASK Intern J Plasticity 2020 coupling of CP spectral RVE solver with FEM for large strain forming simulation](https://drive.google.com/file/d/1895HPPzTXxoolY5Wmdu8TnXVnEtrv5jH/view?usp=sharing)
4. DAMASK paper
> [DAMASK – The Düsseldorf Advanced Material Simulation Kit for modeling multi-physics crystal plasticity, thermal, and damage phenomena from the single crystal up to the component scale](https://drive.google.com/file/d/1P7Lv4S5UKVQ9WXodD4WszcGSINsdfoJV/view?usp=sharing)
5. description about the yield stress model we used in AA6111
> [A new crystal plasticity constitutive model for simulating precipitation hardenable aluminum alloys](https://drive.google.com/open?id=1glcRBsMMq_XYF8cS5qNaX77PinhDGJJG&authuser=kylechien0%40caece.net&usp=drive_fs)
6. description about the work hardening model we used in AA6111, and material behavior with various strain rate and temperature
> [A Combined Precipitation Yield Stress and Work Hardening Model for Al Mg Si Alloys Incorporating the Effects of Strain Rate and Temperature](https://drive.google.com/file/d/1TBbZsrq-nMk6PGIMPsJzKP0yoJ8v7olN/view?usp=sharing)


