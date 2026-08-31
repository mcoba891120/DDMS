// ----------------------------------------------------------------------------------------------------
// > abaqus pytorch user material subroutine
// > ref https://pytorch.org/tutorials/recipes/torchscript_inference.html
// 
// > notation conventions					
// 		- NYE				11 22 33 12 23 13
// 		- abaqus implicit 	11 22 33 12 13 23
// 		- abaqus explicit	11 22 33 12 23 13
// > props & stateNew usage
// 		- props[0]
//			- nstt, number of hidden state variable
// 		- stateNew order
// 			- lmsc hidden state (nstt)
// 			- mean Cauchy Stress (1)
// > WARNING
// 		- shear strain components in VUMAT are tensor components (epsilon)
// 		- DO NOT use torch.sqrt in torchscript !!!! will hang on forever !
// ----------------------------------------------------------------------------------------------------
#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace std;
using namespace torch::indexing;

torch::NoGradGuard
	no_grad;																								// turn off the grad
// set DDMS_VUMAT_MODEL_ROOT to the script_module directory before running; WARNING: needs an absolute path
string
	root 		= getenv("DDMS_VUMAT_MODEL_ROOT") ? getenv("DDMS_VUMAT_MODEL_ROOT") : "",
	data_root 	= root + "/data",
	model_root 	= root + "/model";
torch::jit::Module
	ctn_data	= torch::jit::load(data_root + "/LMSC_n3-hid32-stt32-noiseAJ_container_data.pt", torch::kCPU),	// torchscript data container, e.g., data_scale, init_state
	mdl_CS 		= torch::jit::load(model_root + "/LMSC_n3-hid32-stt32-noiseAJ_module_CS.pt", torch::kCPU);		// torchscript model of Cauchy stress
torch::Tensor
	K = torch::tensor({7.4972e+10}, {torch::kFloat64});														// Bulk modulus (Pa)
auto 
	options = torch::TensorOptions()
					.dtype(torch::kFloat64)
					.layout(torch::kStrided)
					.device(torch::kCPU)
					.requires_grad(false);
torch::Tensor
	transform(torch::Tensor x, torch::Tensor x_scale){
		auto loc = x_scale.index({"...", 0});
		auto scale = x_scale.index({"...", 1});
		return x.sub(loc).div(scale);
	};
torch::Tensor
	inverse_transform(torch::Tensor x, torch::Tensor x_scale){
		auto loc = x_scale.index({"...", 0});
		auto scale = x_scale.index({"...", 1});
		return x.mul(scale).add(loc);
	};
torch::Tensor
	math_sqrt(double x){
		// WARNING: DO NOT use torch::sqrt, will hang forever
		return torch::tensor({sqrt(x)}, {torch::kFloat64});
	};
torch::Tensor
	math_dev6todev5(torch::Tensor dev6){
		auto MAP_ABQ6tom9 = torch::tensor({0,3,4,3,1,5,4,5,2}, {torch::kLong});
		auto dev5 = dev6.new_zeros({dev6.size(0), 5});									// (nblock, 5)
		auto dev33 = dev6.index({Slice(),MAP_ABQ6tom9}).reshape({-1,3,3});				// (nblock, 3, 3)
		dev5.index_put_({Slice(),0}, (dev33.index({Slice(),0,0}) - dev33.index({Slice(),1,1})).div(math_sqrt(2.)));
		dev5.index_put_({Slice(),1}, (2*dev33.index({Slice(),2,2}) - dev33.index({Slice(),0,0}) - dev33.index({Slice(),1,1})).div(math_sqrt(6.)));
		dev5.index_put_({Slice(),2}, math_sqrt(2.)*dev33.index({Slice(),0,1}));
		dev5.index_put_({Slice(),3}, math_sqrt(2.)*dev33.index({Slice(),0,2}));
		dev5.index_put_({Slice(),4}, math_sqrt(2.)*dev33.index({Slice(),1,2}));
		return dev5;
	};
torch::Tensor
	math_dev5todev6(torch::Tensor dev5){
		auto zz = dev5.index({Slice(),1})*math_sqrt(6.)/3;
		auto xx = (dev5.index({Slice(),0})*math_sqrt(2.) - (dev5.index({Slice(),1})*math_sqrt(6.)-2*zz))/2;
		auto yy = xx - dev5.index({Slice(),0})*math_sqrt(2.);
		auto xy = dev5.index({Slice(),2}) / math_sqrt(2.);
		auto xz = dev5.index({Slice(),3}) / math_sqrt(2.);
		auto yz = dev5.index({Slice(),4}) / math_sqrt(2.);

		auto dev6 = dev5.new_zeros({dev5.size(0), 6});									// (nblock, 6)
		dev6.index_put_({Slice(),0}, xx); dev6.index_put_({Slice(),1}, yy); 
		dev6.index_put_({Slice(),2}, zz); dev6.index_put_({Slice(),3}, xy); 
		dev6.index_put_({Slice(),4}, xz); dev6.index_put_({Slice(),5}, yz);
		return dev6;
	};
torch::Tensor
	math_convertImpExp(torch::Tensor m6){
		// convert between implicit & explicit notation
		auto temp4 = m6.index({"...", 4}).clone();
		auto temp5 = m6.index({"...", 5}).clone();
		m6.index_put_({"...", 4}, temp5);
		m6.index_put_({"...", 5}, temp4);
		return m6;
	};

extern "C" void vumat_(int *nblock, int *ndir, int *nshr, int *nstatev, int *nfieldv, int *nprops, int *lanneal,
		double *stepTime, double *totalTime, double *dt, char* cmname, double *coordMp, double *charLength,
		double *props, double *density, double *strainInc, double *relSpinInc,
		double *tempOld, double *stretchOld, double *defgradOld, double *fieldOld,
		double *stressOld, double *stateOld, double *enerInternOld, double *enerInelasOld,
		double *tempNew, double *stretchNew, double *defgradNew, double *fieldNew,
		double *stressNew, double *stateNew, double *enerInternNew, double *enerInelasNew)
{
	torch::Tensor
		dHSdev5, dHSdev5_norm, dHS6, dHSvol,											// Hencky strain
		CSdev5, CSdamp, CSdev6,	CS, CSmean,												// Cauchy stress
		ten_state, chi;																	// lmsc hidden state
	vector<torch::jit::IValue>
		inputs_CS;																		// vector of model input IValues
	int 
		nstt = props[0];
	
	// prepare strain 
	vector<double> vec_dstran(strainInc, strainInc + *nblock*(*ndir + *nshr));
	dHS6 = torch::from_blob(vec_dstran.data(), {*nblock, *ndir + *nshr}, options);		// (nblock, 6), {eps, eps, eps, eps, eps, eps}
	dHS6 = math_convertImpExp(dHS6);													// (nblock, 6), explicit to implicit notation (which is NN trained on)
	dHSvol = dHS6.index({Slice(), Slice(0, 3)});										// (nblock, 3)
	dHSdev5 = math_dev6todev5(dHS6);													// (nblock, 5), result is the same for dHS6/dHSdev6

	// prepare strain norm
	auto temp = dHSdev5.square().sum(-1, true);											// (nblock, 1), prepare for sqrt()
	dHSdev5_norm = torch::zeros({*nblock, 1}, {torch::kFloat64});						// (nblock, 1)
	for (int i = 0; i < *nblock; i++) {
		dHSdev5_norm.index_put_({i}, math_sqrt(temp.index({i}).item<double>()));
	}
	dHSdev5_norm = dHSdev5_norm.clamp(1e-15, 100);										// (nblock, 1)

	// prepare lmsc hidden state (chi)
	vector<double> vec_statev(stateOld, stateOld + *nblock*(*nstatev));
	ten_state = torch::from_blob(vec_statev.data(), {*nblock, *nstatev}, options);
	chi = ten_state.index({Slice(), Slice(0, nstt)});									// (nblock, nstt)

	// reshape
	dHSdev5 = dHSdev5.reshape({*nblock,1,5});											// (b, s, c)

	// prepare input "image"
	inputs_CS.push_back(dHSdev5);
	inputs_CS.push_back(dHSdev5_norm);
	inputs_CS.push_back(chi);

	// stress forward prediction
	mdl_CS.eval();
	mdl_CS.to({torch::kFloat64});
	auto pred_CS = mdl_CS.forward(inputs_CS).toTuple();
	CSdev5 = pred_CS->elements()[0].toTensor();
	chi = pred_CS->elements()[1].toTensor();											// (nblock, nstt)

	// de-reshape & de-normalize
	CSdev5 = CSdev5.reshape({*nblock,5});												// (nblock, 5)
	CSdev5 = inverse_transform(CSdev5, ctn_data.attr("CSdev5_sc").toTensor());

	// add damping term 
	auto gamma = 10;
	auto S0 = ctn_data.attr("CSdev5_sc").toTensor().index({0,1});
	CSdamp = (1/S0) * dHSdev5.reshape({*nblock, 5})/dHSdev5_norm * 
				torch::tanh(gamma * dHSdev5_norm / *dt);								// (nblock, 5)
	CSdev5 = CSdev5 + CSdamp;

	// get actual CS
	CSdev6 = math_dev5todev6(CSdev5);													// (nblock, 6)
	CSmean = ten_state.index({Slice(), Slice(nstt, nstt+1)});							// (nblock, 1)
	CSmean = CSmean + K*dHSvol.sum(-1, true);											// (nblock, 1)
	CS = CSdev6 + CSmean*torch::tensor({1,1,1,0,0,0});									// (nblock, 6)
	CS = math_convertImpExp(CS).flatten();												// (nblock*6), implicit to explicit

	// update stateNew
	ten_state.index_put_({Slice(), Slice(0, nstt+1)}, torch::cat({chi, CSmean}, -1));	// (nblock, depvar)
	ten_state = ten_state.flatten();													// (nblock*depvar)
	for (int i = 0; i < ten_state.size(0); i++){
		stateNew[i] = ten_state.index({i}).item<double>();
	}

	// update stress
	for(int i = 0; i < *nblock*(*ndir + *nshr); i++){	
		stressNew[i] = CS.index({i}).div(1e6).item<double>(); 							// Pa -> MPa
	}


	// >----------------------------------------------------------------------------------------------------
	// for (int n = 0; n < *nblock; n++) {
	// 	cout << "---------- n ----------" << endl << n << endl;
	// 	torch::Tensor
	// 		dHSdev5, dHSdev5_norm, dHS6, dHSvol,											// Hencky strain
	// 		CSdev5, CSdamp, CSdev6,	CS, CSmean,												// Cauchy stress
	// 		ten_state, chi;																	// lmsc hidden state
	// 	vector<torch::jit::IValue>
	// 		inputs_CS;																		// vector of model input IValues
	// 	int 
	// 		nstt = props[0];


	// 	// prepare strain 
	// 	vector<double> vec_dstran(strainInc + n*(*ndir + *nshr), strainInc + (n + 1)*(*ndir + *nshr));
	// 	dHS6 = torch::from_blob(vec_dstran.data(), {1, *ndir + *nshr}, options);		// (nblock, 6), {eps, eps, eps, eps, eps, eps}
	// 	dHS6 = math_convertImpExp(dHS6);													// (nblock, 6), explicit to implicit notation (which is NN trained on)
	// 	dHSvol = dHS6.index({Slice(), Slice(0, 3)});										// (nblock, 3)
	// 	dHSdev5 = math_dev6todev5(dHS6);													// (nblock, 5), result is the same for dHS6/dHSdev6
		
	// 	// prepare strain norm
	// 	auto temp = dHSdev5.square().sum(-1, true);											// (nblock, 1), prepare for sqrt()
	// 	dHSdev5_norm = torch::zeros({1, 1}, {torch::kFloat64});						// (nblock, 1)
	// 	for (int i = 0; i < 1; i++) {
	// 		dHSdev5_norm.index_put_({i}, math_sqrt(temp.index({i}).item<double>()));
	// 	}
	// 	dHSdev5_norm = dHSdev5_norm.clamp(1e-15, 100);										// (nblock, 1)

	// 	// prepare lmsc hidden state (chi)
	// 	vector<double> vec_statev(stateOld + n*(*nstatev), stateOld + (n + 1)*(*nstatev));
	// 	ten_state = torch::from_blob(vec_statev.data(), {1, *nstatev}, options);
	// 	chi = ten_state.index({Slice(), Slice(0, nstt)});									// (nblock, nstt)

	// 	// reshape
	// 	dHSdev5 = dHSdev5.reshape({1,1,5});											// (b, s, c)

	// 	// prepare input "image"
	// 	inputs_CS.push_back(dHSdev5);
	// 	inputs_CS.push_back(dHSdev5_norm);
	// 	inputs_CS.push_back(chi);

	// 	// stress forward prediction
	// 	mdl_CS.eval();
	// 	mdl_CS.to({torch::kFloat64});
	// 	auto pred_CS = mdl_CS.forward(inputs_CS).toTuple();
	// 	CSdev5 = pred_CS->elements()[0].toTensor();
	// 	chi = pred_CS->elements()[1].toTensor();											// (nblock, nstt)

	// 	// de-reshape & de-normalize
	// 	CSdev5 = CSdev5.reshape({1,5});												// (nblock, 5)
	// 	CSdev5 = inverse_transform(CSdev5, ctn_data.attr("CSdev5_sc").toTensor());

	// 	// add damping term 
	// 	auto gamma = 10;
	// 	auto S0 = ctn_data.attr("CSdev5_sc").toTensor().index({0,1});
	// 	CSdamp = (1/S0) * dHSdev5.reshape({1, 5})/dHSdev5_norm * 
	// 				torch::tanh(gamma * dHSdev5_norm / *dt);								// (nblock, 5)
	// 	CSdev5 = CSdev5 + CSdamp;

	// 	// get actual CS
	// 	CSdev6 = math_dev5todev6(CSdev5);													// (nblock, 6)
	// 	CSmean = ten_state.index({Slice(), Slice(nstt, nstt+1)});							// (nblock, 1)
	// 	CSmean = CSmean + K*dHSvol.sum(-1, true);											// (nblock, 1)
	// 	CS = CSdev6 + CSmean*torch::tensor({1,1,1,0,0,0});									// (nblock, 6)
	// 	CS = math_convertImpExp(CS).flatten();												// (nblock*6), implicit to explicit

		

	// 	cout << "########## CS0 ##########" << endl << CS.index({0}).div(1e6) << " ----------- " << stressNew[n*(*ndir + *nshr)+0] << endl;
	// 	cout << "########## CS1 ##########" << endl << CS.index({1}).div(1e6) << " ----------- " << stressNew[n*(*ndir + *nshr)+1] << endl;
	// 	cout << "########## CS2 ##########" << endl << CS.index({2}).div(1e6) << " ----------- " << stressNew[n*(*ndir + *nshr)+2] << endl;
	// 	cout << "########## CS3 ##########" << endl << CS.index({3}).div(1e6) << " ----------- " << stressNew[n*(*ndir + *nshr)+3] << endl;
	// 	cout << "########## CS4 ##########" << endl << CS.index({4}).div(1e6) << " ----------- " << stressNew[n*(*ndir + *nshr)+4] << endl;
	// 	cout << "########## CS5 ##########" << endl << CS.index({5}).div(1e6) << " ----------- " << stressNew[n*(*ndir + *nshr)+5] << endl;

	// }
	// >----------------------------------------------------------------------------------------------------


}
