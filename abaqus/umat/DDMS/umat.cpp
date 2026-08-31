// ----------------------------------------------------------------------------------------------------
// > abaqus pytorch user material subroutine
// > ref https://pytorch.org/tutorials/recipes/torchscript_inference.html
// 
// > notation conventions					
// 		- NYE				11 22 33 12 23 13
// 		- abaqus implicit 	11 22 33 12 13 23
// 		- abaqus explicit	11 22 33 12 23 13
// > props & statev usage
// 		- props[0]
//			- nstt, number of hidden state variable
// 		- statev order
// 			- lmsc hidden state (nstt)
// 			- mean Cauchy Stress (1)
// 			- previous Cauchy Stress (6)
// > WARNING
// 		- shear strain components in UMAT are engineering components (gamma)
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
#include <cstdlib>

using namespace std;
using namespace torch::indexing;

// WARNING: relative path not working
// #include <filesystem>
// torch::jit::Module
// 	ctn_data	= torch::jit::load(filesystem::current_path().string() + "/container.pt"),						// torchscript data container, e.g., data_scale, init_state
// 	mdl_CS		= torch::jit::load(filesystem::current_path().string() + "/model.pt");							// torchscript model of Cauchy stress

// set DDMS_UMAT_MODEL_DIR to the directory containing container.pt & model.pt before running
string
	model_dir	= getenv("DDMS_UMAT_MODEL_DIR") ? getenv("DDMS_UMAT_MODEL_DIR") : "";
torch::jit::Module
	ctn_data	= torch::jit::load(model_dir + "/container.pt"),						// torchscript data container, e.g., data_scale, init_state
	mdl_CS		= torch::jit::load(model_dir + "/model.pt");							// torchscript model of Cauchy stress
torch::Tensor
	K = torch::tensor({7.4972e+10}, {torch::kFloat64}),															// Bulk modulus (Pa)
	I = torch::eye({6});
auto 
	options = torch::TensorOptions()
					.dtype(torch::kFloat64)
					.layout(torch::kStrided)
					.device(torch::kCPU);
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
		auto dev5 = dev6.new_zeros({5});								// (5)
		auto dev33 = dev6.index({MAP_ABQ6tom9}).reshape({3,3});			// (3, 3)
		dev5.index_put_({0}, (dev33.index({0,0}) - dev33.index({1,1})).div(math_sqrt(2.)));
		dev5.index_put_({1}, (2*dev33.index({2,2}) - dev33.index({0,0}) - dev33.index({1,1})).div(math_sqrt(6.)));
		dev5.index_put_({2}, math_sqrt(2.)*dev33.index({0,1}));
		dev5.index_put_({3}, math_sqrt(2.)*dev33.index({0,2}));
		dev5.index_put_({4}, math_sqrt(2.)*dev33.index({1,2}));
		return dev5;
	};
torch::Tensor
	math_dev5todev6(torch::Tensor dev5){
		auto zz = dev5.index({1})*math_sqrt(6.)/3;
		auto xx = (dev5.index({0})*math_sqrt(2.) - (dev5.index({1})*math_sqrt(6.)-2*zz))/2;
		auto yy = xx - dev5.index({0})*math_sqrt(2.);
		auto xy = dev5.index({2}) / math_sqrt(2.);
		auto xz = dev5.index({3}) / math_sqrt(2.);
		auto yz = dev5.index({4}) / math_sqrt(2.);

		auto dev6 = dev5.new_zeros({6});									// (6)
		dev6.index_put_({0}, xx); dev6.index_put_({1}, yy); 
		dev6.index_put_({2}, zz); dev6.index_put_({3}, xy); 
		dev6.index_put_({4}, xz); dev6.index_put_({5}, yz);
		return dev6;
	};

extern "C" void umat_(double *stress, double *statev, double *ddsdde, double *sse, double *spd,
		double *scd, double *rpl, double *ddsddt, double *drplde, double *drpldt,
		double *stran, double *dstran, double *time, double *dtime, double *temp,
		double *dtemp, double *predef, double *dpred, char *cmname, int *ndi,
		int *nshr, int *ntens, int *nstatv, double *props, int *nprops, 
		double *coords, double *drot, double *pnewdt, double *celent, double *dfgrd0, 
		double *dfgrd1, int *noel, int *npt, int *layer, int *kspt, 
		int *kstep, int *kinc, short cmname_len)
{
	torch::Tensor
		dCSdE = torch::zeros({6,6});													// stiffness
	torch::Tensor
		dHSdev5, dHSdev5_norm, dHS, dHSvol,												// Hencky strain
		CSdev5, CSdamp, CSdev6,	CS, dCS, CSmean, CSprev,										// Cauchy stress
		ten_statev, chi;																// lstm hidden state
	vector<torch::jit::IValue>
		inputs_CS;																		// vector of model input IValues
	int 
		nstt = props[0];
	// static int 
	// 	prev_kinc = -1,
	// 	newton_iter = 1;

	// prepare strain & strain norm
	vector<double> vec_dstran(dstran, dstran + 6);
	dHS = torch::from_blob(vec_dstran.data(), {6}, options).requires_grad_();			// (6, ), {eps, eps, eps, gamma, gamma, gamma}, track grad from now 
	// --------------------------------------------------
	// > WARNING DIVERGED, correct stress / correct stiffness (?)
	auto dHS_ = dHS.div(torch::tensor({1,1,1,2,2,2}));									// (6, ), {eps, eps, eps, eps, eps, eps}
	// --------------------------------------------------
	// > WARNING CONVERGED, correct stress / INCORRECT stiffness (?)
	// dHS = dHS.div(torch::tensor({1,1,1,2,2,2}));
	// auto dHS_ = dHS;
	// --------------------------------------------------
	dHSvol = dHS_.index({Slice(0, 3)});													// (3, )
	dHSdev5 = math_dev6todev5(dHS_);													// (5, ), result is the same for dHS/dHSdev6
	dHSdev5_norm = math_sqrt(dHSdev5.square().sum().item<double>()).clamp(1e-15, 100);	// (1, )

	// prepare lmsc hidden state (chi)
	vector<double> vec_statev(statev, statev + *nstatv);
	ten_statev = torch::from_blob(vec_statev.data(), {*nstatv}, options);
	chi = ten_statev.index({Slice(0, nstt)});											// (nstt, )
	
	// reshape 
	dHSdev5 = dHSdev5.unsqueeze(0).unsqueeze(0);										// (b, s, c)
	dHSdev5_norm = dHSdev5_norm.unsqueeze(0); 											// (b, c)
	chi = chi.unsqueeze(0);																// (b, c)
	
	// prepare input "image"
	inputs_CS.push_back(dHSdev5);
	inputs_CS.push_back(dHSdev5_norm);
	inputs_CS.push_back(chi);

	// stress forward prediction
	mdl_CS.eval();
	mdl_CS.to({torch::kFloat64});
	auto pred_CS = mdl_CS.forward(inputs_CS).toTuple();
	CSdev5 = pred_CS->elements()[0].toTensor();
	chi = pred_CS->elements()[1].toTensor();

	// de-reshape & de-normalize
	CSdev5 = inverse_transform(
		CSdev5.squeeze(), ctn_data.attr("CSdev5_sc").toTensor());						// (5, )
	chi = chi.squeeze();																// (nstt, )

	// get actual CS
	CSdev6 = math_dev5todev6(CSdev5);													// (6, )
	CSmean = ten_statev.index({Slice(nstt, nstt+1)});									// (1, )
	CSmean = CSmean + K*dHSvol.sum();													// (1, )
	CS = CSdev6 + CSmean*torch::tensor({1,1,1,0,0,0});									// (6, )

	// get actual ddCSddHS (ddsdde)
	CSprev = ten_statev.index({Slice(nstt+1, nstt+7)});									// (6, )
	dCS = CS - CSprev;
	for (int i = 0; i < 6; i++) {
		dCSdE.index_put_({i}, torch::autograd::grad({dCS}, {dHS}, {I.index({i})}, true)[0]);
	}

	// symmetrise & flatten
	dCSdE = (dCSdE + dCSdE.transpose(0, 1)).div(2);
	dCSdE = dCSdE.flatten();

	// update statev
	for (int i = 0; i < nstt; i++) {statev[i] = chi.index({i}).item<double>();}
	for (int i = 0; i < 1; i++) {statev[nstt+i] = CSmean.index({i}).item<double>();}
	for (int i = 0; i < 6; i++) {statev[nstt+1+i] = CS.index({i}).item<double>();}

	// update stress & ddsdde
	for (int i = 0; i < 6; i++) {stress[i] = CS.index({i}).div(1000000).item<double>();}
	for (int i = 0; i < 36; i++) {ddsdde[i] = dCSdE.index({i}).div(1000000).item<double>();}

}

// // check stiffness
// auto eigv = torch::view_as_real(torch::linalg::eigvals(dCSdE));
// if ((eigv < 0).any().item<bool>()) {
// 	auto info = to_string(*time) + "-" + to_string(*noel) + "-" + to_string(*npt);
// 	cout << "----------" << endl;
// 	cout << "Info: " << info << endl;
// 	cout << "dHS:\n" << dHS << endl;
// 	cout << "dCS:\n" << dCS << endl;
// 	cout << "CS:\n" << CS << endl;
// 	cout << "dCSdE:\n" << dCSdE << endl;
// 	cout << "eigv:\n" << eigv << endl;
// }