/*
 * Open source implementation of the ML3 classifier.
 *
 * If you find this software useful, please cite:
 *
 * "Multiclass Latent Locally Linear Support Vector Machines"
 * Marco Fornoni, Barbara Caputo and Francesco Orabona
 * JMLR Workshop and Conference Proceedings Volume 29 (ACML 2013 Proceedings)
 *
 * Copyright (c) 2013 Idiap Research Institute, http://www.idiap.ch/
 * Written by Marco Fornoni <marco.fornoni@alumni.epfl.ch>
 *
 * This file is part of the ML3 Software.
 *
 * ML3 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * ML3 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ML3. If not, see <http://www.gnu.org/licenses/>.
 *
 * trainML3F.cc
 *
 * MATLAB interface to train a ML3 model on single-precision data
 *
 *  Created on: Apr 20, 2013
 *      Author: Marco Fornoni
 */

#include "MexUtils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nlhs!=1)
		mexErrMsgTxt("The number of output variables must be: 1."
				"\nExample syntax: model=trainML3(model,X_tr,y_tr)");
	if (nrhs!=3 && nrhs!=5)
		mexErrMsgTxt("The number of input variables must be either 3:\n"
				"correct syntax: model=trainML3(model,X_tr,y_tr)\n"
				"or 5:\n"
				"\ncorrect syntax: model=trainML3(model,X_tr,y_tr,X_te,y_te)\n");

	//	Eigen::initParallel();
	//	omp_set_num_threads(4);
	//	mexPrintf("Num threads %d.\n", omp_get_num_threads());
	//	mexPrintf("test");
	//	std::cout<<"test"<<std::endl;

	bool testAllEpochs=false;
	const mxArray *mxX = prhs[1];
	const uint xdim = mxGetM(mxX);
	const uint xlen = mxGetN(mxX);

	const mxArray *mxY = prhs[2];
	const uint ylen = mxGetM(mxY);
	const uint ylen2 = mxGetN(mxY);
	int *dy = (int *) mxGetPr(mxY);

	if (ylen != xlen){
		mexErrMsgTxt("Number of samples and labels should agree.\n");
	}

	const mxClassID  category = mxGetClassID(mxX);

	const mxArray *mxXte;
	const mxArray *mxYte;
	int *dyte ;
	uint xtedim;
	uint xtelen;
	uint ytelen;

	ArrayXi yte;


	if (nrhs==5){
		mxXte = prhs[3];
		xtedim = mxGetM(mxXte);
		xtelen = mxGetN(mxXte);

		mxYte = prhs[4];
		ytelen = mxGetM(mxYte);
		const uint ytelen2 = mxGetN(mxYte);
		dyte = (int *) mxGetPr(mxYte);

		if (xtelen>0 && ytelen>0){
			if (xdim != xtedim){
				mexErrMsgTxt("Training and testing samples should have the same dimensionality.\n");
			}
			if (ytelen != xtelen){
				mexErrMsgTxt("Number of test samples and test labels should agree.\n");
			}
			testAllEpochs=true;
		}
		// MAPS THE MATLAB ARRAY INTO AN EIGEN ARRAY OF INTEGERS
		yte=Map<ArrayXi>(dyte,ytelen);
	}

	// MAPS THE MATLAB ARRAY INTO AN EIGEN ARRAY OF INTEGERS
	const Map<ArrayXi> y(dy,ylen);

	if (mxIsSparse(mxX)) {
		std::cout<<"Sparse input data is NOT supported."<<std::endl;
	}else if(category==mxSINGLE_CLASS) {
		float *val;
		Model<float> model=Model<float>();
		ML3<float> ml3=ML3<float>();
		MexUtils<float> mu=MexUtils<float>();
		mu.load_mex_model(prhs[0], model);

		float *dX = (float *) mxGetData(mxX);
		// MAPS THE MATLAB MATRIX INTO AN EIGEN MATRIX OF DOUBLES
		const Map<MatrixXf> X(dX,xdim,xlen);

		if (model.maxCCCPIter < 0){
			mexErrMsgTxt("The number of CCCP iterations must be >= 0 \n");
		}else if (model.initStep==0 && model.maxCCCPIter==0){
			mexErrMsgTxt("If no initialization is performed the number of CCCP iterations must be >= 1 \n");
		}

		mu.timer_reset();
		if (testAllEpochs){
			float *dXte =  (float *) mxGetData(mxXte);
			MatrixXf Xte;
			// MAPS THE MATLAB MATRIX INTO AN EIGEN MATRIX OF DOUBLES
			Xte=Map<MatrixXf>(dXte,xtedim,xtelen);
			ml3.trainML3(model,X,y,Xte,yte,true);
		}else{
			//			ml3.trainML3(model,X,y,X,y,true);
			ml3.trainML3(model,X,y);
		}
		model.trTime=mu.timer_query();

		// SETS THE model AS AN OUTPUT FOR MATLAB
		mu.setOutput(plhs, model, category);

	}else{
		mexErrMsgTxt("The only supported datatype is single precision floating point.\n");
	}
}
