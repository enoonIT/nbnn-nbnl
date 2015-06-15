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
 * testML3.cc
 *
 * MATLAB interface to test a trained ML3 model
 *
 *  Created on: Apr 20, 2013
 *      Author: Marco Fornoni
 */

#include "MexUtils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nlhs!=1 && nlhs!=2 && nlhs!=3 && nlhs!=4)
		mexErrMsgTxt("The number of output variables must be either 1:\n"
				"example syntax: accuracy=testML3(model,X_te,y_te)\n"
				"or 2:\n"
				"example syntax: [accuracy,pred_labels]=testML3(model,X_test,y_test)\n"
				"or 3:\n"
				"example syntax: [accuracy,pred_labels,decision_vals]=testML3(model,X_test,y_test)\n"
				"or 4:\n"
				"example syntax: [accuracy,pred_labels,decision_vals,pred_beta]=testML3(model,X_test,y_test)\n");
	if (nrhs!=3 && nrhs!=4)
		mexErrMsgTxt("The number of input variables must either be 3:\n"
				"example syntax: accuracy=trainML3(model,X_te,y_te)\n"
				"or 4:\n"
				"example syntax: accuracy=trainML3(model,X_te,y_te,test_local_beta)\n");

	const mxArray *mxX = prhs[1];
	const uint xdim = mxGetM(mxX);
	const uint xlen = mxGetN(mxX);

	const mxArray *mxY = prhs[2];
	const uint ylen = mxGetM(mxY);
	const uint ylen2 = mxGetN(mxY);
	int *dy = (int *) mxGetData(mxY);

	//IF THE METHODS NEEDS TO RETURN THE LOCAL BETA
	//THE LOCAL BETA MATRIX HAS AS MANY ROWS AS THE # OF SAMPLES
	//OTHERWISE IT HAS 0 ROWS
	const uint blen = (nlhs==4)?xlen:0;

	// MAPS THE MATLAB ARRAY INTO AN EIGEN ARRAY OF INTEGERS
	const ArrayXi y = Map<ArrayXi>(dy,ylen);

	// READS THE CLASS_ID OF THE DATA (SINGLE, DOUBLE, ETC..)
	const mxClassID  category = mxGetClassID(mxX);

	if (ylen != xlen){
		mexErrMsgTxt("Number of samples and labels should agree.\n");
	}


	if (mxIsSparse(mxX)) {
		std::cout<<"Sparse input data is NOT supported"<<std::endl;
	}else if(category==mxSINGLE_CLASS) {
		float *val;
		float accuracy;
		Model<float> model=Model<float>();
		ML3<float> ml3=ML3<float>();
		MexUtils<float> mu=MexUtils<float>();
		mu.load_mex_model(prhs[0], model);

		if (xdim != model.nFeats){
			mexErrMsgTxt("Training and testing samples should have the same dimensionality.\n");
		}

		float *dX = (float *) mxGetData(mxX);
		// MAPS THE MATLAB MATRIX INTO AN EIGEN MATRIX OF FLOATS
		const Map<MatrixXf> X(dX,xdim,xlen);

		const uint &C=model.nCla;
		const uint &m=model.m;
		MatrixXf dec_values(xlen,C);
		MatrixXf pred_beta(blen,m);
		ArrayXi pred_labels(xlen);

		if (nrhs==3){
			if (nlhs==4)
					accuracy = ml3.testML3(model,X,y,dec_values,pred_labels,pred_beta);
			else
					accuracy = ml3.testML3(model,X,y,dec_values,pred_labels);
		}else if (nrhs==4){
			std::vector<std::vector<VectorXf> > testLocalBeta;
			mxArray *u;
			float *val;
			const mwSize *dims, *dims2;

			const mxArray *v = prhs[3];
			dims = mxGetDimensions(v);
			testLocalBeta.reserve(dims[1]);
			for(uint j=0; j<dims[1];j++){
				std::vector<VectorXf> col;
				col.reserve(dims[0]);
				for(uint i=0; i< dims[0]; i++){
					u = mxGetCell(v, j*dims[0]+i);
					dims2 = mxGetDimensions(u);
					val = (float*) mxGetData(u);
					col.push_back(Map<VectorXf>(val,dims2[0]));
				}
				testLocalBeta.push_back(col);
			}
			accuracy = ml3.testML3(model,X,y,testLocalBeta,dec_values,pred_labels);
		}
		// SETS THE model AS AN OUTPUT FOR MATLAB
		mu.setOutput(plhs,  nlhs, accuracy, pred_labels, dec_values, pred_beta, category);
	}else  if(category==mxDOUBLE_CLASS) {
		double *val;
		double accuracy;
		Model<double> model=Model<double>();
		ML3<double> ml3=ML3<double>();
		MexUtils<double> mu=MexUtils<double>();
		mu.load_mex_model(prhs[0], model);

		if (xdim != model.nFeats){
			mexErrMsgTxt("Training and testing samples should have the same dimensionality.\n");
		}

		double *dX = (double *) mxGetData(mxX);
		// MAPS THE MATLAB MATRIX INTO AN EIGEN MATRIX OF DOUBLES
		const Map<MatrixXd> X(dX,xdim,xlen);

		const uint &C=model.nCla;
		const uint &m=model.m;
		MatrixXd dec_values(xlen,C);
		MatrixXd pred_beta(blen,m);
		ArrayXi pred_labels(xlen);

		if (nrhs==3){
			if (nlhs==4)
				accuracy = ml3.testML3(model,X,y,dec_values,pred_labels,pred_beta);
			else
				accuracy = ml3.testML3(model,X,y,dec_values,pred_labels);
		}else if (nrhs==4){
			std::vector<std::vector<VectorXd> > testLocalBeta;
			mxArray *u;
			double *val;
			const mwSize *dims, *dims2;

			const mxArray *v = prhs[3];
			dims = mxGetDimensions(v);
			testLocalBeta.reserve(dims[1]);
			for(uint j=0; j<dims[1];j++){
				std::vector<VectorXd> col;
				col.reserve(dims[0]);
				for(uint i=0; i< dims[0]; i++){
					u = mxGetCell(v, j*dims[0]+i);
					dims2 = mxGetDimensions(u);
					val = mxGetPr(u);
					col.push_back(Map<VectorXd>(val,dims2[0]));
				}
				testLocalBeta.push_back(col);
			}
			accuracy = ml3.testML3(model,X,y,testLocalBeta,dec_values,pred_labels);
		}

		// SETS THE model AS AN OUTPUT FOR MATLAB
		mu.setOutput(plhs,  nlhs, accuracy, pred_labels, dec_values , pred_beta, category);
	}else{
		mexErrMsgTxt("The only supported datatypes are single and double precision floating point.\n");
	}
}
