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
 * Model.h
 *
 * Class representing a ML3 model
 *
 *  Created on: Apr 20, 2013
 *      Author: Marco Fornoni
 */

#ifndef MODEL_H_
#define MODEL_H_


#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

template<typename T>
class Model {
public:
	// The internal representation of W
	typedef Matrix< T, Dynamic, Dynamic >  MatrixXT;
	typedef Matrix< T, Dynamic, 1>  VectorXT;

	// The internal representation of W
	T lambda, p, tau, trTime;
	bool averaging, returnLocalBeta;
	uint s, initStep, maxKMIter, iter, maxCCCPIter, nCla, nSamp, nFeats, m, verbose;

	std::vector<MatrixXT> W;
	std::vector<MatrixXT> W2;
	MatrixXT localBetaClass;

	VectorXT avgLoss;
	VectorXT ael;
	VectorXT loss;
	VectorXT obj;
	VectorXT teAcc;

	static const char *fnames[24];

	//Default constructor
	Model();

	//Default destructor
	~Model();

	//Main constructor
	Model(std::vector<MatrixXT> _W, std::vector<MatrixXT> _W2, MatrixXT _localBetaClass,  T _lambda, T _p, T _tau, uint _maxCCCPIter, T _nRep, bool _averaging, uint _maxKMIter, uint _initStep, uint _s0, uint _verbose, bool _returnLocalBeta);

	//Main setter method
	void setParams(std::vector<MatrixXT> _W, std::vector<MatrixXT> _W2, MatrixXT _localBetaClass,  T _lambda, T _p, T _tau, uint _maxCCCPIter, T _nRep, bool _averaging, uint _maxKMIter, uint _initStep, uint _s0, uint _verbose, bool _returnLocalBeta);

};

#include "Model.tc"

#endif /* MODEL_H_ */
