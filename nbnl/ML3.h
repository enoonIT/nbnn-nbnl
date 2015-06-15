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
 * ML3.h
 *
 *  Created on: Sep 12, 2013
 *      Author: Marco Fornoni
 */

#ifndef ML3_H_
#define ML3_H_

#include <float.h>
#include <iostream>
#include <math.h>
#include <float.h>

#include "Model.h"
#include "Clustering.h"

using namespace Eigen;

template<typename T>
class ML3 {
public:
	// The internal representation of W
	typedef Matrix< T, Dynamic, Dynamic >  MatrixXT;
	typedef Matrix< T, Dynamic, 1>  VectorXT;
	typedef Array< T, Dynamic, 1>  ArrayXT;

	// Computes the optimal latent variable values, provided the
	void computeOptimalBeta(VectorXT &f, VectorXT &newLocalBeta, const T p, const T q, const uint m, const T tau);

	void simplexProj(VectorXT &x, VectorXT &v, T z, T p, bool exact);

	// Trains the ML3 model using X and y
	void trainML3(Model<T>& model, const MatrixXT& X, const ArrayXi& y);
	void trainML3(Model<T>& model, const MatrixXT& X, const ArrayXi& y,const MatrixXT& Xte,const ArrayXi& yte, bool testAllEpochs);

	// Tests a trained ML3 model and returns the accuracy
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y, const std::vector<std::vector<VectorXT> > &testLocalBeta, T &avgLoss, const bool fixedBeta, const bool computeLoss, MatrixXT &dec_values, ArrayXi &pred_labels, const bool computeBeta, MatrixXT &pred_beta);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y, const std::vector<std::vector<VectorXT> > &testLocalBeta, MatrixXT &dec_values, ArrayXi &pred_labels, T &avgLoss);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y, const std::vector<std::vector<VectorXT> > &testLocalBeta, MatrixXT &dec_values, ArrayXi &pred_labels, MatrixXT &pred_beta);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y, const std::vector<std::vector<VectorXT> > &testLocalBeta, MatrixXT &dec_values, ArrayXi &pred_labels);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y,MatrixXT &dec_values, ArrayXi &pred_labels, T &avgLoss);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y,MatrixXT &dec_values, ArrayXi &pred_labels, MatrixXT &pred_beta);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y,MatrixXT &dec_values, ArrayXi &pred_labels);
	T testML3(const Model<T> &model, const MatrixXT &X, const ArrayXi &y);

	//Empty constructor
	ML3(){}

	//Empty distructor
	~ML3(){}
};

#include "ML3.tc"

#endif /* ML3_H_ */


