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
 * Clustering.h
 *
 *  Created on: Feb 24, 2014
 *      Author: Marco Fornoni
 */

#ifndef Clustering_H_
#define Clustering_H_

//#include "ML3.h"

using namespace Eigen;

template<typename T>
class Clustering {
public:
	// The internal representation of W
	typedef Matrix< T, Dynamic, Dynamic >  MatrixXT;
	typedef Matrix< T, Dynamic, 1>  VectorXT;
	typedef Array< T, Dynamic, 1>  ArrayXT;

	// Trains a k-means model using X, m cluster centers, for a maximum of maxIter epochs, storing the cluster centers in M
	static void trainKMeans(const MatrixXT &X, const uint m, const uint maxIter, const uint verbose, MatrixXT& M);

	//Empty constructor
	Clustering(){}

	//Empty distructor
	~Clustering(){}
};

#include "Clustering.tc"

#endif /* Clustering_H_ */


