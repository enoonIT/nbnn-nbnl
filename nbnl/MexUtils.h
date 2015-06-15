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
 * MexUtils.h
 *
 * Utilities to interface ML3 to MATLAB
 *
 *  Created on: 2 May 2013
 *      Author: Marco Fornoni
 */

#ifndef MEX_UTILS_H_
#define MEX_UTILS_H_
#include <sys/time.h>
#include <mex.h>

#include "ML3.h"

#define TIMER_MAXETIME 2678400;
static struct itimerval timer_virt;

using namespace Eigen;

template<typename T>
class MexUtils{
public:
	typedef Matrix< T, Dynamic, Dynamic >  MatrixXT;
	typedef Matrix< T, Dynamic, 1>  VectorXT;
	typedef Array< T, Dynamic, 1>  ArrayXT;

	// resets the POSIX ITIMER_VIRTUAL
	void timer_reset( void );
	// measures the training time using the POSIX ITIMER_VIRTUAL
	double timer_query( void );

	void load_mex_model(const mxArray *prhs, Model<T> &model);

	void setOutput(mxArray * plhs[], const Model<T> model, const mxClassID  category);

	void setOutput(mxArray * plhs[], int nlhs, T accuracy, ArrayXi &pred_labels, MatrixXT &dec_values, MatrixXT &pred_beta, const mxClassID category);

	mxArray* mxCreateTScalar(T value, const mxClassID category);
};

#include "MexUtils.tc"

#endif /* MEX_UTILS_H_ */

