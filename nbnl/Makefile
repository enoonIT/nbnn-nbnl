##########################################################################
#  Open source implementation of the ML3 classifier.
#
#  If you find this software useful, please cite:
#
#  "Multiclass Latent Locally Linear Support Vector Machines"
#  Marco Fornoni, Barbara Caputo and Francesco Orabona
#  JMLR Workshop and Conference Proceedings Volume 29 (ACML 2013 Proceedings)
#
#  Copyright (c) 2013 Idiap Research Institute, http://www.idiap.ch/
#  Written by Marco Fornoni <marco.fornoni@alumni.epfl.ch>
# 
#  This file is part of the ML3 Software.
# 
#  ML3 is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 3 as
#  published by the Free Software Foundation.
# 
#  ML3 is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with ML3. If not, see <http://www.gnu.org/licenses/>.
# 
#  Makefile for ML3
#  Marco Fornoni, October 7th 2013
##########################################################################

# Tries to automatically detect matlab path
# If it fails you should replace it with the proper path
MATLABDIR ?= `readlink -m \`which matlab\`|head -c -12`

# Makes use of the default eigen3 installation path
# Replace it with the proper path on your machine if it's different
EIGEN_INC_PATH ?= /usr/include/eigen3

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION_DEBUG = -g -largeArrayDims CFLAGS='$$CFLAGS -Wall-Og'
MEX_OPTION = -largeArrayDims CFLAGS='$$CFLAGS -Wall -O3 -msse2'
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

MATH_INC_PATH = $(MATLABDIR)/extern/include
MATH_OBJ_PATH = $(MATLABDIR)/bin/glnxa64
ML3_INC_PATH= $(shell pwd)

HDR = Model.h Model.tc ML3.h ML3.tc MexUtils.h MexUtils.tc Clustering.h Clustering.tc
INC = -I$(MATH_INC_PATH) -I$(ML3_INC_PATH) -I$(EIGEN_INC_PATH)
MEXSOURCE_TR_D = trainML3D.cc
MEXSOURCE_TR_F = trainML3F.cc
MEXSOURCE_TE = testML3.cc

all: trainML3D.$(MEX_EXT) trainML3F.$(MEX_EXT) testML3.$(MEX_EXT)

#Training algorithm compiled for double precision
trainML3D.$(MEX_EXT): $(MEXSOURCE_TR_D) $(HDR)
	$(MEX) $(MEX_OPTION) $(INC) $(MEXSOURCE_TR_D);

#Training algorithm compiled for single precision	
trainML3F.$(MEX_EXT): $(MEXSOURCE_TR_F) $(HDR)
	$(MEX) $(MEX_OPTION) $(INC) $(MEXSOURCE_TR_F);

#Testing method supporting both double and single precision
testML3.$(MEX_EXT): $(MEXSOURCE_TE) $(HDR)
	$(MEX) $(MEX_OPTION) $(INC) $(MEXSOURCE_TE);

clean:
	rm -f *.$(MEX_EXT) *~
