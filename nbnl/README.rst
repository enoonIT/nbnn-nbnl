Multiclass Latent Locally Linear SVM
==========================================

| Copyright (c) 2013 Idiap Research Institute, http://www.idiap.ch/
| Written by `Marco Fornoni <http://fornoni.github.io/>`_ <marco.fornoni@alumni.epfl.ch>
|
| Idiap Research Institute,
| Centre du Parc, P.O. Box 592,
| Rue Marconi 19,
| 1920 Martigny, Switzerland
| Telephone: +41 27 721 77 57
| Fax: +41 27 721 77 12

This file is part of the ML3 Software.

ML3 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

ML3 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ML3. If not, see <http://www.gnu.org/licenses/>.


About
-----
Kernelized Support Vector Machines (SVM) have gained the status of off-the-shelf 
classifiers, able to deliver state of the art performance on almost any problem. 
Still, their practical use is constrained by their computational and memory 
complexity, which grows super-linearly with the number of training samples. 
In order to retain the low training and testing complexity of linear classifiers 
and the exibility of non linear ones, a growing, promising alternative is 
represented by methods that learn non-linear classifiers through local combinations 
of linear ones.

The `Multiclass Latent Locally Linear SVM <http://publications.idiap.ch/downloads/papers/2013/Fornoni_ACML2013_2013.pdf>`_ 
(ML3) can learn complex decision functions, traditionally given by kernels, through 
the use of locally linear decision functions. Differently from kernel classifiers, 
ML3 makes use of a set of linear models that are locally linearly combined to form 
a non-linear decision boundary in the input space. Thanks to the latent 
formulation, the combination coefficients are modeled as latent variables and 
efficiently estimated using an analytic solution.

ML3 has potential applications on large-scale problems, requiring powerful 
classifiers and efficient learning methods, whose training complexity with 
respect to the number of samples is not super-linear.


Usage
-----
This is a mixed C++ and MATLAB (c) implementation of the ML3 
algorithm, with the main algorithm being implemented in a mex file. 
It is develped under Ubuntu 12.10, Matlab R2013a and it makes use
of the `Eigen 3.1 library <http://eigen.tuxfamily.org>`_.
Configurations differing from the above are not officially supported.

In order to use the software you need to:

1. Install the Eigen 3.1 library, using:
    `$ sudo apt-get install libeigen3-dev`

2. Compile ML3 for your architecture, using 
    `$ make`

3. From MATLAB, instantiate the ML3 algorithm using 
    `algo=ML3();`

4. Train the algorithm using
    `model=algo.train(features,labels);`

5. Test the algorithm using 
    `[dec_values,predict_labels,accuracy,confusion]=algo.test(features,labels,model);`


Cite ML3
--------
If you find this software useful, please cite::

  @INPROCEEDINGS{Fornoni_ACML2013_2013,
         author = {Fornoni, Marco and Caputo, Barbara and Orabona, Francesco},
         editor = {Ong, Cheng Soon and Ho, Tu-Bao},
       keywords = {Latent SVM, Locally Linear Support Vector Machines, multiclass classification},
       projects = {Idiap},
          title = {Multiclass Latent Locally Linear Support Vector Machines},
      booktitle = {JMLR W\&CP, Volume 29: Asian Conference on Machine Learning},
           year = {2013},
          pages = {229-244},
       location = {Canberra, Australia},
           issn = {1938-7228},
            url = {http://jmlr.org/proceedings/papers/v29/},
            pdf = {http://publications.idiap.ch/downloads/papers/2013/Fornoni_ACML2013_2013.pdf}
  }
