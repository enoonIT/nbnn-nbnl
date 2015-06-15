classdef ML3 < handle
    %
    % Open source implementation of the ML3 classifier.
    %
    % If you find this software useful, please cite:
    %
    % "Multiclass Latent Locally Linear Support Vector Machines"
    % Marco Fornoni, Barbara Caputo and Francesco Orabona
    % JMLR Workshop and Conference Proceedings Volume 29 (ACML 2013
    % Proceedings) 
    %
    % Copyright (c) 2013 Idiap Research Institute, http://www.idiap.ch/
    % Written by Marco Fornoni <marco.fornoni@alumni.epfl.ch>
    %
    % This file is part of the ML3 Software.
    %
    % ML3 is free software: you can redistribute it and/or modify
    % it under the terms of the GNU General Public License version 3 as
    % published by the Free Software Foundation.
    %
    % ML3 is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    % GNU General Public License for more details.
    %
    % You should have received a copy of the GNU General Public License
    % along with ML3. If not, see <http://www.gnu.org/licenses/>.
    %
    
    properties
        lambda              % regularization weight
        
        p                   % defines which value of p will be used in the
                            % p-norm regularizer
        
        m                   % number of sub-models/class
        
        maxCCCPIter         % maximum number of CCCP iterations
        
        initStep            % the number of initialization iterations (with
                            % all the local weights fixed -randomly-, the
                            % default is 1)
        
        s0                  % the constant to add to the sample counter
                            % (the default is 0)
        
        averaging           % if true computes the average of the solutions
                            % on the last epoch and uses it as the final
                            % solution (default is true)
        
        returnLocalBeta     % If true saves the sample-to-model assignments
                            % (only for the correct class) and returns it
                            % as part of the model (default is false, on
                            % large datasets it requires a lot of memory)
        
        verbose             % if verbose==0 no output is produced,
                            % if verbose==1 synthetic output is produced,
                            % if verbose==2 measures the loss and the
                            % objective function, at each iteration,
                            % (default is 1, set it to 0 for benchmarking
                            % purposes)
        
        addBias             % if true, adds a 1 at the end of each feature
                            % vector, to emulate learning of a bias (for
                            % each model)
        
        maxKMIter           % the maximum number of k-means epochs used to
                            % initialize W and, consequently, the cluster
                            % centers (default is 0, i.e. k-means is not
                            % performed)
    end
    
    methods
        function newClassifier=ML3(lambda,m,maxCCCPIter,p)
            %
            % Constructs a ML3 object
            %
            % lambda            regularization weight (by default 1/n)
            %
            % m                 number of sub-models/class (by default 10)
            %
            % maxCCCPIter       maximum number of CCCP iterations
            %
            % p                 defines which value of p will be used in
            %                   the p-norm regularizer (by default 1.5)
            %
            
            if ~exist('lambda','var')
                lambda='auto';
            end
            if ~exist('p','var')
                p=1.5;
            end
            if ~exist('maxCCCPIter','var')
                maxCCCPIter=30;
            end
            if ~exist('m','var')
                m=10;
            end
            newClassifier.lambda=lambda;
            newClassifier.p=p;
            
            newClassifier.maxKMIter=uint32(0);
            newClassifier.maxCCCPIter=uint32(maxCCCPIter);
            newClassifier.m=uint32(m);
            newClassifier.verbose=uint32(1);
            newClassifier.initStep=uint32(1);
            newClassifier.s0=uint32(0);
            
            newClassifier.addBias=1;
            newClassifier.averaging=true;
            newClassifier.returnLocalBeta=false;
        end
        
        function model=train(obj,features,labels)
            %
            % Trains the model using the provided features and labels
            %
            % features          dxn feature matrix
            %
            % labels            nx1 vector of labels
            %
            % IT SUPPORTS SINGLE AND DOUBLE PRECISION FEATURES
            % FOR CHEAPER OR, MORE PRECISE COMPUTATIONS
            %
            
            if obj.addBias
                features(end+1,:)=obj.addBias;
            end
            model=obj.trainModel(features,labels, obj.lambda,obj.m,obj.maxCCCPIter,obj.p,obj.averaging,obj.maxKMIter,obj.initStep,obj.s0,obj.verbose,obj.returnLocalBeta);
        end
        
        function [dec_values,predict_labels,accuracy,confusion,predict_beta]=test(obj,features,labels,model)
            %
            % Tests the model using the provided features and labels
            %
            % features          dxn feature matrix
            %
            % labels            nx1 vector of labels
            %
            % IT SUPPORTS SINGLE AND DOUBLE PRECISION FEATURES
            % FOR CHEAPER OR, MORE PRECISE COMPUTATIONS
            %
            
            if obj.addBias
                features(end+1,:)=obj.addBias;
            end
            if nargout >=5
                [dec_values,predict_labels,accuracy,confusion,predict_beta]=obj.testModel(features,labels,model);
            else
                [dec_values,predict_labels,accuracy,confusion]=obj.testModel(features,labels,model);
            end
        end
        
    end
    
    methods (Static)
        
        function model=initModel(X,y,lambda,m,maxCCCPIter,p,averaging,maxKMIter,initStep,s0,verbose,returnLocalBeta)
            %
            % PERFORMS INITIALIZATION OF THE MODEL
            
            % It uses the class of the training data matrix (single, or
            % double precision) and initialize all the other model
            % variables accordingly
            targetClass=class(X);
            
            model.lambda=cast(lambda,targetClass);
            model.p=cast(p,targetClass);
            model.tau=cast(1,targetClass);
            
            model.maxKMIter=uint32(maxKMIter);
            model.maxCCCPIter=uint32(maxCCCPIter);
            model.verbose=uint32(verbose);
            model.nCla=uint32(max(unique(y)));
            model.nSamp=uint32(size(X,2));
            model.nFeats=uint32(size(X,1));
            model.m=uint32(m);
            model.initStep=uint32(initStep);
            model.s=uint32(s0);
            
            model.averaging=logical(averaging);
            model.returnLocalBeta=logical(returnLocalBeta);
            
            maxEpochs=maxCCCPIter+initStep;
            model.avgLoss=zeros(maxEpochs,1);
            model.ael=zeros(maxEpochs,1);
            model.loss=zeros(maxEpochs,1);
            model.obj=zeros(maxEpochs,1);
            model.teAcc=zeros(maxEpochs,1);
        end
        
        
        function model=trainModel(X,y,lambda,m,maxCCCPIter,p,averaging,maxKMIter,initStep,s0,verbose,returnLocalBeta,Xte,yte)
            %
            % TRAINS A ML3 MODEL
            %
            % X                   dxn feature matrix
            %
            % y                   nx1 labels vector
            %
            % lambda              regularization weight (by default 1/n)
            %
            % m                   number of sub-models/class used
            %                     (by default 10)
            %
            % maxCCCPIter         maximum number of CCCP iterations
            %                     (by default 30)
            %
            % p                   defines which value of p-norm will be
            %                     used (by default 1.5)
            %
            % averaging           if true computes the average of the
            %                     solutions on the last epoch and uses it
            %                     as the final solution
            %
            % maxKMIter           the maximum number of k-means epochs used to
            %                     initialize W and, consequently, the cluster
            %                     centers (default is 0, i.e. k-means is not
            %                     performed)
            %
            % initStep            if true, the first epoch is run as a
            %                     normal SVM (with all the local
            %                     maximizations disabled)
            %
            % s0                  the coefficient to be added to s, in the
            %                     learning rate (eta=1/(lambda*(s+s0));
            %
            % verbose             if true measures the loss and the
            %                     objective function, at each iteration,
            %                     if false measures the AEL
            %
            % returnLocalBeta     if true saves the sample-to-model
            %                     assignments (only for the correct class)
            %                     and returns it as part of the model
            %
            % Xte                 if present (and non-empty) the model is
            %                     tested on these features at each epoch
            %
            % yte                 if the lables for the testing features
            %
            
            if ~exist('lambda','var') || (ischar(lambda) && strcmp(lambda,'auto'))
                lambda=1/numel(y);
            end
            if ~exist('p','var') || (ischar(p) && strcmp(p,'auto'))
                p=1.5;
            end
            if ~exist('maxCCCPIter','var')
                maxCCCPIter=30;
            end
            if ~exist('m','var')
                m=10;
            end
            if ~exist('averaging','var')
                averaging=true;
            end
            if ~exist('maxKMIter','var')
                maxKMIter=0;
            end
            if ~exist('initStep','var')
                initStep=1;
            end
            if ~exist('s0','var') || (ischar(s0) && strcmp(s0,'auto'))
                s0=0;
            end
            if ~exist('verbose','var')
                verbose=uint32(1);
            end
            if ~exist('returnLocalBeta','var')
                returnLocalBeta=false;
            end
            if ~exist('Xte','var')
                Xte=[];
            end
            if ~exist('yte','var')
                yte=[];
            end
            
            % labels are expected to be between 0 and C-1
            % if necessary, a mapping is created to enforce this
            % requirement
            uy=unique(y);
            my=max(max(y),numel(uy));
            if  numel(uy)~=my || sum(uy~=reshape((1:my),size(uy)))
                mapping=uy;
                y2=y;
                yte2=yte;
                for i=1:numel(mapping)
                    y2(y==mapping(i))=i;
                    yte2(yte2==mapping(i))=i;
                end
                y=y2;
                yte=yte2;
                clear y2 yte2;
            end
            
            % performs initialization of the model
            model=ML3.initModel(X,y,lambda,m,maxCCCPIter,p,averaging,maxKMIter,initStep,s0,verbose,returnLocalBeta);
            
            % trains the model using the provided mex file
            if isa(X,'double')
                model=trainML3D(model,full(X),int32(y-1),full(cast(Xte,class(X))),int32(yte-1));
            elseif isa(X,'single')
                model=trainML3F(model,full(X),int32(y-1),full(cast(Xte,class(X))),int32(yte-1));
            else
                error('The only supported data types are single and double precision floating points');
            end
            
            if exist('mapping','var')
                model.mapping=mapping;
            end
        end
        
        function [dec_values,predict_labels,accuracy,confusion,predict_beta]=testModel(X,labels,model)
            %
            % TESTS A ML3 MODEL
            %
            % X                 dxn feature matrix
            % labels            nx1 labels vector
            % model             a trained ML3 model
            %
            
            % if necessary, maps the ground-truth to the labels used for 
            % training (e.g. if the ground-truth is in {-1,1})
            if isfield(model,'mapping')
                labels2=labels;
                for i=1:numel(model.mapping)
                    labels2(labels==model.mapping(i))=i;
                end
                labels=labels2;
                mapping=model.mapping;
                model=rmfield(model,'mapping');
            end
            
            if nargout >=5, computeBeta=true; else, computeBeta=false; end
            
            % tests the model and computes the confusion matrices
            if computeBeta
                [accuracy,predict_labels,dec_values,predict_beta]=testML3(model,full(cast(X,class(model.lambda))),int32(labels-1));
            else
                [accuracy,predict_labels,dec_values]=testML3(model,full(cast(X,class(model.lambda))),int32(labels-1));
            end
            predict_labels=predict_labels+1;
            confusion=ML3.cfusion(int16(labels),int16(predict_labels));
            
            % if necessary, remaps-back the predicted labels to the
            % original ones (e.g. if the ground-truth was in {-1,1})
            if exist('mapping','var')
                predict_labels2=cast(predict_labels,class(labels));
                for i=1:numel(mapping)
                    predict_labels2(predict_labels==i)=mapping(i);
                end
                predict_labels=predict_labels2;
            end
        end
        
        function c = cfusion(x,y)
            %
            % Computes the confusion matrix c,
            % using the ground-truth output x
            % and the predicted output y
            %
            
            ux = unique(x);
            ux = reshape(ux,1,numel(ux));
            uy = unique(y);
            uy = reshape(uy,1,numel(uy));
            
            c = zeros(numel(ux),numel(uy));
            for i = ux
                z = y(x == i);
                for j = uy
                    c(i,j) = sum(z == j);
                end
            end
        end
    end
end
