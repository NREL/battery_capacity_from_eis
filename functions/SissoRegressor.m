%{
Paul Gasper, NREL, September 2020

SISSO (Sure-Independence Screening -> Sparsifying Operator (here, L0 regularization)):
R. Ouyang, S. Curtarolo, E. Ahmetcik et al., Phys. Rev. Mater.2, 083802 (2018)
R. Ouyang, E. Ahmetcik, C. Carbogno, M. Scheffler, and L. M. Ghiringhelli, J. Phys.: Mater. 2, 024002 (2019).

Code has been translated into MATLAB from the Python implementation of SISSO, which can be found at:
https://analytics-toolkit.nomad-coe.eu/hub/user-redirect/notebooks/tutorials/compressed_sensing.ipynb
%}

classdef SissoRegressor
    properties
        % Input properties:
        % Number of non-zero coefficients / maximum number of dimension of the model
        nNonzeroCoefs {mustBeNumeric} = 1
        % Number of features collected per SIS step (these are then searched exhaustively for models with 1:n_non_zero_coeffs dimension.
        nFeaturesPerSisIter {mustBeNumeric} = 1
        % If true, in the L0 step all combinations of sis_collected features will be checked.
        % If false, combinations of features of the same SIS step will be neglected
        allL0Combinations {mustBeNumericOrLogical} = true;
        
        % Internal properties:
        coefs = [];
        coefsStandardized = [];
        intercept = [];
        listOfCoefs = {};
        listOfNonzeroCoefs = {};
        listOfIntercepts = {};
        rmses = [];
        selectedIndicesSis = {};
        unselectedIndicesSis = [];
        selectedIndicesL0 = {};
        selectedIndicesCurrent = [];
        scales = 1;
        means = 0;
    end
    methods
        function obj = SissoRegressor(nNonzeroCoefs, nFeaturesPerSisIter, allL0Combinations)
            if nargin == 0
                obj.nNonzeroCoefs = 1;
                obj.nFeaturesPerSisIter = 1;
                obj.allL0Combinations = true;
            elseif nargin == 1
                obj.nNonzeroCoefs = nNonzeroCoefs;
                obj.nFeaturesPerSisIter = 1;
                obj.allL0Combinations = true;
            elseif nargin == 2
                obj.nNonzeroCoefs = nNonzeroCoefs;
                obj.nFeaturesPerSisIter = nFeaturesPerSisIter;
                obj.allL0Combinations = true;
            elseif nargin == 3
                obj.nNonzeroCoefs = nNonzeroCoefs;
                obj.nFeaturesPerSisIter = nFeaturesPerSisIter;
                obj.allL0Combinations = allL0Combinations;
            end
        end
        
        function obj = fitSisso(obj, x, y)
            %obj = fitSisso(obj, x, y)
            % Runs the regression algorithm, using the nNonzeroCoefs and
            % nFeaturesPerSisIter properties of obj.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   x (double): Array of values of potential model features.
            %   y (double): Vector of the response variable values.
            % Output:
            %   obj: Instance of the SissoRegressor class, with the results
            %   from the regression stored in the class properties.
            
            % Turn off warnings, there are often rank-dificient matrices
            % when running the L0 optimization at higher dimensions.
            warning('off')
            % Use the sisso regressor to optimize the
            % x: matrix of input features for each response, [Nsamples, Nfeatures]
            % y: vector of response data, [Nsample,1]
            checkParams(obj,size(x,2));
            obj.unselectedIndicesSis = 1:size(x,2);
            % Standardize input features:
            obj = setMeansAndScales(obj, x);
            x = standardizeData(obj, x);
            % Center response variable:
            yMean = mean(y);
            yCentered = y - yMean;
            
            for iter = 1:obj.nNonzeroCoefs
                % Residuals for the first iteration are just the centered P
                % value (model is simply the mean)
                residuals = [];
                if iter == 1
                    residuals = yCentered;
                else
                    residuals = yCentered - predictFromStandardizedData(obj,x);
                end
                
                % SIS: get the indicies of the nFeaturesPerSisIter that
                % are closest to the residuals (first iter -> response,
                % following iters, residuals of the model from the previous
                % iter)
                [obj, nClosestIndices, bestProjectionScore] = sis(obj, x, residuals);
                obj.selectedIndicesSis = [obj.selectedIndicesSis nClosestIndices];
                
                % SA step or L0 step (if iter > 1)
                if iter == 1
                    % RMSE of response - intercept
                    obj.coefsStandardized = bestProjectionScore/length(y);
                    obj.selectedIndicesCurrent = nClosestIndices(1);
                    rmse = sqrt(sum((yCentered - predictFromStandardizedData(obj,x)).^2) ./ length(y));
                else
                    % Perform L0 regularization
                    [obj.coefsStandardized, obj.selectedIndicesCurrent, rmse] = exhaustiveRegularization(obj, x, yCentered, obj.selectedIndicesSis);
                end
                
                % Process and save model outcomes
                % Transform standardized coefs into original scale coefs
                [coefsThisIter, obj.intercept] = getNonstandardizedCoefs(obj, obj.coefsStandardized, yMean);
                % Generate coefs array with zeros except the selected indices
                obj.coefs = zeros(1,size(x,2));
                obj.coefs(obj.selectedIndicesCurrent) = coefsThisIter;
                % Append lists of coefs, indicies, rmses...
                obj.listOfNonzeroCoefs = [obj.listOfNonzeroCoefs coefsThisIter];
                obj.listOfCoefs = [obj.listOfCoefs obj.coefs];
                obj.listOfIntercepts = [obj.listOfIntercepts obj.intercept];
                obj.selectedIndicesL0 = [obj.selectedIndicesL0 obj.selectedIndicesCurrent];
                obj.rmses = [obj.rmses rmse];
            end
        end
        
        function yPred = predictSisso(obj, x, dim)
            %Y_pred = PREDICTSISSO(obj, X, dim)
            % Predict the response for the given input data from the using
            % the previous sisso model (or the one specified by 'dim').
            % Inputs:
            %   x (double): array of input features for each response, 
            %     [Nsamples, Nfeatures]
            %   dim (int): index of desired fitted SISSO model
            % Output:
            %   yPred (double): Vector of predicted response variable
            %   values.
            
            if isempty(dim)
                dim = obj.nNonzeroCoefs;
            end
            
            % Use only selected indices/features of D and add a column of
            % ones for the intercept/bias
            xModel = x(:, obj.selectedIndicesL0{dim});
            xModel = [ones(size(x,1),1), xModel];
            coefsModel = [obj.listOfIntercepts{dim}; obj.listOfNonzeroCoefs{dim}];
            yPred = xModel * coefsModel;
        end
        
        function printModels(obj, features)
            %PRINTMODELS(obj, features)
            % Prints the model constructed from the selected feature list.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   features (cell): Cellstr of feature labels.
            fprintf("%14s %16s\n", 'RMSE', 'Model')
            for modelDim = 1:obj.nNonzeroCoefs
                dispstr = getModelString(obj, features, modelDim);
                disp(dispstr)
            end
            disp(" ");
        end
        
        function checkParams(obj, nColumns)
            %CHECK_PARAMS(obj, nColumns)
            % Checks settings of the fit.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   nColumns (int): Number of columns in the X array.
            
            error_str = strcat("nNonzeroCoefs * nFeaturesPerSisIter is larger /n"...
                ,"than the number of columns in your input matrix./n"...
                ,"Choose a smaller nNonzeroCoefs or a smaller nFeaturesPerSisIter.");
            if nColumns < (obj.nNonzeroCoefs * obj.nFeaturesPerSisIter)
                error(error_str)
            end
        end
        
        function dispL0steps(obj)
            % Calculate and display the number of L0 optimizations during
            % the SO step of SISSO:
            nL0Steps = 0;
            fprintf("L0 optimizations for 1D model: 0\n");
            if obj.allL0Combinations
                for dim = 2:obj.nNonzeroCoefs
                    L0CalcsThisIter = nchoosek(obj.nFeaturesPerSisIter * dim, dim);
                    nL0Steps = nL0Steps + L0CalcsThisIter;
                    fprintf("L0 optimizations for %dD model: %d\n", dim, L0CalcsThisIter);
                end
            else
                for dim = 2:obj.nNonzeroCoefs
                    L0CalcsThisIter = prod(obj.nFeaturesPerSisIter ^ dim);
                    nL0Steps = nL0Steps + L0CalcsThisIter;
                    fprintf("L0 optimizations for %dD model: %d\n", dim, L0CalcsThisIter);
                end
            end
            fprintf("Total # of L0 optimizations: %d\n", nL0Steps)
        end
        
        function obj = setMeansAndScales(obj, x)
            %obj = SETMEANSANDSCALES(obj, x)
            % Stores the mean and standard deviation of x in the
            % corresponding properties of obj.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   x (double): Array of values of potential model features.
            obj.means = mean(x);
            obj.scales = std(x);
        end
        
        function x = standardizeData(obj, x)
            %obj = STANDARDIZEDATA(obj, x)
            % Standardizes all columns of x.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   x (double): Array of values of potential model features.
            x = (x - obj.means) ./ obj.scales;
        end
        
        function yPred = predictFromStandardizedData(obj, x)
            %Y_predict = PREDICTFROMSTANDARDIZEDDATA(obj, x)
            % Calculates the predicition of the response variable using the
            % optimized coefficient values and chosen features from the
            % most recent iteration of fitSisso using a standardized input
            % feature array x.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   x (double): Array of standardized values of potential model 
            %     features. 
            % Outputs:
            %   yPred: Vector of predicted response variable values.
            yPred = x(:,obj.selectedIndicesCurrent) * obj.coefsStandardized;
        end
        
        function [obj, indicesNClosestOut, bestProjectionScore] = sis(obj, x, y)
            %[obj, indicesNClosestOut, bestProjectionScore] = SIS(obj, x, y)
            % Finds the nFeaturesPerSisIter feature columns with the
            % lowest projection scores compared to 1) Y (if this is the
            % first iteration of SIS) or 2) the residual errors of the
            % previous SISSO iteration.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   x (double): Array of values of potential model features.
            %   y (double): Vector of the values of the response variable.
            % Outputs:
            %   obj: Instance of the SissoRegressor class, with some
            %     regression results stored in obj properties.
            %   indicesNClosestOut (int): Column indicies of the
            %     features selected during this SIS iteration.
            %   bestProjectionScore (double): The best projection score
            %     from this SIS iteration.
            
            % Evaluate how close each feature is to the target without
            % already selected indicies from prior iterations
            projectionScores = y'*x(:,obj.unselectedIndicesSis);
            absProjectionScores = abs(projectionScores);
            
            % Sort the values according to their absolute projection score
            % starting from the closest, and get the indices of the
            % nFeaturesPerSisIter closest values.
            [~,indicesSorted] = sort(absProjectionScores);
            indicesSorted = flip(indicesSorted);
            indicesNClosest = indicesSorted(1:obj.nFeaturesPerSisIter);
            bestProjectionScore = projectionScores(indicesNClosest(1));
            
            % Transform indicesNClosest according to original indices of
            % 1:size(D,2) and delete the selected ones from
            % obj.unselectedIndicesSis
            indicesNClosestOut = obj.unselectedIndicesSis(indicesNClosest);
            obj.unselectedIndicesSis = obj.unselectedIndicesSis(~any(obj.unselectedIndicesSis == indicesNClosest(:)));
        end
        
        function [coefsStandardized, selectedIndicesCurrent, rmse] = exhaustiveRegularization(obj, x, y, listOfSisIndices)
            %[coefsStandardized, selectedIndicesCurrent, rmse] = EXHAUSTIVEREGULARIZATION(obj, x, y, listOfSisIndices)
            % Exhaustively searches models formed by all possible
            % combinations of the features in list_of_sis_indices to find
            % the best model structure (curr_selected_indicies),
            % coefficients (coefsStandardized), and root mean square error (rmse).
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   x (double): Array of values of potential model features.
            %   y (double): Vector of the values of the response variable.
            %   listOfSisIndices (cell): cell array with arrays of the
            %     indices of possible features output from each iteration
            %     of the SIS step.
            % Outputs:
            %   coefsStandardized (double): Array of coefficient values for the
            %     best model.
            %   selectedIndicesCurrent (int): Array of indices denoting
            %     which feature columns are used by the best model.
            %   rmse (double): Root mean square error of the prediction
            %     made by the best model.
            
            sqaureErrorMin = y'*y;
            coefsMin = [];
            indicesCombinationMin = [];
            
            % Check each least sqaures error of combination of each indices
            % array of listOfSisIndices. If obj.allL0Combinations is
            % false, combinations of features from the same SIS iteration
            % will be neglected.
            if obj.allL0Combinations
                combinations = combnk([listOfSisIndices{:}], length(listOfSisIndices));
            else
                combinations = obj.cartesianProduct(listOfSisIndices);
            end
            
            for i = 1:size(combinations,1)
                thisCombination = combinations(i,:);
                xCombination = x(:,thisCombination);
                coefsCombination = xCombination\y;
                residuals = y - (xCombination * coefsCombination);
                squareError = sum(residuals.^2);
                if squareError < sqaureErrorMin
                    sqaureErrorMin = squareError;
                    coefsMin = coefsCombination;
                    indicesCombinationMin = thisCombination;
                end
            end
            coefsStandardized = coefsMin;
            selectedIndicesCurrent = indicesCombinationMin;
            rmse = sqrt(sqaureErrorMin / size(x,1));
        end
        
        function [coefsOrig, interceptOrig] = getNonstandardizedCoefs(obj, coefsStandardized, interceptStandardized)
            %[coefsOrig, interceptOrig] = GETNONSTANDARDIZEDCOEFS(obj, coefsStandardized, interceptStandardized)
            % Transform coefs of a linear model with standardized input to
            % coefs of a linear model with original (nonstandardized) input.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   coefsStandardized (double): Coefficient values assuming
            %     standardized input features.
            %   interceptStandardized (double): Model intercept value assuming 
            %     standardized input features.
            % Outputs:
            %   coefsOrig (double): Coefficient values for raw input
            %     features.
            %   interceptOrig (double): Model intercept value for original
            %     input features.
            coefsOrig = coefsStandardized ./ obj.scales(obj.selectedIndicesCurrent)';
            interceptOrig = interceptStandardized - (obj.means(obj.selectedIndicesCurrent) ./ obj.scales(obj.selectedIndicesCurrent))*coefsStandardized;
        end
        
        function modelstr = getModelString(obj, features, modelDim)
            %model_str = GETMODELSTRING(obj, features, modelDim)
            % Creates a nice string for displaying potential models found
            % during the operation of fitSisso.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   features (cell): Cellstr of feature labels.
            %   modelDim (int): Dimension of the model stored in the
            %     properties of obj to create a string for.
            % Output:
            %   modelstr (char): Nice string displaying the resulting
            %     model of dimension model_dim.
            
            coefs_dim = [obj.listOfIntercepts{modelDim}, obj.listOfNonzeroCoefs{modelDim}'];
            selectedFeatures = {[]};
            for idx = obj.selectedIndicesL0{modelDim}
                selectedFeatures = [selectedFeatures features(idx)];
            end
            modelstr = sprintf("%dD: \t%8f\t", modelDim, obj.rmses(modelDim));
            for dim = 1:modelDim+1
                c = coefs_dim(dim);
                c_abs = abs(c);
                if c > 0
                    sign = '+';
                else
                    sign = '-';
                end
                if dim == 1 % intercept
                    modelstr = strcat(modelstr, sprintf("%0.3f ", c));
                else % feature
                    modelstr = strcat(modelstr, sprintf("%s %0.3f %s ", sign, c_abs, selectedFeatures{dim}));
                end
            end
        end
    end
    methods(Static)
        function x = cartesianProduct(sets)
            %x = CARTESIANPRODUCT(varargin)
            % Modified from https://www.mathworks.com/matlabcentral/fileexchange/5475-cartprod-cartesian-product-of-multiple-sets
            % This is to copy the behavior of itertools.product in Python.
            
            %CARTPROD Cartesian product of multiple sets.
            %
            %   x = CARTESIANPRODUCT(A,B,C,...) returns the cartesian product of the sets
            %   A,B,C, etc, where A,B,C, are numerical vectors.
            %
            %   Example: A = [-1 -3 -5];   B = [10 11];   C = [0 1];
            %
            %   X = cartprod(A,B,C)
            %   X =
            %
            %     -5    10     0
            %     -3    10     0
            %     -1    10     0
            %     -5    11     0
            %     -3    11     0
            %     -1    11     0
            %     -5    10     1
            %     -3    10     1
            %     -1    10     1
            %     -5    11     1
            %     -3    11     1
            %     -1    11     1
            %
            %   This function requires IND2SUBVECT, also available on the MathWorks
            %   File Exchange site.
            
            numSets = length(sets);
            sizeThisSet = zeros(numSets, 1);
            for i = 1:numSets
                % Check each cell entry, sort it if its okay.
                thisSet = sort(sets{i});
                if ~isequal(numel(thisSet),length(thisSet))
                    error('All inputs must be vectors.')
                end
                if ~isnumeric(thisSet)
                    error('All inputs must be numeric.')
                end
                if ~isequal(thisSet,unique(thisSet))
                    error(['Input set' ' ' num2str(i) ' ' 'contains duplicated elements.'])
                end
                sizeThisSet(i) = length(thisSet);
                sets{i} = thisSet;
            end
            x = zeros(prod(sizeThisSet),numSets);
            for i = 1:size(x,1)
                % Envision imaginary n-d array with dimension "sizeThisSet" ...
                % = length(varargin{1}) x length(varargin{2}) x ...
                idxVect = SissoRegressor.ind2subVect(sizeThisSet,i);
                for j = 1:numSets
                    x(i,j) = sets{j}(idxVect(j));
                end
            end
        end
        
        function X = ind2subVect(siz, idx)
            %X = IND2SUBVECT(siz, idx)
            % From https://www.mathworks.com/matlabcentral/fileexchange/5476-ind2subvect-multiple-subscript-vector-from-linear-index
            
            %IND2SUBVECT Multiple subscripts from linear index.
            %   IND2SUBVECT is used to determine the equivalent subscript values
            %   corresponding to a given single index into an array.
            %
            %   X = IND2SUBVECT(SIZ,IND) returns the matrix X = [I J] containing the
            %   equivalent row and column subscripts corresponding to the index
            %   matrix IND for a matrix of size SIZ.
            %
            %   For N-D arrays, X = IND2SUBVECT(SIZ,IND) returns matrix X = [I J K ...]
            %   containing the equivalent N-D array subscripts equivalent to IND for
            %   an array of size SIZ.
            %
            %   See also IND2SUB.  (IND2SUBVECT makes a one-line change to IND2SUB so as
            %   to return a vector of N indices rather than retuning N individual
            %   variables.)%IND2SUBVECT Multiple subscripts from linear index.
            %   IND2SUBVECT is used to determine the equivalent subscript values
            %   corresponding to a given single index into an array.
            %
            %   X = IND2SUBVECT(SIZ,IND) returns the matrix X = [I J] containing the
            %   equivalent row and column subscripts corresponding to the index
            %   matrix IND for a matrix of size SIZ.
            %
            %   For N-D arrays, X = IND2SUBVECT(SIZ,IND) returns matrix X = [I J K ...]
            %   containing the equivalent N-D array subscripts equivalent to IND for
            %   an array of size SIZ.
            %
            %   See also IND2SUB.  (IND2SUBVECT makes a one-line change to IND2SUB so as
            %   to return a vector of N indices rather than returning N individual
            %   variables.)
            
            if iscolumn(siz)
                siz = siz';
            end
            % All MathWorks' code from IND2SUB, except as noted:
            n = length(siz);
            k = [1 cumprod(siz(1:end-1))];
            idx = idx - 1;
            for i = n:-1:1
                X(i) = floor(idx/k(i))+1;      % replaced "varargout{i}" with "X(i)"
                idx = rem(idx,k(i));
            end
        end
    end
end