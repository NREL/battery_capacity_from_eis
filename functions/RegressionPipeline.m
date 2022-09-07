% Paul Gasper, NREL, 2/2022

classdef RegressionPipeline
    %REGRESSIONPIPELINE Flexible machine-learning pipeline for regression
    %The REGRESSIONPIPELINE class empowers flexible and repeatable machine-
    %learning by handling common machine-learning project steps, such as 
    %feature engineering, data normalization, feature selection, model 
    %training, and model prediction in an organized and rigorously defined 
    %fashion. The REGRESSIONPIPELINE class outputs predictions as 
    %Prediction class objects, which package model outputs together with 
    %precalculated fit statistics, if there is target variable data to 
    %compare against. Flexibility for the feature engineering, feature 
    %selection, and standardization is provided by storing each method as a 
    %function handle which is then input as an array of feature
    %transformation function_handles when instantiating the
    %REGRESSIONPIPELINE, allowing any feature transformation sequence to be
    %input. Any feature transformation hyperparameters defined during model
    %training are automatically saved and used during test predictions. For
    %instance, when standardizing data, you use the means and standard
    %deviations learned from the training data to normalize the test data.
    %
    %Common model training and testing methods, such as a train/test split
    %or cross-validation, can be run automatically using the traintest() or
    %crossvalidate() methods. These simply call the train() and predict()
    %methods using the data splitting or cross-validation schemes provided
    %as inputs to the method. Cross-validation or data splitting schemes
    %assume can be defined using the MATLAB function 'cvpartition' or, for
    %data sets containing several unique series or timeseries data, neither
    %of which are well handlded by the cvpartition function, the functions
    %'cvpartseries' can be used for partitioning on the basis of series, or
    %for timeseries (one or multiple), the 'timeseriessplit' function can
    %be used.
    %
    %During training, the pipeline follows the procedure:
    %   1) Apply the feature transformation sequence, storing any feature
    %   transformation hyperparameters that are defined during training
    %   2) Train the Model using the ModelFunc
    %   3) Return the Prediction
    %During testing, the trained model is then used to make predictions:
    %   1) Apply the feature transformation sequence, using any feature
    %   transformation hyperparameters defined during training
    %   2) Make a prediction using Model.predict()
    %   3) Return model Prediction
    %
    %Note that all feature transformation functions in the feature
    %transformation sequence must conform to the format:
    %   function [X_, hyp] = example(X, hyp)
    %where hyp is any hyperparameters that are set during training, for
    %example, selected feature indices or normalization factors. The input
    %X and output X_ are tables. The input hyp is name-value pairs, which
    %can be easily handled using an arguments block, and the output hyp is
    %a cell array of name-value pairs. Feature transformation functions
    %that require the target variable as input, such as many feature
    %selection algorithms, can use the special name-value pair {'y', []} as
    %a fixed hyperparameter input, and the target data from any given
    %training set will automatically be input to the function.
    
    
    %{
    TO-DO:    
    Could add in a TargetTransform sequence, as well. This would require a
    TargetUntransfrom sequence. Not sure how to handle uncertainty with
    arbitrary target transforms.
    
    Handle YPred uncertainty by defining a distribution for every row in
    YPred? For GP, this is easy (Gaussian distributions for every
    prediction). For others, would need to generate a wide variety of
    confidence intervals and then fit a distribution for each row, or check
    if each model always outputs a Gaussian uncertainty distribution or not
    (then can simply pull the 1sigma CI and define a Gaussian using that).
    %}
    
    properties
        % Cell of feature transforming function_handles
        FeatureTransformationSequence
        % Cell of fixed hyperparameters
        FeatureTransformationFixedHyp
        % Cell of trained hyperparameters
        FeatureTransformationTrainedHyp
        
        % A function_handle that trains a regression model, like fitrgp
        ModelFunc 
        % Name-value pairs to pass to ModelFunc
        ModelFuncOpts 
        % Model saving/modification options, like compacting and weights
        ModelOpts 
        % Trained model to make predictions with
        Model     
        
        % Feature data variable names
        FeatureVars 
        % Target data variable names
        TargetVar   
    end
    
    methods
        function obj = RegressionPipeline(ModelFunc, opts)
            %REGRESSIONPIPELINE Construct an instance of this class
            %   Construct a REGRESSIONPIPELINE by defining a model and any
            %   feature generation or selection methods.
            %   Inputs (required):
            %       ModelFunc (function_handle): a function_handle that
            %           returns a model object that has been trained on the
            %           training data and will be stored in obj.Model. The
            %           Model object must have a 'predict' method that
            %           makes predictions on new data.
            %   Inputs (optional):
            %       ModelFuncOpts (cell): Name-value pairs to input into 
            %           the ModelFunc as options.
            %       ModelOpts (struct): Name-value pairs used to modify the
            %           how the model is stored or used.
            %           ModelOpts.CompactModel (logical): save a compact
            %               version of the model.
            %           ModelOpts.Weights (vector): weights of rows for
            %               training.
            %       FeatureTransformationSequence (cell): cell array of
            %           function_handle objects
            %       FeatureTransformationFixedHyp (cell): cell array of
            %           fixed hyperparameter inputs into functions called 
            %           by the FeatureTransformationSequence.
            %   Output:
            %       obj: Instance of the RegressionPipeline class
            
            % Input parsing
            arguments
                ModelFunc function_handle
                opts.ModelFuncOpts cell = {}
                opts.ModelOpts struct = struct();
                opts.FeatureTransformationSequence cell = {}
                opts.FeatureTransformationFixedHyp cell = {}
            end
            
            % Argument validation
            if ~isempty(opts.FeatureTransformationFixedHyp)
                assert(length(opts.FeatureTransformationFixedHyp) == length(opts.FeatureTransformationSequence),...
                    "If manually setting any elements in FeatureTransformationFixedHyp, FeatureTransformationFixedHyp and FeatureTransformationSequence must be the same length.")
            end
            if isempty(opts.FeatureTransformationSequence)
                opts.FeatureTransformationTrainedHyp = {};
            else
                if isempty(opts.FeatureTransformationFixedHyp)
                    opts.FeatureTransformationFixedHyp = cell(length(opts.FeatureTransformationSequence), 1);
                end
                opts.FeatureTransformationTrainedHyp = cell(length(opts.FeatureTransformationSequence), 1);
            end
            
            % Default ModelOpts
            if isempty(fieldnames(opts.ModelOpts))
                opts.ModelOpts.CompactModel = false;
                opts.ModelOpts.Weights = [];
            end
            if ~any(strcmp(fieldnames(opts.ModelOpts), 'CompactModel'))
                opts.ModelOpts.CompactModel = false;
            end
            
            % Assign properties
            obj.ModelFunc = ModelFunc;
            obj.ModelFuncOpts = opts.ModelFuncOpts;
            obj.ModelOpts = opts.ModelOpts;
            obj.FeatureTransformationSequence = opts.FeatureTransformationSequence;
            obj.FeatureTransformationFixedHyp = opts.FeatureTransformationFixedHyp;
            obj.FeatureTransformationTrainedHyp = opts.FeatureTransformationTrainedHyp;
        end
        
        %% Pipeline training and testing        
        function [obj, PTrain] = train(obj, X, Y, seriesIdx)
            %TRAIN Trains the pipeline.
            %   TRAIN learns any feature transformation pipeline
            %   hyperparameters and optimizes the Model using ModelFunc and
            %   any optional inputs defined in ModelOpts on the feature
            %   table X and target table Y.
            %   Inputs (required):
            %       obj (RegressionPipeline)
            %       X (table): Feature table
            %       Y (table): 1-dimensional target table
            %   Inputs (optional):
            %       seriesIdx (double): Vector denoting individual data
            %           series with unique indices
            %   Outputs:
            %       obj
            %       PTrain (Prediction): Results from the training data
            
            % Input parsing
            arguments
                obj
                X table
                Y table
                seriesIdx double = ones(height(X), 1)
            end
            % Input validation
            assert(width(Y) == 1, ...
                "RegressionPipeline is only for predicting 1D target variables.")
            assert(height(X) == height(Y), ...
                "Must be the same number of rows in X and Y.")
            
            % Feature transformation sequence
            [Xt, hyp] = runFeatureTransformationSequence(obj, X, Y, "IsTraining", true);
            obj.FeatureTransformationTrainedHyp = hyp;
            
            % Feature/Target variable names
            obj.FeatureVars = string(Xt.Properties.VariableNames);
            obj.TargetVar = string(Y.Properties.VariableNames);
            
            % Model training
            if ~isempty(obj.ModelOpts.Weights)
                obj.Model = obj.ModelFunc(Xt{:,:}, Y{:,:}, 'Weights', obj.ModelOpts.Weights(seriesIdx), obj.ModelFuncOpts{:});
            else
                obj.Model = obj.ModelFunc(Xt{:,:}, Y{:,:}, obj.ModelFuncOpts{:});
            end
            
            % Optional model modifications
            if obj.ModelOpts.CompactModel
                % Compact the model object, if this model type supports it
                obj.Model = compact(obj.Model);
            end
            
            % Results
            PTrain = predictTransformed(obj, Xt, Y, seriesIdx);
        end
        
        function P = predict(obj, X, Y, seriesIdx)
            %PREDICT Generate a prediction on X after feature transformation
            %   PREDICT transforms the features according to the feature
            %   transformation sequence and then creates a prediction. If
            %   target data (Y) is provided, the result is compared to the
            %   actual data.
            %   Inputs (required):
            %       obj (RegressionModel)
            %       X (table): Feature table
            %   Inputs (optional):
            %       Y (table): Target table
            %       seriesIdx (double): Vector denoting individual data
            %           series with unique indices
            %   Output:
            %       P (Prediction): Result of the prediction
            
            % Input parsing
            arguments
                obj
                X table
                Y table = table()
                seriesIdx double = ones(height(X), 1)
            end
            
            % Transform and predict
            Xt = runFeatureTransformationSequence(obj, X, Y, "IsTraining", false);
            P = predictTransformed(obj, Xt, Y, seriesIdx);
        end
        
        function P = predictTransformed(obj, Xt, Y, seriesIdx)
            %PREDICTTRANSFORMED Generate a prediction on Xt.
            %   PREDICTTRANSFORMED Generates a prediction on the input
            %   feature matrix Xt using obj.Model. If target data (Y) is
            %   provided, the result is compared to the actual data. If the
            %   model allows for predicting uncertainty bounds, the
            %   standard deviation and 95% confidence intervals of each
            %   prediction are also repoted.
            %   Inputs (required):
            %       obj (RegressionModel)
            %       Xt (table): Feature table after transformation
            %   Inputs (optional):
            %       Y (table): Target table
            %       seriesIdx (double): Vector denoting individual data
            %           series with unique indices
            %   Output:
            %       P (Prediction): Result of the prediction
            
            % Input parsing
            arguments
                obj
                Xt table
                Y table = table()
                seriesIdx double = ones(height(X), 1)
            end
            % Make the prediction
            switch class(obj.Model)
                case 'DummyRegressor' %fitrdummy
                    [yPred, yStd, y95CI] = predict(obj.Model, Xt{:,:});
                    YPred = array2table([yPred, yStd, y95CI],...
                            'VariableNames', [obj.TargetVar+"Pred", obj.TargetVar+"PredStd", obj.TargetVar+"Pred95CI_lb", obj.TargetVar+"Pred95CI_ub"]);
                case 'RegressionGP' %fitrgp
                    if strcmp(obj.Model.PredictMethod, 'bcd')
                        % No confidence interval
                        yPred = predict(obj.Model, Xt{:,:});
                        YPred = array2table(yPred, 'VariableNames', obj.TargetVar+"Pred");
                    else
                        [yPred, yStd, y95CI] = predict(obj.Model, Xt{:,:});
                        YPred = array2table([yPred, yStd, y95CI],...
                            'VariableNames', [obj.TargetVar+"Pred", obj.TargetVar+"PredStd", obj.TargetVar+"Pred95CI_lb", obj.TargetVar+"Pred95CI_ub"]);
                    end
                case 'CompactRegressionGP' %fitrgp
                    if any(strcmp(obj.Model.PredictMethod, {'bcd','sr','fic'}))
                        % No confidence interval
                        yPred = predict(obj.Model, Xt{:,:});
                        YPred = array2table(yPred, 'VariableNames', obj.TargetVar+"Pred");
                    else
                        [yPred, yStd, y95CI] = predict(obj.Model, Xt{:,:});
                        YPred = array2table([yPred, yStd, y95CI],...
                            'VariableNames', [obj.TargetVar+"Pred", obj.TargetVar+"PredStd", obj.TargetVar+"Pred95CI_lb", obj.TargetVar+"Pred95CI_ub"]);
                    end
                case {'RegressionLinear'} %fitrlinear
                    % High dimensional linear models
                    % No confidence interval
                    yPred = predict(obj.Model, Xt{:,:});
                    YPred = array2table(yPred, 'VariableNames', obj.TargetVar+"Pred");
                case 'LinearModel' %fitlm, stepwiselm
                    [yPred, y95CI] = predict(obj.Model, Xt{:,:});
                    yStd = abs(y95CI(:,2) - y95CI(:,1))./3.92;
                    YPred = array2table([yPred, yStd, y95CI],...
                            'VariableNames', [obj.TargetVar+"Pred", obj.TargetVar+"PredStd", obj.TargetVar+"Pred95CI_lb", obj.TargetVar+"Pred95CI_ub"]);
                case 'NonLinearModel' %fitnlm
                    [yPred, y95CI] = predict(obj.Model, Xt{:,:});
                    yStd = abs(y95CI(:,2) - y95CI(:,1))./3.92;
                    YPred = array2table([yPred, yStd, y95CI],...
                        'VariableNames', [obj.TargetVar+"Pred", obj.TargetVar+"PredStd", obj.TargetVar+"Pred95CI_lb", obj.TargetVar+"Pred95CI_ub"]);
                case {'RegressionSVM', 'CompactRegressionSVM'} %fitrsvm
                    % No confidence interval
                    yPred = predict(obj.Model, Xt{:,:});
                    YPred = array2table(yPred, 'VariableNames', obj.TargetVar+"Pred");
                case {'RegressionTree', 'CompactRegressionTree',...
                        'classreg.learning.regr.RegressionEnsemble',...
                        'classreg.learning.regr.RegressionBaggedEnsemble'...
                        } %fitrtree, fitrensemble
                    % No confidence interval
                    yPred = predict(obj.Model, Xt{:,:});
                    YPred = array2table(yPred, 'VariableNames', obj.TargetVar+"Pred");
                otherwise
                    error('Model class type unrecognized.')
            end
            % Return results
            P = Prediction(Xt, YPred, Y, seriesIdx);
        end
        
        function [obj, PTrain, PTest] = traintest(obj, X, Y, split, seriesIdx)
            %TRAINTEST Trains and tests on the defined splits of X and Y
            %   Train the RegressionPipeline on the training split,
            %   learning any feature transformation hyperparameters as well
            %   as optimizing the model, and then test the pipeline on a
            %   test split.
            %   Inputs (required):
            %       obj (RegressionPipeline)
            %       X (table): Feature data
            %       Y (table): Target data
            %       split (cvpartition-type or logical): Data splitter object.
            %           Must return logical masks or array indices with the
            %           same number of rows as X and Y using the methods 
            %           training(split) and test(split). Can also simply be
            %           a logical mask that indexes the testing data
            %           (training = Y(~split), testing = Y(split)).
            %   Inputs (optional):
            %       seriesIdx (double): Vector denoting individual data
            %           series with unique indices
            %   Output:
            %       obj (RegressionPipeline)
            %       PTrain (Prediction): Results on the training data
            %       PTest (Prediction): Results on the test data
            
            % Input parsing
            arguments
                obj
                X table
                Y table
                split
                seriesIdx double = ones(height(X), 1)
            end
            
            % Train
            if islogical(split)
                maskTrain = ~split;
            else
                maskTrain = training(split);
            end
            XTrain = X(maskTrain, :);
            YTrain = Y(maskTrain, :);
            [obj, PTrain] = train(obj, XTrain, YTrain, seriesIdx(maskTrain));
            
            % Test
            if islogical(split)
                maskTest = split;
            else
                maskTest = test(split);
            end
            XTest = X(maskTest, :);
            YTest = Y(maskTest, :);
            PTest = predict(obj, XTest, YTest, seriesIdx(maskTest));
        end
        
        function [obj, PTrain, RPCrossVal, PCrossVal] = crossvalidate(obj, X, Y, split, seriesIdx)
            %CROSSVALIDATE Cross-validates on various splits of X and Y
            %   For each train/test split in 'split', the RegressionPipeline
            %   is trained and then tested. The cross-val result is the 
            %   combined results from every test split. The output obj and
            %   PTrain are trained on all the data.
            %   Inputs (required):
            %       obj (RegressionPipeline)
            %       X (table): Feature data
            %       Y (table): Target data
            %       split (cvpartition or likewise): Data splitter object.
            %           Must return logical masks or array indices with the
            %           same number of rows as X and Y using the methods 
            %           training(split) and test(split).
            %   Inputs (optional):
            %       seriesIdx (double): Vector denoting individual data
            %           series with unique indices
            %   Output:
            %       obj (RegressionPipeline)
            %       PTrain (Prediction): Results on the training data
            %       RPCrossVal (RegressionPipeline): Array of
            %           RegressionPipeline objects from training on each
            %           fold.
            %       PCrossVal (Prediction): Results on the test data
            
            % Input parsing
            arguments
                obj
                X table
                Y table
                split
                seriesIdx double = ones(height(X), 1)
            end
            
            % Train on the entire data set
            [obj, PTrain] = train(obj, X, Y, seriesIdx);
            
            % Iterate through cross-validation splits
            folds = zeros(height(X), 1);
            for i = 1:split.NumTestSets
                % Train
                maskTrain = training(split, i);
                XTrain = X(maskTrain, :);
                YTrain = Y(maskTrain, :);
                obj_ = obj;
                if any(strcmp(obj_.ModelFuncOpts, 'Weights'))
                    weights = evenlyWeightDataSeries(seriesIdx(maskTrain));
                    idx = find(strcmp(obj_.ModelFuncOpts, 'Weights'));
                    obj_.ModelFuncOpts{idx+1} = weights;
                end
                [Pipe, ~] = train(obj_, XTrain, YTrain, seriesIdx(maskTrain));
                
                % Test
                maskTest = test(split, i);
                XTest = X(maskTest, :);
                YTest = Y(maskTest, :);
                PTest = predict(Pipe, XTest, YTest, seriesIdx(maskTest));
                
                % Store results for this test split
                if i == 1
                    % Create a matrix to store results from all splits
                    varsY = PTest.YPred.Properties.VariableNames;
                    yPred = zeros(height(Y), length(varsY));
                end
                yPred(maskTest,:) = PTest.YPred{:,:};
                folds(maskTest) = i;
                
                % Save the model object
                if i == 1
                    RPCrossVal = Pipe;
                else
                    RPCrossVal = [RPCrossVal; Pipe];
                end
            end
            YPred = array2table(yPred, 'VariableNames', varsY);
            PCrossVal = Prediction(X, YPred, Y, seriesIdx, folds);
        end
        
        %% Feature transformation sequence
        function [Xt, hyp, Xt_] = runFeatureTransformationSequence(obj, X, Y, opts)
            %RUNFEATURETRANSFORMATIONSEQUENCE Transforms features.
            %   RUNFEATURETRANSFORMATIONSEQUENCE iteratively applies
            %   feature transformation function_handles defined in the
            %   FeatureTransformationSequence. Hyperparameters that are
            %   output by any of the feauture transformation sequence
            %   functions during training are saved in
            %   FeatureTransformationHyp and input to the function during
            %   testing. Feature transformation functions may also accept a
            %   pre-defined hyperparameter as input during both training
            %   and testing. Testing/Training is defined using the optional
            %   input, IsTraining, default 0 (testing).
            %   Inputs (required):
            %       obj (RegressionPipeline)
            %       X (table): Input feature table
            %       Y (table): Input target table
            %   Inputs (optional):
            %       IsTraining (logical): Denotes if the sequence is being
            %           trained or used for testing. Default false
            %           (testing).
            %       OutputAllSteps (logical): Denotes if Xt should be saved
            %           at all steps and output. Useful for
            %           debugging/investigating the feature transformation
            %           sequence. This may use a lot of memory if the
            %           feature matrix is large.
            %   Outputs:
            %       Xt (table): Transformed feature table
            %       hyp (cell): Feature transformation hyperparameters
            %           learned during training
            %       Xt_ (cell): Cell array of Xt tables at each step in the
            %           FeatureTransformationSequence, optionally output
            
            % Input parsing
            arguments
                obj
                X table
                Y table
                opts.IsTraining logical = false
                opts.OutputAllSteps logical = false
            end
            % Naive case (no feature transformations)
            Xt = X; hyp = {}; Xt_ = {};
            % Other cases
            if ~isempty(obj.FeatureTransformationSequence)
                % Prepare outputs
                if opts.IsTraining
                    hyp = cell(length(obj.FeatureTransformationSequence),1);
                end
                if opts.OutputAllSteps
                    Xt_ = cell(length(obj.FeatureTransformationSequence),1);
                end
                % Iterate through the feature transformation sequence
                for step = 1:length(obj.FeatureTransformationSequence)
                    stepFunc = obj.FeatureTransformationSequence{step};
                    stepHypFixed = obj.FeatureTransformationFixedHyp{step};
                    if isempty(stepHypFixed)
                        stepHypFixed = {};
                    end
                    if opts.IsTraining
                        % Training
                        % Catch special input 'y'
                        if any(strcmp('y', [stepHypFixed{:}]))
                            idx = find(strcmp('y', [stepHypFixed{:}]));
                            stepHypFixed{idx+1} = Y{:,:};
                        end
                        if nargout(stepFunc) == 1
                            % No trained hyperparameters
                            Xt = stepFunc(Xt, stepHypFixed{:});
                        else
                            % Trained hyperparameters
                            [Xt, hyp{step}] = stepFunc(Xt, stepHypFixed{:});
                        end
                    else
                        % Testing
                        stepHypTrained = obj.FeatureTransformationTrainedHyp{step};
                        if isempty(stepHypTrained)
                            stepHypTrained = {};
                        end
                        Xt = stepFunc(Xt, stepHypFixed{:}, stepHypTrained{:});
                    end
                    % Optional output
                    if opts.OutputAllSteps
                        Xt_{step} = Xt;
                    end
                end
            end
        end
    end
    
    methods(Static)
        %% Example feature transformation methods
        % describe inptus outputs
        
        % Feature normalization methods
        % Z-score normalization
        function [X_, hyp] = normalizeZScore(X, hyp)
            %NORMALIZEZSCORE Example feature normalization method.
            arguments
                X table
                hyp.means double = [];
                hyp.stds  double = [];
            end
            vars = X.Properties.VariableNames;
            x = X{:,:};
            if nargin == 1
                [x_, means, stds] = zscore(x);
                hyp = {"means", means, "stds", stds};
            else
                x_ = (x - hyp.means) ./ hyp.stds;
            end
            X_ = array2table(x_, 'VariableNames', vars);
        end
        
        % Rescale normalization
        function [X_, hyp] = normalizeRescale(X, hyp)
            %NORMALIZERESCALE Example feature normalization method.
            arguments
                X table
                hyp.min double = []
                hyp.max double = []
            end
            vars = X.Properties.VariableNames;
            x = X{:,:};
            if nargin == 1
                xmin = min(x); xmax = max(x);
                x_ = (x - xmin) ./ (xmax - xmin);
                hyp = {"min", xmin, "max", xmax};
            else
                x_ = (x - hyp.min) ./ (hyp.max - hyp.min);
            end
            X_ = array2table(x_, 'VariableNames', vars);
        end
        
        % Feature generation
        function [X_, hyp_] = generateFeaturesPCA(X, hyp)
            %GENERATEFEATURESPCA Example feature generation method using PCA.
            %   Trains the PCA means and coefficients on the training set.
            %   These can then be used to generate PCA features on a test
            %   set. Requires the number of components, 'n', to be input.
            %   The input 'n' can also be less than one, in which case, the
            %   number of components is the minimum number requires to
            %   explain n% of the variability in X.
            %   Inputs (required):
            %       X (table): Feature table
            %   Name-value inputs (required) (fixed hyperparameters):
            %       n (double): Either the number of PCA components (n>=1),
            %           or the percent variability that is explained by the
            %           PCA components (0 < n < 1)
            %       KeepPriorFeatures (logical): Whether or not to keep the
            %           features prior to the PCA transformation, default
            %           is true
            %   Name-value inputs (optional) (trained hyperparameters):
            %       coeff (double): PCA coefficients
            %       means (double): PCA means
            %   Outputs (required):
            %       X_ (table): Transformed features
            %       hyp_ (cell): Name-value pairs for trained 
            %           hyperparameters, coeff and means
            
            % Input parsing
            arguments
                X table
                % Fixed hyperparameters
                hyp.n double
                hyp.KeepPriorFeatures logical = true
                % Trained hyperparameters
                hyp.coeff double = []
                hyp.means double = []
            end
            
            % Grab data
            vars = X.Properties.VariableNames;
            x = X{:,:};

            % Get the principal components
            if isempty(hyp.coeff) % Training
                if hyp.n > 0 && hyp.n < 1 % Use percent explained variability
                    [coeff,score,~,~,explained,means] = pca(x);
                    sumexplained = cumsum(explained)./100;
                    n = find(sumexplained > hyp.n, 1);
                    x_ = score(:, 1:n);
                    hyp_ = {"coeff", coeff(:,1:n), "means", means};
                elseif hyp.n > 1
                    n = hyp.n;
                    [coeff,score,~,~,~,means] = pca(x, "NumComponents", n);
                    x_ = score;
                    hyp_ = {"coeff", coeff(:,1:n), "means", means};
                else
                    error("hyp.n must be between 0 and 1, or an integer greater than 1")
                end
            else % Testing
                n = size(hyp.coeff, 2);
                x_ = (x - hyp.means) * hyp.coeff;
            end
            
            % Generate names for the new variables
            newVars = repmat("pca", n, 1) + transpose(compose("%d", 1:n));
            
            % Assemble output
            if hyp.KeepPriorFeatures
                x_ = [x, x_];
                newVars = [string(vars), newVars'];
            end
            X_ = array2table(x_, 'VariableNames', newVars);
        end
        
        % Feature selection
        function [X_, hyp_] = selectFeaturesRReliefF(X, hyp)
            %SELECTFEATURESRRELIEFF Example feature selection using RRefliefF
            %   Ranks feature importance using a distance-based metric
            %   calculated by the RReliefF algorithm (see documentation for
            %   'relieff'). Returns a specific number of features, set in
            %   hyp.n, in order of importance.
            %   Inputs (required):
            %       X (table): Feature table
            %   Name-value inputs (required) (fixed hyperparameters):
            %       y (double): Target variable vector
            %       n (double): Number of features to output
            %       k (double): Number of nearest neighbors used by the
            %           ReliefF algorithm. Usually, at k=1, estimates are
            %           unreliable for noisy data, and at k=height(X), the
            %           algorithm will fail to find important predictors.
            %           Suggest starting at k=5 to k=10.
            %   Name-value inputs (optional) (trained hyperparameters):
            %       idxSelected (double): array of selected feature indices
            %   Outputs (required):
            %       X_ (table): Transformed features
            %       hyp_ (cell): idxSelected name-value pair
            
            % Input parsing
            arguments
                X table
                % Fixed hyperparameters
                hyp.y double
                hyp.n double
                hyp.k double
                % Trained hyperparameters
                hyp.idxSelected double = []
            end
            
            % Grab data
            vars = X.Properties.VariableNames;
            x = X{:,:};
            
            % Select features
            if isempty(hyp.idxSelected)
                % Training
                y = hyp.y;
                k = hyp.k;
                n = hyp.n;
                [idx, ~] = relieff(x, y, k);
                idxSelected = idx(1:n);
            else
                % Testing
                idxSelected = hyp.idxSelected;
            end
            hyp_ = {'idxSelected', idxSelected};
            x_ = x(:, idxSelected);
            vars = vars(idxSelected);
            
            % Return
            X_ = array2table(x_, 'VariableNames', vars);
        end
    end
    
end
