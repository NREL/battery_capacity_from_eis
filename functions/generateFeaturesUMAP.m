function [X_, hyp_] = generateFeaturesUMAP(X, hyp)
%GENERATEFEATURESUMAP Example feature generation method using UMAP.
%   Trains UMAP on the training set. Test set features can be transformed
%   after training. Requires the number of neighbors, 'nNeighbors', and
%   minimum transformed distance, 'minDist', as hyperparameter inputs, as
%   well as 'n', the number of dimensions (features) after transformation.
%   Defaults are the same value as the default from the Python
%   implementation by the original inventors.
%   Inputs (required):
%       X (table): Feature table
%   Name-value inputs (required) (fixed hyperparameters):
%       n (double): the number of features in the transformed space
%           nNeighbors, default = 2
%       nNeighbors (double): number of nearest neighbors for UMAP
%           transform, 1:height(X) allowable range. A small number 
%           preserves local similarity while missing global trends, and 
%           vice versa for a large number. Default 15.
%       minDist (double): Minimum distance between points after
%           transformation. Impacts how closely similar points can be
%           located in the transformed space. Rnage of 0:1. Default 0.1
%       KeepPriorFeatures (logical): Whether or not to keep the
%           features prior to the PCA transformation, default
%           is false
%   Name-value inputs (optional) (trained hyperparameter):
%       umap (UMAP): UMAP object
%   Outputs (required):
%       X_ (table): Transformed features
%       hyp_ (cell): Name-value pair with trained hyperparameter, UMAP

% Input parsing
arguments
    X table
    % Fixed hyperparameters
    hyp.n double = 2
    hyp.nNeighbors double = 15
    hyp.minDist double = 0.1
    hyp.KeepPriorFeatures logical = false
    % Trained hyperparameters
    hyp.umap = []
end

% Grab data
vars = X.Properties.VariableNames;
x = X{:,:};

% Get the UMAP reduction
n = hyp.n;
if isempty(hyp.umap) % Training
    umap = UMAP('n_components', hyp.n, 'min_dist', hyp.minDist, 'n_neighbors', hyp.nNeighbors);
    umap = umap.fit(x);
    x_ = umap.embedding;
    hyp_ = {'umap', umap};
else % Testing
    umap = hyp.umap;
    x_ = umap.transform(x);
end

% Generate names for the new variables
newVars = repmat("umap", n, 1) + transpose(compose("%d", 1:n));

% Assemble output
if hyp.KeepPriorFeatures
    x_ = [x, x_];
    newVars = [string(vars), newVars'];
end
X_ = array2table(x_, 'VariableNames', newVars);
end