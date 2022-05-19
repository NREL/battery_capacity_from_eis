function X_ = generateFeaturesStatistical(X, hyp)
%GENERATEFEATURESSTATISTICAL Generate statistical features.
%   Generates statistical features for Zreal, Zimag, Zmag, and Zphz.
%   Inputs (required):
%       X (table): Feature table
%   Name-value inputs (required) (fixed hyperparameters):
%       KeepPriorFeatures (logical): Whether or not to keep the
%           features prior to the PCA transformation, default
%           is false.
%   Outputs (required):
%       X_ (table): Transformed features

% Input parsing
arguments
    X table
    % Fixed hyperparameters
    hyp.KeepPriorFeatures logical = false
end

% Grab data
vars = X.Properties.VariableNames;
x = X{:,:};
% Split Zreal, Zimag, Zmag, Zphz
w = size(x,2)/4;
x1 = x(:,1:w);
x2 = x(:,w+1:(w*2));
x3 = x(:,(w*2+1):(w*3));
x4 = x(:,(w*3+1):end);
% Calculate statistics for each variable
x1_ = [var(x1,[],2), mean(x1,2), median(x1,2), iqr(x1,2), mad(x1,0,2), mad(x1,1,2), range(x1,2)];
x2_ = [var(x2,[],2), mean(x2,2), median(x2,2), iqr(x2,2), mad(x2,0,2), mad(x2,1,2), range(x2,2)];
x3_ = [var(x3,[],2), mean(x3,2), median(x3,2), iqr(x3,2), mad(x3,0,2), mad(x3,1,2), range(x3,2)];
x4_ = [var(x4,[],2), mean(x4,2), median(x4,2), iqr(x4,2), mad(x4,0,2), mad(x4,1,2), range(x4,2)];
x_ = [x1_,x2_,x3_,x4_];

% Generate names for the new variables
statsFuncs = ["var_","mean_","median_","IQR_","MAD_","MdAD_","range_"];
varTypes = ["Zreal";"Zimag";"Zmag";"Zphz"];
newVars = statsFuncs+varTypes; newVars = newVars';
newVars = reshape(newVars, [], 1);

% Assemble output
if hyp.KeepPriorFeatures
    x_ = [x, x_];
    newVars = [string(vars), newVars'];
end
X_ = array2table(x_, 'VariableNames', newVars);
end