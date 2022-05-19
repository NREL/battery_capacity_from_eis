function X_ = selectOnlyZrealZimag(X, hyp)
arguments
    X table
    hyp = []
end
% Grab data
vars = X.Properties.VariableNames;
x = X{:,:};
% Pull out Zreal and Zimag
idx = 1:138;
vars_ = vars(idx);
x_ = x(:, idx);
% Return
X_ = array2table(x_, 'VariableNames', vars_);
end