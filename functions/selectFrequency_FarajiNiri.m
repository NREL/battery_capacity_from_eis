function X_ = selectFrequency_FarajiNiri(X, hyp)
arguments
    X table
    hyp.idxFreq double
end
% Grab data
vars = X.Properties.VariableNames;
x = X{:,:};
% Pull out frequency features
idxFreq = hyp.idxFreq;
if isrow(idxFreq)
    idxFreq = idxFreq';
end
idx = [0,61] + idxFreq;
idx = reshape(idx, [], 1);
vars_ = vars(idx);
x_ = x(:, idx);
% Return
X_ = array2table(x_, 'VariableNames', vars_);
end