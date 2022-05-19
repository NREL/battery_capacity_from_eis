function X_ = generateFeaturesGraphicalM10C(X, hyp)
%GENERATEFEATURESGRAPHICALM10C Graphical features for -10C data.
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

% frequency vector
freq = [31622.7800000000,25118.8600000000,19952.6200000000,15848.9300000000,12589.2500000000,10000,7943.28200000000,6309.57300000000,5011.87200000000,3981.07200000000,3162.27800000000,2511.88600000000,1995.26200000000,1584.89300000000,1258.92500000000,1000,794.328200000000,630.957300000000,501.187200000000,398.107200000000,316.227800000000,251.188600000000,199.526200000000,158.489300000000,125.892500000000,100,79.4328200000000,63.0957300000000,50.1187200000000,39.8107200000000,31.6227800000000,25.1188600000000,19.9526200000000,15.8489300000000,12.5892500000000,10,7.94328000000000,6.30957000000000,5.01187000000000,3.98107000000000,3.16228000000000,2.51189000000000,1.99526000000000,1.58489000000000,1.25893000000000,1,0.794330000000000,0.630960000000000,0.501190000000000,0.398110000000000,0.316230000000000,0.251190000000000,0.199530000000000,0.158490000000000,0.125890000000000,0.100000000000000,0.0794300000000000,0.0631000000000000,0.0501200000000000,0.0398100000000000,0.0316200000000000,0.0251200000000000,0.0199500000000000,0.0158500000000000,0.0125900000000000,0.0100000000000000,0.00794000000000000,0.00631000000000000,0.00501000000000000];

% feature 1: Z values at f=1E4
idx1 = 6;
idx1 = [0:69:275] + idx1;
xout1 = x(:,idx1);

% feature 2: min(Zreal) at f<1E4
x1_ = x1; x1_(:,[1:6]) = Inf;
[~, idx2] = min(x1_, [], 2);
idx2_ = sub2ind(size(x1), (1:size(x1,1))', idx2);
xout2 = [freq(idx2)', x1(idx2_), x2(idx2_), x3(idx2_), x4(idx2_)];

% feature 3: max(-Zimag(f<10^2 & f>10^-1))
x2_ = x2; x2_(:,[1:25,60:69]) = Inf;
[~, idx3] = max(-1.*x2_, [], 2);
idx3_ = sub2ind(size(x2), (1:size(x2,1))', idx3);
xout3 = [freq(idx3)', x1(idx3_), x2(idx3_), x3(idx3_), x4(idx3_)];

% feature 4: min(-Zimag(f<10^0 & f>10^-2)
x2_ = x2; x2_(:,[1:50]) = -Inf;
[~, idx4] = min(-1.*x2_, [], 2);
idx4_ = sub2ind(size(x2), (1:size(x2,1))', idx4);
xout4 = [freq(idx4)', x1(idx4_), x2(idx4_), x3(idx4_), x4(idx4_)];

% feature 5: Z values at f(end)
idx5 = 69;
idx5 = [0:69:275] + idx5;
xout5 = x(:,idx5);

% Assemble output
x_ = [xout1, xout2, xout3, xout4, xout5];
newVars = ["Zreal1","Zimag1","Zmag1","Zphz1",...
    "freq2","Zreal2","Zimag2","Zmag2","Zphz2",...
    "freq3","Zreal3","Zimag3","Zmag3","Zphz3",...
    "freq4","Zreal4","Zimag4","Zmag4","Zphz4",...
    "Zreal5","Zimag5","Zmag5","Zphz5"];
if hyp.KeepPriorFeatures
    x_ = [x, x_];
    newVars = [string(vars), newVars];
end
X_ = array2table(x_, 'VariableNames', newVars);
end