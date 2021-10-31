
close all;

if ~exist('data','var')
    if exist('C:\Users\kurtb\Documents\MATLAB\AMS520-Project\Matlab','dir')
        addpath(genpath('C:\Users\kurtb\Documents\MATLAB\AMS520-Project\Matlab'))
    end
    if exist('/Users/kbutler/Documents/MATLAB/AMS Project','dir')
        addpath(genpath('/Users/kbutler/Documents/MATLAB/AMS Project'))
    end
    addpath C:\Users\kurtb\Documents\MATLAB\topology
    import_gas;
end
    % Select some features
    % For now I'm gonna use the easy stuff
    data = [natgas.STOCKS,natgas.JFKTEMP, natgas.CLTTEMP, natgas.ORDTEMP, natgas.HOUTEMP, natgas.LAXTEMP];
    target = natgas.NXT_CNG_STK;
    weeks = (1:size(data,1))';

% Normalize
data = normalize(data);

% OLS
X = embed(data,3,1);
T = embed(target,3,1);
y = T(:,end);

b = X\y;
yp = X*b;

%% VARX model 2
data = table2array(natgas);
data = data(341:end,2:9);
target = natgas.NXT_CNG_STK(341:end);


% Normalize
data = normalize(data);

% OLS
Q = 3;
X = embed(data,Q,1);
T = embed(target,Q,1);
y0 = T(:,end);
t2 = (1:numel(y0))' + weeks(end) - numel(y0) - Q+1;

% Remove NaN
id = find(isnan(sum(X,2)));
X(id,:) = [];
y0(id) = [];
t2(id) = [];

b = X\y0;
yp2 = X*b;

%% Plotting
t = (1:size(y,1))';
plot(t,y,'k',t,yp,'r',t2,yp2,'-.','LineWidth',1.5);
grid on
legend('Raw targets','VARX model 1','VARX model 2','Location','best')
