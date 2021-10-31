
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
stocks = [natgas.STOCKS];
next = [natgas.NXT_CNG_STK];
weeks = (1:size(data,1))';

subplot(2,1,1)
plot(weeks,stocks)
grid on 

subplot(2,1,2)
diff = @(x) [x(2:end)-x(1:end-1);0];
plot(weeks,diff(stocks), weeks, next);
grid on