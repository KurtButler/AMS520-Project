
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
    data = [natgas.STOCKS,natgas.JFKTEMP, natgas.CLTTEMP, natgas.ORDTEMP, natgas.HOUTEMP, natgas.LAXTEMP, natgas.NXT_CNG_STK];
    weeks = (1:size(data,1))';

% Normalize
data = normalize(data);

% Detrending
datatrend = sgolayfilt( data, 3, 11);
datadetrended = data - datatrend;

% Trend predictions
trendp = zeros(size(datatrend,1),1);
for n = 11:size(trendp,1)
    segment = [ datatrend(n-10:n-1,end); 0];
    segment(end) = segment(end-1) + (segment(end-1)-segment(end-2));
    segment = sgolayfilt( segment, 3, 11);
    trendp(n) =  segment(end);
end

M = embed(datadetrended,3,1);
X = M(:,1:end-1);
y = M(:,end);

% OLS
b = X\y;
yp = X*b;

% % NARX
% gp = fitrgp(X,y);
% yp2 = resubPredict(gp);

subplot(3,1,1)
plot(weeks,datadetrended,'LineWidth',1.5);
grid on
legend('Stocks','JFK Temp','CLT Temp','ORD Temp','HOU Temp','LAX Temp','Next Change Stock')

subplot(3,1,2)
plot(weeks,data(:,end),'k', weeks,datatrend(:,end),'g',weeks,trendp,'r','LineWidth',1.5);
grid on
legend('Target','SG-trend')

subplot(3,1,3)
t = (1:size(y,1))';
plot(t,y,'k',t,yp,'r','LineWidth',1.5);
grid on
