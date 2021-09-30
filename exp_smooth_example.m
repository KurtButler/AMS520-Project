

close all;

N = 100;

a = 0.4;

nn = (1:N)';
x = conv( randn(N,1), normpdf([-1:0.1:1]), 'same');

%% ES
s = zeros(size(x));
f = nan(size(x));
s(1) = x(1);
for n = 2:N
    s(n) = a*x(n) + (1-a)*s(n-1);
end
for n = 1:N-1
    f(n+1) = (1-a)*s(n);
end

%% ES forecasts


%% Plotting
figure(12)

plot(nn,x,nn,f,'LineWidth',2);

grid on
title(sprintf('Exp smoothing \na=%0.2f',a))