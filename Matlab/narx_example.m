
close all;

x; % signal
L = 1; % prediction length

X = embed( x(1:end-L) , 5, 1);
Y = x(1+L+5-1:end);

net = feedforwardnet([50,30,10]);
net = train(net, X',Y');
xp = predict(net,X)';

plot( (1:numel(x))',x, ...
      (1:numel(Y))'+L, xp);