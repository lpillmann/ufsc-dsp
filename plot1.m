figure
X = linspace(0,2*pi,50)';
Y = [cos(X), 0.5*sin(X)];
stem(X,Y)

figure
X = linspace(0,10,20)';
Y = (exp(0.25*X));
stem(X,Y,'filled')

figure
X = linspace(0,2*pi,50);
Y = exp(0.3*X).*sin(3*X);
h = stem(X,Y);