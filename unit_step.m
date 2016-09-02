n1 = input('Lower limit: ')
n2 = input('Upper limit: ')
n = n1:n2;
y = [n>=0];

figure;
stem(n,y);
title('Unit Step Signal - Discrete');