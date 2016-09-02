function [n, f] = f_impulse(n1, n2, shift)
%n1 = input('Lower limit: ')
%n2 = input('Upper limit: ')
if nargin == 2
	shift = 0;
end

n = n1:n2;
f = [n==shift];

figure;
stem(n,f);