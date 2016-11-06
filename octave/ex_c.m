% exercise (c)

n = 10;

x = f_impulse(-n,n);
h = f_step(-n,n,1);
%w = f_impulse(-n,n,1) - f_impulse(-n,n);

%H = conv(h,w);
y = conv(x,h);

close all;

stem(n,y);

