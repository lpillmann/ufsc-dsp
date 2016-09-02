x_min = -10;
x_max = 10;

y_min = -2;
y_max = 2;


x = [0 1 1 1 1 1 1 1 1 1 1];
y = [0 1 1 1 1 1 1 1 1 1 1];
%y = [1 1];

clin = conv(x,y);

%figure;

subplot(3,1,1);
stem(x);
axis([x_min x_max y_min y_max]);

subplot(3,1,2);
stem(y);
axis([x_min x_max y_min y_max]);

subplot(3,1,3);
stem(clin);
axis([x_min x_max]);