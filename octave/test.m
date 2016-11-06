N = 20; % total samples
r = N/2; % "false zero"


f = zeros(1,N); %initialize f
n = 1:N;

% P1 13.2 Q1
x = cos(pi*n); 
s = [n>=r]; % unit step
x = x.*s;

for i = r:N
    f(i) = 0.25 * (x(i-2) + x(i-3) + x(i-4) + x(i-5)); % P1 13.2 1
end

stem(n,f)