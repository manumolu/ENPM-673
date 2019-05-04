x = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8];
y = [0,250,420,730,670,450,310,600,820,820,690,170,0,0,0,0,0];

% x = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55];
% y = [670,450,310,600,820,820,690,170];

p =polyfit(x,y,17);

y1 = polyval(p,x);
figure
plot(x,y,'o');
hold on
plot(x,y1)
hold off
title('Polynomial Plot');
Q = trapz(x,y);



q = polyint(p);
a = 0.25
b = 0.6;
I = diff(polyval(q,[a b]));

a1 = 0.2;
b1 = 0.55;
I1 = diff(polyval(q,[a1 b1]));

% f = fit(x.',y.','gauss2');
% figure;
% plot(f,x,y)
% title('Curve Fit Plot');