% Modified from an implementation by Laurens van der Maaten, 2010
n = 500;
t = (2*pi) * (1 + 2 * rand(n, 1));
col = t;  
height = 15 * rand(n, 1);
X = [t .* cos(t) height t .* sin(t)] + .001 * randn(n, 3);
t = [t height];