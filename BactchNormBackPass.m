function g = BactchNormBackPass(g, S, u, v)
eps = 1e-6;
vec = ones(1,size(S,1))';
vecn = ones(1,size(S,2));
sigma1 = ((v+eps).^(-0.5));
sigma2 = ((v+eps).^(-1.5));
G1 = g.*(sigma1*vecn);
G2 = g.*(sigma2*vecn);
D = S - u*vecn;
c = (G2.*D)*(vecn');
g = G1 -1/size(vecn,2)*G1-D.*(c*vecn);

