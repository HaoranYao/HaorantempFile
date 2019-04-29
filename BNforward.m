function [newx, mu, v]= BNforward(s,eps,mu_a,v_a)
if nargin == 4
    mu = mu_a;
    v = v_a;
else
    mu = mean(s,2);
    v = var(s,0,2).*(size(s,2)-1)/size(s,2);
end
newx = diag((v + eps).^(-0.5))*(s - repmat(mu, 1, size(s, 2)));
end