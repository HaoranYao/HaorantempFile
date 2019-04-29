function [grad_W, grad_b,grad_gamma,grad_beita] = ComputeGradientss(X, Y, P, h, S, W, lambda, k, mu, v)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - P:                K*N
%           - W:                K*d
%           - lambda:           1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1

grad_Wk = zeros(size(Y, 1), size(h{end}, 1));
grad_bk = zeros(size(Y, 1), 1);
g_prev = zeros(size(X, 2), size(h{end}, 1));
eps = 0.001;

for i = 1 : size(X, 2)
    Pi = P(:, i);
    hi = h{end}(:, i);
    Yi = Y(:, i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_bk = grad_bk + g';
    grad_Wk = grad_Wk + g'*hi';
    
    g = g*W{end};
    g_prev(i, :) = g*diag(hi > 0);  
end

grad_W{k} = 2*lambda*W{end} + grad_Wk/size(X, 2);
grad_b{k} = grad_bk/size(X, 2);

for l = k - 1 : -1 : 1
    % batch normalization for backpass
    if nargin == 10
        g_prev = BN_backward(g_prev, mu{l}, v{l}, S{l}, eps);
    end

    grad_b{l} = mean(g_prev)';
    if l == 1
        grad_W{l} = g_prev'*X';
    else
        grad_W{l} = g_prev'*h{l - 1}';
    end
    grad_W{l} = grad_W{l}/size(X, 2) + 2*lambda*W{l};
    if l > 1
        g_prev = g_prev*W{l};
        g_prev = g_prev.*(h{l - 1} > 0)';       % equvalent to gi*diag(Ind(si>0))
    end    
end
grad_gamma = 0;
grad_beita = 0;
end