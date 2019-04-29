function [Wstar, bstar,gammastar,beitastar,m_out,v_out] = MiniBatchGD(X, Y, GDparams, W, b, gamma,beita,lambda, ne)
% X - d * N - the image pixel data, entries between 0~1 
% Y - K * N - contains the one-hot representation of the label of image
% W - K * d - the weights of the network
% b - K * 1 - the bias
% GDparams - object - contains the n_batch, eta, n_epochs
n_batch = GDparams.n_batch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs;
N = size(X, 2);
d = size(X, 1);
K = size(Y, 1);
k = size(W,2);
t = (ne-1)*N/n_batch;
alpha = 0.9;
for j = 1 : N/n_batch
    t = t+j-1;
    eta = update_eta(900,1e-5,1e-1,t);
    j_start = (j - 1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start:j_end;
    Xbatch = X(:, j_start:j_end);
    Ybatch = Y(:, j_start:j_end);
    [S,Sbar,H,P,mu,v] = EvaluateClassifier(Xbatch, W, b,gamma,beita);
    for i = 1 : k - 1
        if j == 1
            m_out = mu;
            v_out = v;
        else
            m_out{i} = alpha*m_out{i} + (1 - alpha)*mu{i};
            v_out{i} = alpha*v_out{i} + (1 - alpha)*v{i};
        end
    end
%     [grad_W, grad_b,grad_gamma,grad_beita] = ComputeGradientss(Xbatch, Ybatch,P, H, S, W, lambda,3,mu,v);
    [grad_W, grad_b,grad_gamma,grad_beita] = ComputeGradients(Xbatch, Ybatch,S,Sbar, H, P, W, lambda,mu,v);
    for i = 1:length(grad_W)
        W{i} = W{i}-eta*grad_W{i};
        b{i} = b{i} - eta*grad_b{i};
    end
    for i = 1:length(grad_gamma)
        gamma{i} = gamma{i} - eta*grad_gamma{i};
        beita{i} = beita{i} - eta*grad_beita{i};
    end
        
end
Wstar = W;
bstar = b;
gammastar = gamma;
beitastar = beita;
end
