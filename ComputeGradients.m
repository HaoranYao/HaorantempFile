function [grad_W, grad_b,grad_gamma,grad_beita] = ComputeGradients(X, Y, s_batch,Sbar,H_batch, P, W, lambda,u,v)
% compute the gradients for the Loss to W and b using the efficient way
% N - the number of images in this batch
% K - the number of labels(10)
% d - 1 * 1 - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% Y - K * N - contains the one-hot representation of the label of image
% P - K * N - the result of the softMax output
% grad_W - K * d - the gradient of Loss to W
% grad_b - K * 1 - the gradient of Loss to b
% b1 - m * 1
% b2 - K * 1
% W1 -  m * d
% s1 - m * N
% W2 - K * m
% P - K * N
% b1 = b{1};
% b1 = repmat(b1,1,N);
% b2 = b{2};
% b2 = repmat(b2,1,N);
% m = size(W2, 2);
% d = size(X, 1);
% K = size(Y, 1);
k = size(W, 2);
N = size(X, 2);
G_batch = P - Y;
for i = 1:k
    grad_L_W{i}=0;
end
for i = 2:k
    vec = ones(N, 1);
    j = k - i + 2;
    grad_L_W{j} = (G_batch * H_batch{j-1}')./ N + 2*lambda*W{j};
    grad_L_b{j} = (G_batch * vec)./N;
    G_batch = W{j}' * G_batch;
    flag = zeros(size(H_batch{j-1}));
    flag(H_batch{j-1}>0) = 1;
    G_batch = G_batch.*flag;
%     G_batch = BactchNormBackPass(G_batch,s_batch{j-1},u{j-1},v{j-1});
    G_batch = BN_backward(G_batch',u{j-1},v{j-1},s_batch{j-1},1e-6);
    G_batch = G_batch';
    grad_gamma{j-1} = 1/N*(G_batch.*Sbar{j-1})*vec;
    grad_beita{j-1} = 1/N*G_batch*vec;
end
grad_L_W{1} = (G_batch * X')./N + 2*lambda*W{1};
grad_L_b{1} = (G_batch * vec)./N;
grad_W = grad_L_W;
grad_b = grad_L_b;
end
