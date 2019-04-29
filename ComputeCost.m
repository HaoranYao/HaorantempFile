function J = ComputeCost(X, Y, W, b, lambda, gamma,beita,mu_a, v_a)
% to compute the cost of the network using cross-entropy function plus a
% regularization term on W
% J - 1 * 1 - the sum of the loss of the network¡¯s predictions for the images in X 
% relative to the ground truth labels and the regularization term on W
% N - the number of images(10000)
% K - the number of labels(10)
% d - 1 * 1 - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% Y - K * N - contains the one-hot representation of the label of image
% W - K * d - the weights of the network
% b - K * 1 - the bias
if nargin == 9
    [S,Sbar,H,P,mu,v] = EvaluateClassifier(X, W, b,gamma,beita, mu_a, v_a);
else
    [S,Sbar,H,P,mu,v] = EvaluateClassifier(X, W, b,gamma,beita);
end
N = size(X, 2);
temp1 = Y' * P;
% the number on the diagnol of Y is the loss
temp2 = -log(diag(temp1));
cross_loss = sum(temp2)/N;
regu_loss = 0;
for i = 1: length(W)
    temp_loss = lambda * sum(sum(W{1}.^2));
    regu_loss = regu_loss + temp_loss;
    
end
J = cross_loss + regu_loss;
end