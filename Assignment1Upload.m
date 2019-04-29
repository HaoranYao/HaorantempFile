%%
clear;
%%
% load the data
% create the train batch, validation batch, test batch
[train_x, train_Y, train_y] = LoadBatch('data_batch_1.mat');
[val_x, val_Y, val_y] = LoadBatch('data_batch_2.mat');
[test_x, test_Y, test_y] = LoadBatch('test_batch.mat');

%%
% to initialize the W and b
% mean = 0
% std = 0.01
% K = 10
% d = 3072
[W, b] = Initialize(0, 0.01, 10, 3072);
%%
% set the GD params
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;
lambda = 1;
K = size(test_Y,1);
%%
%compute the initial accuracy for train data and test data 
%compute the initial loss for train data and validation data
Loss_train = ComputeCost(train_x, train_Y, W, b, lambda);
Loss_val = ComputeCost(val_x, val_Y, W, b, lambda);
acc_train = ComputeAccuracy(train_x, train_y, W, b);
acc_test = ComputeAccuracy(test_x, test_y, W, b);

%%
batch_size = 1;
P = EvaluateClassifier(train_x(:, 1:batch_size), W, b);
[num_b, num_w] = ComputeGradsNumSlow(train_x(:, 1:batch_size),train_Y(:, 1:batch_size), W, b, lambda, 1e-6);
[grad_W, grad_b] = ComputeGradients(train_x(:, 1:batch_size),train_Y(:, 1:batch_size), P, W, lambda);
check_b = max(abs(num_b - grad_b)./max(abs(num_b),abs(grad_b)));
check_W = max(max(abs(num_w - grad_W)./max(abs(num_w),abs(grad_W))));
%%
for i = 1:GDparams.n_epochs
     [W, b] = MiniBatchGD(train_x, train_Y, GDparams, W, b, lambda);
     newLoss_train = ComputeCost(train_x, train_Y, W, b, lambda);
     Loss_train = [Loss_train newLoss_train];
     newLoss_val = ComputeCost(val_x, val_Y, W, b, lambda);
     Loss_val = [Loss_val newLoss_val]; 
     new_acc_tr = ComputeAccuracy(train_x, train_y, W,b);
     acc_train = [acc_train new_acc_tr];
     new_acc_te = ComputeAccuracy(test_x, test_y, W, b);
     acc_test = [acc_test new_acc_te];
     disp('=================================');
     disp(['epoch: ' num2str(i)]);
     disp(['train set loss: ' num2str(newLoss_train)]);
     disp(['validation set loss: ' num2str(newLoss_val)]);
     disp(['train set accuracy: ' num2str(new_acc_tr * 100) '%']);
     disp(['test set accuracy: ' num2str(new_acc_te * 100) '%']);
end
%%
figure()
figurex = 1:GDparams.n_epochs;
plot(figurex, Loss_train(2:41), 'r');
hold on;
plot(figurex, Loss_val(2:41), 'b');
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');
%%
for i = 1 : 10
im = reshape(W(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure()
montage(s_im, 'size', [1, 10])

%%
function acc = ComputeAccuracy(X, y, W, b)
% computes the accuracy of the network¡¯s predictions 
% acc - 1 * 1 -  the percentage of examples for which it gets the correct answer
P = EvaluateClassifier(X, W, b);
[max_value, max_index] = max(P);
count = 0;
for i = 1:10000
    if max_index(i) ~= y(i)
        count = count + 1;
    end
end
acc = 1 - (count / length(y));
end
%%
function J = ComputeCost(X, Y, W, b, lambda)
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
N = size(X, 2);
P = EvaluateClassifier(X, W, b);
temp1 = Y' * P;
% the number on the diagnol of Y is the loss
temp2 = -log(diag(temp1));
cross_loss = sum(temp2)/N;
regu_loss = lambda * sum(sum(W.^2));
J = cross_loss + regu_loss;
end
%%
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
% compute the gradients for the Loss to W and b using the efficient way
% N - the number of images in this batch
% K - the number of labels(10)
% d - 1 * 1 - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% Y - K * N - contains the one-hot representation of the label of image
% W - K * d - the weights of the network
% b - K * 1 - the bias
% P - K * N - the result of the softMax output
% grad_W - K * d - the gradient of Loss to W
% grad_b - K * 1 - the gradient of Loss to b
N = size(X, 2);
d = size(X, 1);
K = size(Y, 1);
grad_W = zeros(K, d);
grad_b = zeros(K, 1);
G_batch = P - Y;
grad_L = (G_batch * X')./ N;
grad_W = 2 * lambda * W + grad_L;
vec = ones(N, 1);
grad_b = (G_batch * vec)./ N;
end
%%
function P = EvaluateClassifier(X, W, b)
% to evaluate the network
% In this part we will do:
% s = W*X + b
% p - K * n - softmax(s)
% N - the number of images(10000)
% K - the number of labels(10)
% d - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% W - K * d - the weights of the network
% b - K * 1 - the bias
%
% set the parameters
N = size(X,2);
K = size(W,1);
%
% calculate P
temp = W * X;
bias = repmat(b, 1, N);
s = temp + bias;
ex = exp(s);
temp2 = sum(ex);
sigma = repmat(temp2, K, 1);
P = ex./sigma;
end
%%
function [W, b] = Initialize(mean, std, K, d)
W = mean + randn(K, d) * std;
b = mean + randn(K, 1) * std;
end
%%
function [X, Y, y] = LoadBatch(filename)
% Read in the data from a CIFAR-10 batch file and returns the image and
% label data in separate files
% N - the number of images(10000)
% K - the number of labels(10)
% d - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% Y - K * N - contains the one-hot representation of the label of image
% y - 1 * N - contains the label for each image encoded from 1~10
%% 
% load the data and extract the data and labels
% change the label range from 0~9 to 1~10
rawData = load(filename);
data = rawData.data';
label = rawData.labels;
newlabel = label + 1;
%%
% create the onehot label 
onehot = zeros(10,10000);
%onehot = zeros(K,N);
for i = 1:10000
    onehot(newlabel(i),i)=1;
end
%%
X = double(data)/255;
Y = double(onehot);
y = double(newlabel');
end
%%
function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
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
for j = 1 : N/n_batch
    j_start = (j - 1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start:j_end;
    Xbatch = X(:, j_start:j_end);
    Ybatch = Y(:, j_start:j_end);
    P = EvaluateClassifier(Xbatch, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
    W = W - eta*grad_W;
    b = b - eta*grad_b;
end
Wstar = W;
bstar = b;
end
