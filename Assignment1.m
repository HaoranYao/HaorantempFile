%%
clear;
%%
% load the data
% create the train batch, validation batch, test batch
[train_x, train_Y, train_y] = LoadBatch('data_batch_1.mat');
[val_x, val_Y, val_y] = LoadBatch('data_batch_2.mat');
[test_x, test_Y, test_y] = LoadBatch('test_batch.mat');
train_x = Normalize(train_x);
val_x = Normalize(val_x);
test_x = Normalize(test_x);
% data_x = [];
% data_Y = [];
% data_y = [];
% numberofbatch = 5;
% for i = 1:numberofbatch
%     [temp_x, temp_Y, temp_y] = LoadBatch(['data_batch_',num2str(i),'.mat']);
%     data_x = [data_x temp_x];
%     data_Y = [data_Y temp_Y];
%     data_y = [data_y temp_y];
% end
% numberoftrain = numberofbatch * 10000 - 5000
% train_x = data_x(:,1:numberoftrain);
% train_Y = data_Y(:,1:numberoftrain);
% train_y = data_y(:,1:numberoftrain);
% val_x = data_x(:,numberoftrain+1:numberofbatch*10000);
% val_Y = data_Y(:,numberoftrain+1:numberofbatch*10000);
% val_y = data_y(:,numberoftrain+1:numberofbatch*10000);
% 
% [test_x, test_Y, test_y] = LoadBatch('test_batch.mat');


%%
% to initialize the W and b
% mean = 0
% std = 0.01
% K = 10
% d = 3072
k = 3;
m = [50 30];
[W, b,gamma,beita] = Initialize(m, 0.01, 10, 3072);
%%
% set the GD params
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 0;
%%
ns = 2*size(train_x,2)/(GDparams.n_batch);
nums_in_epochs = size(train_x,2)/(GDparams.n_batch);
GDparams.eta = 0.01;
GDparams.n_epochs = 8*ns/nums_in_epochs;
lambda = 0.01;
K = size(test_Y,1);
%%
batch_size = 10000;
[S,Sbar,H,P,mu,v] = EvaluateClassifier(train_x(:, 1:batch_size), W, b,gamma,beita);
%%
NetParams.W = W;
NetParams.b = b;
NetParams.use_bn = true;
NetParams.gammas = gamma;
NetParams.betas = beita;
Grads = ComputeGradsNumSlow_option1(train_x(:, 1:batch_size),train_Y(:, 1:batch_size), NetParams, lambda, 1e-6);
%%
[grad_W, grad_b] = ComputeGradients(train_x(:, 1:batch_size),train_Y(:, 1:batch_size), H, P, W, lambda);
check_b1 = max(abs(num_b{1} - grad_b{1})./max(abs(num_b{1}),abs(grad_b{1})));
check_W1 = max(max(abs(num_w{1} - grad_W{1})./max(abs(num_w{1}),abs(grad_W{1}))));
check_b2 = max(abs(num_b{2} - grad_b{2})./max(abs(num_b{2}),abs(grad_b{2})));
check_W2 = max(max(abs(num_w{2} - grad_W{2})./max(abs(num_w{2}),abs(grad_W{2}))));
check_b3 = max(abs(num_b{3} - grad_b{3})./max(abs(num_b{3}),abs(grad_b{3})));
check_W3 = max(max(abs(num_w{3} - grad_W{3})./max(abs(num_w{3}),abs(grad_W{3}))));
%%
% random search 
n_pairs = 50;
l_max = -1;
l_min = -5;
Lambda = [];
acc_va = [];
acc_test1 = [];
for i = 1 : n_pairs
    l = l_min + (l_max - l_min) * rand(1, 1);
    lambda = 10^l;
    [W, b] = Initialize(m, 0.01, 10, 3072);
    [W, b, Loss_train, Loss_val, acc_train, acc_test] = ...
    train(W, b,train_x, train_y, train_Y,val_x, val_y, val_Y, test_x,test_Y,test_y,GDparams, lambda);
    Lambda = [Lambda, lambda];
    acc_va = [acc_va, ComputeAccuracy(val_x, val_y, W, b)];
    acc_test1 = [acc_test1, ComputeAccuracy(test_x, test_y, W, b)];
    
end
%%
% %uniform search
% n_pairs = 30;
% l_max = 10^(-8.5);
% l_min = 10^(-10);
% Lambda = [];
% acc_va = [];
% acc_test1 = [];
% for i = 1 : n_pairs
%     l = l_min + (i-1)*(l_max - l_min)/29;
%     [W, b] = Initialize(m, 0.01, 10, 3072);
%     [W, b, Loss_train, Loss_val, acc_train, acc_test] = ...
%     train(W, b,train_x, train_y, train_Y,val_x, val_y, val_Y, test_x,test_Y,test_y,GDparams, lambda);
%     Lambda = [Lambda, lambda];
%     acc_va = [acc_va, ComputeAccuracy(val_x, val_y, W, b)];
%     acc_test1 = [acc_test1, ComputeAccuracy(test_x, test_y, W, b)];
%     
% end
%%
[W, b, Loss_train, Loss_val, acc_train, acc_test] = ...
train(W, b,gamma,beita,train_x, train_y, train_Y,val_x, val_y, val_Y, test_x,test_Y,test_y,GDparams, lambda,k, m);
%%
figure()
figurex = 1:GDparams.n_epochs;
plot(figurex, Loss_train(2:GDparams.n_epochs+1), 'r');
hold on;
plot(figurex, Loss_val(2:GDparams.n_epochs+1), 'b');
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