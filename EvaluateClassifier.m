function [S,Sbar,H,P,mu,v] = EvaluateClassifier(X, W, b,gamma,beita, mu_a, v_a)
% to evaluate the network
% In this part we will do:
% s = W*X + b
% p - K * n - softmax(s)
% N - the number of images(10000)
% K - the number of labels(10)
% d - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% b1 - m * 1
% b2 - K * 1
% W1 -  m * d
% s1 - m * N
% W2 - K * m
% P - K * N
%%
% set the parameters

N = size(X,2);
k = size(W,2);
eps = 1e-3;
for i = 1:k -1
    Wi = W{i};
    bi = repmat(b{i}, 1, size(X, 2));
    S{i} = Wi*X + bi;
    if nargin == 7
        [newx, mu_i, v_i]= BNforward(S{i},eps,mu_a{i},v_a{i});
    else
        [newx, mu_i, v_i]= BNforward(S{i},eps);
    end
    Sbar{i} = newx;
    gammai = repmat(gamma{i},1,size(newx,2));
    beitai = repmat(beita{i},1,size(newx,2));
    newxf = newx.*gammai + beitai;
%     newxf = newx;
    mu{i} = mu_i;
    v{i} = v_i;
    H{i} = max(0,newxf);
    X = H{i};
end
%%
% calculate P
b = repmat(b{end}, 1, size(H{end}, 2));
X = H{end};
W = W{end};
Ss = W*X+ b;
ex = exp(Ss);
temp2 = sum(ex);
sigma = repmat(temp2, size(Ss,1), 1);
P = ex./sigma;

end

