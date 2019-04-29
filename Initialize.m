function [W, b,gamma,beita] = Initialize(m, std, K, d)
% X - d * N
% W1 -  m * d
% s1 - m * N
% W2 - K * m
% P - K * N
para = [d, m, K]
for i = 1: size(para,2) -1
   
    W{i} = 0 + randn(para(i + 1), para(i)) * std;
    b{i} = zeros(para(i + 1), 1);
end

for i = 1:size(m,2)
    gamma{i} = 0 + randn(m(i),1)*std;
    beita{i} = 0 + randn(m(i),1)*std;
end

