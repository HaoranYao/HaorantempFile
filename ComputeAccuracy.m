function acc = ComputeAccuracy(X, y, W, b,gamma,beita,m,v)
% computes the accuracy of the network¡¯s predictions 
% acc - 1 * 1 -  the percentage of examples for which it gets the correct answer
if nargin == 8 
    [S,Sbar,H,P,mu,v] = EvaluateClassifier(X, W, b,gamma,beita, m, v);
else
    [S,Sbar,H,P,mu,v] = EvaluateClassifier(X, W, b,gamma,beita);
end
[~, max_index] = max(P);
count = 0;
for i = 1:size(y,2)
    if max_index(i) ~= y(i)
        count = count + 1;
    end
end
acc = 1 - (count / length(y));
end
