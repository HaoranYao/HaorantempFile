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
