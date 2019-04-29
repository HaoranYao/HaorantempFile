function [new_x, new_y, new_Y] = shuffle(train_x, train_y, train_Y)
% shuffle the train_set and the corressponding label
% N - the number of images(10000)
% K - the number of labels(10)
% d - the dimensionality of each image (3072 = 32*32*3)
% X - d * N - the image pixel data, entries between 0~1 
% Y - K * N - contains the one-hot representation of the label of image
% y - 1 * N - contains the label for each image encoded from 1~10
N = size(train_x,2);
randIndex = randperm(N);
new_x = train_x(:,randIndex);
new_y = train_y(:,randIndex);
new_Y = train_Y(:,randIndex);
end
