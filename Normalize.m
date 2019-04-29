function data_x = Normalize(data_input)
mean_input = mean(data_input, 2);
data_x = data_input - repmat(mean_input,[1,size(data_input,2)]);
end
