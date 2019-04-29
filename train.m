function [W, b, gamma,beita,Loss_train, Loss_val, acc_train, acc_test] = train(W, b,gamma,beita,train_x, train_y, train_Y,val_x, val_y, val_Y, test_x,test_Y,test_y,GDparams, lambda,k, m)
Loss_train = ComputeCost(train_x, train_Y, W, b, lambda,gamma,beita);
Loss_val = ComputeCost(val_x, val_Y, W, b, lambda,gamma,beita);
acc_train = ComputeAccuracy(train_x, train_y, W, b,gamma,beita);
acc_test = ComputeAccuracy(test_x, test_y, W, b,gamma,beita);
for i = 1:GDparams.n_epochs
     [W, b,gamma,beita,m_out,v_out] = MiniBatchGD(train_x, train_Y, GDparams, W, b,gamma,beita, lambda,i);
     newLoss_train = ComputeCost(train_x, train_Y, W, b, lambda,gamma,beita,m_out,v_out);
     Loss_train = [Loss_train newLoss_train];
     newLoss_val = ComputeCost(val_x, val_Y, W, b, lambda,gamma,beita,m_out,v_out);
     Loss_val = [Loss_val newLoss_val]; 
     new_acc_tr = ComputeAccuracy(train_x, train_y, W,b,gamma,beita,m_out,v_out);
     acc_train = [acc_train new_acc_tr];
     new_acc_te = ComputeAccuracy(test_x, test_y, W, b,gamma,beita,m_out,v_out);
     acc_test = [acc_test new_acc_te];
     disp('=================================');
     disp(['epoch: ' num2str(i)]);
     disp(['train set loss: ' num2str(newLoss_train)]);
     disp(['validation set loss: ' num2str(newLoss_val)]);
     disp(['train set accuracy: ' num2str(new_acc_tr * 100) '%']);
     disp(['test set accuracy: ' num2str(new_acc_te * 100) '%']);
end
end