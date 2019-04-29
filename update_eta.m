function neweta = update_eta(n_s,eta_min,eta_max,t)
temp = mod(t,2*n_s);
if temp<n_s
    neweta = eta_min + temp*(eta_max-eta_min)/n_s;
else
    neweta = eta_max - (temp-n_s)*(eta_max-eta_min)/n_s;
end
