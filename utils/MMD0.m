function [M0] = MMD0(Xs, Xt, C, Ws, Wt)
% TODO : Unified MMD function
% Marginal MMD matrix for Ball and Centers
% Inputs:
%%% Xs  : Source domain feature matrix, dim * ns
%%% Xt  : Target domain feature matrix, dim * nt
%%% Ws  : Source domain transform matrix, ns * new_ns
%%% Wt  : Source domain transform matrix, nt * new_nt
%%% C   : the number of labels
%%% new_X = X * W
ns = size(Xs,2);
nt = size(Xt,2);
if nargin == 3 
    Ws = eye(ns);
    Wt = eye(nt);
end

e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
e = blkdiag(Ws,Wt) * e;
M0 = e * e' * C;  % multiply C for better normalization
end

