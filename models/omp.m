function [A_hat, stats] = omp(Z, Psi, K, thres, prior_nz)
%
% Performs Orthogonal Matching Pursuit.
%
% ---- INPUT PARAMETERS ----
%   Z: matrix with the signals, M*PxNxR
%   Psi: matrix with the intervention signals, M*PxQxR
%   K: maximum number of nonzero elements, 1x1
%   thres: threshold involved in the stopping criterion and related with
%          the variance of the noise, 1x1
%   prior_nz: logical matrix with prior knowledge about the structure, NxQ
%
% ---- OUTPUT PARAMETERS ----
%   A_hat: estimated coeeficient matrix, NxQ
%   stats.cur_err: currect error of OMP, NxK+1
%   stats.full_err: currect error of OMP, Nx1
%
% (c) yannis.pantazis@gmail.com, CausalPath, CSD, UOC, 2016
%


% initialize
N = size(Z, 2);
MP = size(Psi, 1);
Q = size(Psi, 2);
R = size(Psi, 3);

A_hat = zeros(N, Q);
cur_err = zeros(N, K+1);
full_err = zeros(N, 1);

lambda = 10^-4; % regularization weight for full model
w = 0.5 + 0.5*(1:MP)/MP; % windowing
w = ones(size(w));
W = diag(w.^2);

% reshape the matrices
Psi_ = zeros(MP*R, Q);
Z_ = zeros(MP*R, N);
for r = 1:R
    idx = (r-1)*MP + (1:MP);
    Psi_(idx, :) = Psi(:,:,r);
    Z_(idx, :) = Z(:,:,r);
end
Psi = Psi_;
Z = Z_;

% normalize
normPsi = sqrt(sum(Psi.^2, 1));
normPsi = ones(size(normPsi)); % no normalization
Psi = Psi ./ repmat(normPsi, size(Psi,1), 1);

% run orthogonal matching pursuit
for n = 1:N
    Z_n = Z(:,n);
    Psi_n = Psi;
        
    % complete LS solution & l2 residual error
    A_hat_tmp =  (Psi_n'  * W * Psi_n + lambda*eye(Q))^-1 * (Psi_n' * W * Z_n);
    rec_err = Z_n - Psi_n * A_hat_tmp;
    full_err(n) = sqrt(sum(rec_err.^2));

    % initialize
    i = 1;
    nz_idx = prior_nz(n,:)';

    A_hat_tmp = zeros(Q, 1);

    % compute residual signal
    res_sgnl = Z_n;
    cur_err(n,1) = sqrt(sum(res_sgnl.^2));

%     plot(res_sgnl);hold on;
    
    % OMP iteration
    while i<=K && cur_err(n,i) > (1+thres)*full_err(n)
        % find the next component
        prj = abs(Psi_n' * W * res_sgnl);
        
        [~, idx_max] = max(prj);
        
        nz_idx(idx_max) = true;
        
        % LS solution
        Psi_RLS =  Psi_n(:, nz_idx)'  * W * Psi_n(:, nz_idx) + eye(sum(nz_idx));

        Z_RLS = Psi_n(:, nz_idx)' * W * Z_n;

        A_hat_tmp(nz_idx) = Psi_RLS^-1 * Z_RLS;
%         A_hat_tmp'
        
        % perform sanity check
        if idx_max==1 || idx_max==10 || idx_max==11
            if A_hat_tmp(idx_max)<0
                nz_idx(idx_max) = false;
                A_hat_tmp(idx_max) = 0;
                Psi_n(:,idx_max) = 0.001*rand(MP,1);
                Psi_RLS =  Psi_n(:, nz_idx)' * W * Psi_n(:, nz_idx) + eye(sum(nz_idx));
                Z_RLS = Psi_n(:, nz_idx)' * W * Z_n;
                A_hat_tmp(nz_idx) = Psi_RLS^-1 * Z_RLS;
                i = i-1;
                thres = 1.5*thres;
%                 break;
            end
        else
            if A_hat_tmp(idx_max)>0
                nz_idx(idx_max) = false;
                A_hat_tmp(idx_max) = 0;
                Psi_n(:,idx_max) = 0.001*rand(MP,1);
                Psi_RLS =  Psi_n(:, nz_idx)' * W * Psi_n(:, nz_idx) + eye(sum(nz_idx));
                Z_RLS = Psi_n(:, nz_idx)' * W * Z_n;
                A_hat_tmp(nz_idx) = Psi_RLS^-1 * Z_RLS;
                i = i-1;
                thres = 1.5*thres;
%                 break;
            end
        end
        
        % compute residual signal
        res_sgnl = Z_n - Psi_n * A_hat_tmp;
        cur_err(n,i+1) = sqrt(sum(res_sgnl.^2));
%         plot(res_sgnl);
        
        i = i+1;
    end
%     hold off;
    
    % save the coefficients
    A_hat(n,:) = A_hat_tmp(1:Q)' ./ normPsi;
end

stats.cur_err = cur_err;
stats.full_err = full_err;

