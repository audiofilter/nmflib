% this script shows demonstrates some of the algorithms available in NMFlib
% using artificial data

% first make some data
n = 10;
m = 500;
r = 5;

W0 = normalize_W(rand(n,r),1);
H0 = rand(r,m);
V = W0*H0;

% this set of NMF variants don't require any extra parameters
algs1 = {@nmf_kl, @nmf_euc, @nmf_kl_loc, @nmf_convex};

W = cell(1,length(algs1));
H = cell(1,length(algs1));
E = cell(1,length(algs1));
figure;
for i = 1:length(algs1)
    tic
    [W{i},H{i},E{i}] = nmf_alg(V, r, 'alg', algs1{i}, 'verb', 2, 'niter', 500);
    toc
    
    subplot(length(algs1),1,i); plot(E{i}); 
    xlabel('Iterations'); ylabel('Error'); 
    title(functiontostring(algs1{i}), 'Interpreter', 'none');
end

% these variants implement sparsity in one way or another so we need to
% supply a weight for the penalty term
algs2 = {@nmf_kl_ns, @nmf_kl_sparse_es, @nmf_kl_sparse_v, @nmf_euc_sparse_es};
alpha = 0.1;

Ws = cell(1,length(algs2));
Hs = cell(1,length(algs2));
Es = cell(1,length(algs2));
figure;
for i = 1:length(algs2)
    tic
    [Ws{i},Hs{i},Es{i},C] = nmf_alg(V, r, 'alg', algs2{i}, 'verb', 2, ...
                                  'norm_w', 2, 'alpha', alpha, 'niter', 500);
    toc
    
    subplot(length(algs2),1,i); hold on; 
    plot(Es{i}); 
    xlabel('Iterations'); ylabel('Error'); 
    title([functiontostring(algs2{i}) ' (alpha=' num2str(alpha) ')'], 'Interpreter', 'none');
end

% now let's try the beta-divergence with different values of beta
beta = [0 0.5 1 2]; % corresponds to Itakura-Saito, I-divergence, Euclidean distances
figure;
for i = 1:length(beta)
    tic
    [w,h,e] = nmf_alg(V, r, 'alg', @nmf_beta, 'verb', 2, ...
                      'norm_w', 1, 'beta', beta(i), 'niter', 500);
    toc
    
    subplot(length(beta),1,i); plot(e); 
    xlabel('Iterations'); ylabel('Error'); 
    title(['nmf_beta (beta=' num2str(beta(i)) ')'], 'Interpreter', 'none');
end

% now let's try the Amari-divergence with different values of alpha
alpha = [-1 .9 0.5 1.01 2]; % 
figure;
for i = 1:length(alpha)
    tic
    [w,h,e] = nmf_alg(V, r, 'alg', @nmf_amari, 'verb', 2, ...
                      'norm_w', 1, 'alpha', alpha(i), 'niter', 500);
    toc
    
    subplot(length(alpha),1,i); plot(e); 
    xlabel('Iterations'); ylabel('Error'); 
    title(['nmf_amari (alpha=' num2str(alpha(i)) ')'], 'Interpreter', 'none');
end

