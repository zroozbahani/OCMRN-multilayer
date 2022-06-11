function [F,H,rnllk]=multilayerPGMTAanfpgd(G,F,H,Fmask,Hmask,lambdaF,lambdaH)
% [F,H,rnllk]=multilayerPGMTAanfpgd(G,F0,H0,Fmask,Hmask,lambdaH,lambdaF)
% model:  'anf': Asymmetrized Non-negative factorization
% optimization:   Projected Gradient Descent 
%
% G: adjacency matrix of multilayer network (a cell array of size L, each cell is NxN sparse/full, binary)
%
% F: node-community membership strength, 
%    Nx2K
% H: community-layer strength,
%    LxK
%
% rnllk: the regularized negative log likelihood of the multilayerPGMTAanf
%
% N: number of nodes
% L: number of layers
% K: number of communities
%
% lambdaF: regularization coefficient of F, higher value more symmetry. 
% lambdaH: regularization coefficient of H, higher value more sparse H.
%

%% set the optimization parameters to your desired values, here
eta =3;         % base learning rate. Might set to smaller value if your network is huge. gd is totaly novice to adjust learning rate adaptively, so it is you that should set proper value for it.
rel_tol = 1e-5;  % relative tolerance for rnllk convergance. 1e-5 seems to bu quite good choice.
max_itr = 20;  % max iteration
every_show_once = 10; % every 10 iteration show optimization info once. Set to your desired value. Set 0 to turn it off.


%% Argument Default Settings
L = length(G);
for l=1:L,
  [N,M]=size(G{l});
  if N~=M, fprintf('All layers must be NxN\n'); return, end
  G{l}=logical(G{l});
  for u=1:N,
    G{l}(u,u)=false;  % Convention: (u,u) is not in E
  end
end

[M,KK]=size(F);
if N~=M || mod(KK,2)~=0, fprintf('F0 must be %dx(2*K)\n',N); return, end
K=KK/2;
[Lh,Kh]=size(H);
if Lh~=L || Kh~=K, fprintf('H0 must be %dx%d\n',L,K); return, end
[Nfm,KKfm]=size(Fmask);
if Nfm~=N || KKfm~=KK, fprintf('Fmask must be %dx%d\n',N,KK); return, end
[Lh,Kh]=size(Hmask);
if Lh~=L || Kh~=K, fprintf('Hmask must be %dx%d\n',L,K); return, end

%% Initialize local variables
uinc=cell(N,L);
vinc=cell(N,L);
for l=1:L,
  [U{l},V{l}]=find(G{l});
  for i=1:N,
    uinc{i,l}=find(U{l}==i);
    vinc{i,l}=find(V{l}==i);
  end
end

ns = N*(N-1)*L;
lambdaH = ns*lambdaH;
lambdaF = ns*lambdaF;
inveta = ns/eta;
x = [F(:);H(:)];
%%%% Begin of optimization %%%%
%% You can replace the following block with fmincon if you have the optimization toolbox.
old_rnllk = Inf;
stale = 0;
itr=0;
while 1,
  [rnllk, g] = PGMTA_anf_rnllk(x);
  
  %% Controls for convergance and divergance criteria
  if rnllk < old_rnllk,
    if (old_rnllk-rnllk)/(rnllk+1) < rel_tol,
      if stale==20, break, end
      stale = stale + 1;
    else
      stale = 0;
    end
  else
    if stale==5, break, end
    stale = stale + 1;
  end
  old_rnllk = rnllk;
  itr = itr+1;
  if itr>=max_itr, break, end
  
  %% Show some info during optimization
  if itr==1 || mod(itr,every_show_once)==0,
    fprintf('itr= %d, rnllk= %g, abs(gF)= %g, abs(gH)= %g, sp(F)= %.1f, sp(H)= %.1f\n',itr,rnllk,mean(abs(g(1:N*KK))),mean(abs(g(N*KK+1:N*KK+L*K))),100-100*nnz(x(1:N*KK))/(N*KK),100-100*nnz(x(N*KK+1:N*KK+L*K))/(L*K));
  end
  
  %%%% Projected Gradient Descent is here! %%%
  %% adjust learning rate to avoid bulk vanishing
  eeta = 1/max(0.9/max(g(x>0.01)./x(x>0.01)),inveta);
  %% the gradient descent
  x = x - eeta * g;
  %% the projection to the non-negative subspace (x>=0)
  x = max(x,0);
end

fprintf('itr= %d, rnllk= %g, abs(gF)= %g, abs(gH)= %g, sp(F)= %.1f, sp(H)= %.1f\n',itr,rnllk,mean(abs(g(1:N*KK))),mean(abs(g(N*KK+1:N*KK+L*K))),100-100*nnz(x(1:N*KK))/(N*KK),100-100*nnz(x(N*KK+1:N*KK+L*K))/(L*K));
%%%% End of optimization %%%%

F = reshape(x(1:N*KK),N,KK);
H = reshape(x(N*KK+1:N*KK+L*K),L,K);


%%%%%%%%%%% local functions %%%%%%%%%%% 
    function [rnllk,g]=PGMTA_anf_rnllk(x)
        Fout = reshape(x(1:N*K),N,K);
        Fin = reshape(x(N*K+1:N*KK),N,K);
        H = reshape(x(N*KK+1:N*KK+L*K),L,K);
        Delta = Fout-Fin;
        sumHc = sum(H,1); Fisum = sum(Fin,1); Fosum = sum(Fout,1);
        rnllk = dot(sumHc.*Fisum,Fosum,2)-sum(dot(sumHc.*Fin,Fout,2),1);
        gFout = sumHc .* (Fisum - Fin);
        gFin = sumHc .* (Fosum - Fout);
        gHg = Fosum.*Fisum - dot(Fout,Fin,1);
        for l=1:L,
          T = H(l,:).*Fin(V{l},:);
          [ArcLOD, gArcLOD] = isoftplus( dot( T, Fout(U{l},:), 2 ) );
          rnllk = rnllk - sum(ArcLOD);
          gArcLOD = gArcLOD';
          for u=1:N,
            arcsuhead = uinc{u,l};
            gFout(u,:) = gFout(u,:) - gArcLOD(arcsuhead) * T(arcsuhead,:);
          end
          T = H(l,:).*Fout(U{l},:);
          for v=1:N,
            arcsvtail = vinc{v,l};
            gFin(v,:) = gFin(v,:) - gArcLOD(arcsvtail) * T(arcsvtail,:);
          end
          gH(l,:) = gHg - gArcLOD * (Fout(U{l},:).*Fin(V{l},:));
        end
        sumHl = sum(H,2);
        rnllk = rnllk + lambdaH/2 * ( sum(sumHc.^2)/L - sum(sumHl.^2)/K ) + lambdaF/2 * sum(Delta(:).^2);
        gH = gH + lambdaH * ( sumHc/L - sumHl/K );
        gFout = gFout + lambdaF * Delta;
        gFin = gFin - lambdaF * Delta;

        gFout = Fmask(:,1:K) .* gFout;
        gFin = Fmask(:,K+1:KK) .* gFin;
        gH = Hmask .* gH;
        g = [gFout(:); gFin(:); gH(:)];
    end

    function [y,g]=isoftplus(x)
      % A numerically stable implementation for inverse of softplus and its gradient.
      % By definition, softplus(x)=log(exp(x)+1), and isoftplus(x)=log(exp(x)-1).
      g = 1-exp(-x);
      y = x+log(max(g,0));
      g = 1./g;
    end
    
end
