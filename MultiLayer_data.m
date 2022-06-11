clc;
clear all;
load G;
load selectedNodes_pmmm;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    number of community
k = 10;%number of community
maxNodeCount=numel(G{1}(:,1));
n_Layers = numel(G);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   initial
H0 = abs(1+rand(n_Layers,k)/10);
Hmask = ones(n_Layers,k);
F0 = rand(maxNodeCount,k*2)/20;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   subsampling
for i=1:k
       a= s{i};
       for j=1:numel(a)
           F0(a(j),i)=1;
           F0(a(j),i+k)=1;
       end
    end
%**********************
for n=1:500
        Fmask = zeros(maxNodeCount,k*2); 
    Y=filterGraph_gAll1one2();
     Fmask(Y,:)=1;
     
     for i=1:k
         Fmask(s{i},:)=1/100;
     end
%***********************
    [F,H]=multilayerPGMTAanfpgd(G,F0,H0,Fmask,Hmask,0,1e-5);
    F0=F;
    H0=H;
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   result
figure; imagesc(F)
figure; [~,C]=max(F(:,1:k)+F(:,k+1:2*k),[],2); plot(C,'.'); box off;
%********************************* Determining community membership
 CommunityMatrix_in = F(:,1:k) ;
 CommunityMatrix_out= F(:,k+1:2*k);
 %******************* MAX
%  for i= 1:maxNodeCount 
%     [m j] = max(CommunityMatrix_in (i,:));
%     CommunityMatrix_in (i,k+1) = j;
%     [mm jj] = max(CommunityMatrix_out (i,:));
%     CommunityMatrix_out (i,k+1) = jj;
%  end
 %******************** BIGCLAM
 teta=sqrt(-log10(1-(1/3980)))
 %******************** DR Katan Teta MinMax
  for i= 1:maxNodeCount 
    [m_in j] = max(CommunityMatrix_in (i,:));
   m=min(m_in);
    logical_CommunityMatrix_in=CommunityMatrix_in>=m ;
    [m_out jj] = max(CommunityMatrix_out (i,:));
    mm=min(m_out);
logical_CommunityMatrix_out=CommunityMatrix_out>=mm ;
  end
for j=1:k
    v=logical_CommunityMatrix_in(:,j);
    sb_in{j}=find(v==1);
end
for j=1:k
    v=logical_CommunityMatrix_out(:,j);
    sb_out{j}=find(v==1);
end
    
 
 %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save F_subsample2 F;
save H_subsample2 H;
% save CommunityMatrix_in2 CommunityMatrix_in;
% save CommunityMatrix_out2 CommunityMatrix_out;
save logical_CommunityMatrix_in logical_CommunityMatrix_in;
save logical_CommunityMatrix_out logical_CommunityMatrix_out;
save sb_in sb_in;

save sb_out sb_out;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% computation of subsample
function [Y]=filterGraph_gAll1All2()
load G
u= randi([20 3970]);
L=numel(G);
for l=1:L
    A=G{l}(u,:);
     c{l}=find(A~=0);
end
Y=[];
for i=1:L
    K=c{i};
    for l=1:L
        for j=1:numel(K)
            A=G{l}(K(j),:);
            S=find(A~=0);
            Y=union(S,Y);
        end
    end
end
end 

function [Y]=filterGraph_gAll1one2()
load G
u= randi([20 3970]);
L=numel(G);
L2=randi([1 L]);
for l=1:L
    A=G{l}(u,:);
     c{l}=find(A~=0);
end
all_N1=c{1}; 
for i=2:L
    all_N1=union(all_N1,c{i});
end

Y=[];

    K=c{L2};
   
        for j=1:numel(K)
            A=G{L2}(K(j),:);
            S=find(A~=0);
            Y=union(S,Y);
        end
  all_N1_N2=union(all_N1,Y);
  Y= all_N1_N2;

end









