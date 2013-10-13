path_to_maxent = '/your/path/to/maxent'
addpath( genpath( path_to_maxent ));

data_dir = '/path/to/your/digits/data/dir/'

% you'll need to add headers to this file
output_file = strcat( data_dir, 'cosplay_p.csv' )

% headers in both train and test
% no labels in test
train_data = csvread( strcat( data_dir, 'train.csv' ), 1, 0 );	
testx = csvread( strcat( data_dir, 'test.csv' ), 1, 0 );	

%%%
 
trainx = train_data(:, 2:end);
train_y = train_data(:, 1);

% one hot

yt = zeros( size( train_y, 1 ), 10 );
for i = 1:10
	digit = i - 1;
	j = train_y == digit;
	yt( j, i ) = 1;
end

% 
 
tic
fprintf('computing random feature map');
 
% (uncentered) pca to 50 ... makes subsequent operations faster,
% but also makes the random projection more efficient by focusing on
% where the data is
 
opts.isreal = true; 
[v,~]=eigs(double(trainx'*trainx),50,'LM',opts);
trainx=trainx*v;
testx=testx*v; 
clear v opts;
 
% estimate kernel bandwidth using the "median trick"
% this is a standard Gaussian kernel technique
 
[n,k]=size(yt);
[m,p]=size(testx);
sz=3000;
perm=randperm(n);
sample=trainx(perm(1:sz),:);
norms=sum(sample.^2,2);
dist=norms*ones(1,sz)+ones(sz,1)*norms'-2*sample*sample';
scale=1/sqrt(median(dist(:)));
 
clear sz perm sample norms dist;
 
% here is the actual feature map:
% Gaussian random matrix, uniform phase, and cosine
 
d=4000;
r=randn(p,d);
b=2.0*pi*rand(1,d);
trainx=cos(bsxfun(@plus,scale*trainx*r,b));
testx=cos(bsxfun(@plus,scale*testx*r,b));
 
fprintf(' finished: ');
toc
 
 
 
 
tic
fprintf('starting logistic regression (this takes a while)\n');
 
% get @maxent and lbfgs.m from http://www.cs.grinnell.edu/~weinman/code/
% if you get an error about randint being undefined, change it to randi
 
%addpath recognition;
%addpath opt;
%addpath local;
 
C0=maxent(k,d);
[~,trainy]=max(yt');
options.MaxIter=300; 
options.Display='off';
C1=train(C0,trainy,trainx,'gauss',4.2813,[],[],[],options);
% regularizer was chosen by cross-validation as follows
%perm=randperm(n);
%it=logical(zeros(1,n));
%it(perm(1:int32(0.8*n)))=1;
%[C1,V]=cvtrain(C0,trainy(perm),trainx(perm,:),'gauss',10.^linspace(-4,4,20), ...
%               [],0,[],it,[],@accuracy);
         
fprintf('finished: ');
toc
fprintf('train accuracy is %g\n',accuracy(C1,trainy,trainx));

P = map(C1,testx);
P = P - 1;
ids = 1:28000;
p = [ ids' P ];
csvwrite( output_file, p )
