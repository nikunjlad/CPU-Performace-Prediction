fprintf('Loading dataset...\n\n');
load('comp.mat');
X = comp(:,1:7);
y = comp(:,8);
[X, mu, sigma] = normalize(X);

figure; scatter(X(:,1),y);
figure; scatter(X(:,2),y);
figure; scatter(X(:,3),y);
figure; scatter(X(:,4),y);
figure; scatter(X(:,5),y);
figure; scatter(X(:,6),y);
figure; scatter(X(:,7),y);
