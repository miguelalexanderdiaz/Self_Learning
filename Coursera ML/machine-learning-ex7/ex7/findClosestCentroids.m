function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
warning ("off", "Octave:broadcast")
diff=zeros(size(X,1),K);
for i=1:K
  %calculate the euclidean distance between the centroids and every example x_i
  diff(:,i)=norm(X-centroids(i,:),2,'rows');
endfor


%find the minimum of the distances between x_i and the centroids
min=min(diff,[],2);

%fill the idx vector with the position where the minimum whas find (don't know how to vectorize this)
for i=1:size(X,1)
  idx(i)=find(diff(i,:)==min(i),1);
endfor

%display(diff(1:3,:))

% =============================================================

end

