function [D] = squareDist(X, Z)

D = slmetric_pw(X', Z', 'sqdist');  %matrix of distances between the points

end