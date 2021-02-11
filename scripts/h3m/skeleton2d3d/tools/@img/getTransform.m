function [ T ] = getTransform( obj, center, scale, rot, res )

h = 200 * scale;
T = eye(3);

% Scaling
T(1,1) = res / h;
T(2,2) = res / h;

% Translation
T(1,3) = res * (-center(1) / h + 0.5);
T(2,3) = res * (-center(2) / h + 0.5);

% Rotation
% TODO: finish implementation
assert(rot == 0);

end