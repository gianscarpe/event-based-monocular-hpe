function [ cl ] = colorHM( obj, x )
% Converts a one-channel grayscale image to a color heatmap image

cl = zeros(3, size(x,1), size(x,2));
cl(1,:,:) = gauss(x,0.5,0.6,0.2) + gauss(x,1.0,0.8,0.3);
cl(2,:,:) = gauss(x,1.0,0.5,0.3);
cl(3,:,:) = gauss(x,1.0,0.2,0.3);

end

function g = gauss(x,a,b,c)
g = a * exp(-((x - b).^2)/(2*c*c));
end

