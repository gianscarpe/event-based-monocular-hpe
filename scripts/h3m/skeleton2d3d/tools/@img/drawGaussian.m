function [ img ] = drawGaussian( obj, img, pt, sigma )
% Draw a 2D gaussian

w = size(img,2);
h = size(img,1);

% Check that any part of the gaussian is in-bounds
ul = [floor(pt(1)-3*sigma), floor(pt(2)-3*sigma)];
br = [floor(pt(1)+3*sigma), floor(pt(2)+3*sigma)];
% If not, return the image as is
if ul(1) > w || ul(2) > h || br(1) < 1 || br(2) < 1
    return
end
% Generate gaussian
% 1. the maximum valus is normalized to 1
% 2. actual sigma is size/4
% verified by comparing image.gaussian(13) and matlab command: 
%   h = fspecial('gaussian', 13, 13/4); h/max(max(h))
sz = 6 * sigma + 1;
g = fspecial('gaussian', sz, sz/4);
g = g/max(max(g));
% Usable gaussian range
g_x = [max(1,-ul(1)), min(br(1),w)-max(1,ul(1))+max(1,-ul(1))];
g_y = [max(1,-ul(2)), min(br(2),h)-max(1,ul(2))+max(1,-ul(2))];
% Image range
img_x = [max(1,ul(1)), min(br(1),w)];
img_y = [max(1,ul(2)), min(br(2),h)];
assert(g_x(1) > 0 && g_y(1) > 0);
img(img_y(1):img_y(2),img_x(1):img_x(2)) = ...
    img(img_y(1):img_y(2),img_x(1):img_x(2)) + g(g_y(1):g_y(2),g_x(1):g_x(2));
img(img > 1) = 1;

end

