function [ pa ] = imgpath( obj, idx )

pa = sprintf('%04d/%06d.jpg', obj.ind2sub(idx,1), obj.ind2sub(idx,2));

end

