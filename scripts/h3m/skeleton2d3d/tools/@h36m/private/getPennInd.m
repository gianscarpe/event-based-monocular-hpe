function [ ind ] = getPennInd( obj, dim )
% Get Penn Action's joint indices

joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
ind = [];
for i = joints
    for j = 1:dim
        ind = [ind; (i-1)*dim+j];  %#ok
    end
end

end

