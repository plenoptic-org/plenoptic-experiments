function export_matlab_ps_stats(overwrite)
% Export one MATLAB statistics reference for each case in sweep.json.
% Run from the experiment directory with prepared TIFFs in inputs/ and the
% MATLAB texture and pyramid toolboxes in matlab_ps_code/.
%
% overwrite - Optional logical flag. By default, reuse a cached reference
%   if its image and parameters match and raise an error on mismatch.
%   Pass true to regenerate every reference.

if nargin < 1
    overwrite = false;
end
addpath(genpath('matlab_ps_code'));
sweep = jsondecode(fileread('sweep.json'));
output_dir = fullfile('validation', 'matlab_statistics');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

for image = string(sweep.images(:)).'
    image_path = fullfile('inputs', image + ".tif");
    im0 = imread(image_path);
    for Nsc = sweep.n_scales(:).'
        for Nor = sweep.n_orientations(:).'
            for Na = sweep.spatial_corr_width(:).'
                case_id = sprintf('%s-scales%d-ori%d-spat%d', image, Nsc, Nor, Na);
                statistics_path = fullfile(output_dir, [case_id '.mat']);
                export_case(statistics_path, case_id, image_path, im0, Nsc, Nor, Na, overwrite);
            end
        end
    end
end
end

function export_case(statistics_path, case_id, image_path, im0, Nsc, Nor, Na, overwrite)
if isfile(statistics_path) && ~overwrite
    cached = load(statistics_path, 'im0', 'Nsc', 'Nor', 'Na');
    assert(isequal(cached.im0, im0) && cached.Nsc == Nsc && ...
           cached.Nor == Nor && cached.Na == Na, ...
           'Cached image/parameters differ for %s; rerun with overwrite=true.', ...
           case_id);
    fprintf('Reusing %s\n', statistics_path);
    return;
end

params = textureColorAnalysis(im0, Nsc, Nor, Na);

% Export the same PCA transform used inside textureColorAnalysis.
[Ny, Nx, Nclr] = size(im0);
rgb = reshape(double(im0), Ny*Nx, Nclr);
pcaMean = mean(rgb).';
rgbCentered = rgb - ones(Ny*Nx, Nclr) * diag(pcaMean);
Cclr0 = innerProd(rgbCentered) / (Ny*Nx);
[V, D] = eig(Cclr0);
pcaMatrix = pinv(sqrt(D)) * V.';
imPCA = reshape(rgbCentered * V * pinv(sqrt(D)), Ny, Nx, Nclr);

matlab_version = version;
save(statistics_path, 'im0', 'params', 'Nsc', 'Nor', 'Na', ...
     'pcaMean', 'Cclr0', 'V', 'D', 'pcaMatrix', 'imPCA', ...
     'matlab_version', 'image_path', 'case_id', '-v7');
fprintf('Saved %s\n', statistics_path);
end
