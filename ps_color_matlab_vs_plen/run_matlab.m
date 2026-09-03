% Compute the color statistics of an image, and export them together with
% useful data like the PCA matrix and the PCA transformed image.
% Synthesize a metamer with the original Matlab procedure

image_name = 'DSCF4315';
image_path = fullfile('inputs', [image_name '.tif']);
statistics_dir = 'matlab_statistics';
metamer_dir = 'matlab_metamers';

if ~exist(statistics_dir, 'dir')
    mkdir(statistics_dir);
end
if ~exist(metamer_dir, 'dir')
    mkdir(metamer_dir);
end

set(groot, 'DefaultFigureVisible', 'off');

im0 = imread(image_path);

Nsc = 4;
Nor = 4;
Na = 7;
Niter = 50;
seed = 0;

% Compute the statistics
params = textureColorAnalysis(im0, Nsc, Nor, Na);

% Export the exact MATLAB PCA transform and intermediate PCA image so Python
% can use the same basis, ordering, signs, and 0--255 input values.
% Neither the PCA nor the transformed image are saved by the Matlab code,
% so we recompute them here with the same operations.
[Ny, Nx, Nclr] = size(im0);
rgb = reshape(double(im0), Ny*Nx, Nclr);
pcaMean = mean(rgb).';
rgbCentered = rgb - ones(Ny*Nx, Nclr) * diag(pcaMean);
Cclr0 = innerProd(rgbCentered) / (Ny*Nx);
[V, D] = eig(Cclr0);
pcaMatrix = pinv(sqrt(D)) * V.';
imPCA = reshape(rgbCentered * V * pinv(sqrt(D)), Ny, Nx, Nclr);

statistics_path = fullfile(...
    statistics_dir, ['matlab_' image_name '.mat']);
save(statistics_path, 'im0', 'params', 'Nsc', 'Nor', 'Na', 'Niter', ...
     'seed', 'pcaMean', 'Cclr0', 'V', 'D', 'pcaMatrix', 'imPCA', '-v7');

% The fifth mask entry disables convergence figures, which is useful for a
% non-interactive cluster run and avoids legacy figure-handle arithmetic.
cmask = [1; 1; 1; 1; 0];
matlab_metamer = textureColorSynthesis(params, [Ny Nx seed], Niter, cmask);

metamer_path = fullfile(metamer_dir, ['matlab_' image_name '.tif']);
imwrite(matlab_metamer, metamer_path);
save(statistics_path, 'matlab_metamer', 'cmask', '-append');

close all;
fprintf('Saved statistics to %s\n', statistics_path);
fprintf('Saved metamer to %s\n', metamer_path);
