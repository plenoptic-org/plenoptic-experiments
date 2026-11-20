% Synthesize a baseline color PS metamer for each image in sweep.json.

% Run from ps_color_matlab_vs_plen; see README for dependency setup.
addpath(genpath('matlab_ps_code'));

sweep = jsondecode(fileread('sweep.json'));
metamer_dir = 'matlab_metamers';

if ~exist(metamer_dir, 'dir')
    mkdir(metamer_dir);
end

set(groot, 'DefaultFigureVisible', 'off');

Nsc = 4;
Nor = 4;
Na = 7;
Niter = 50;
seed = 0;

for index = 1:numel(sweep.images)
    image_name = sweep.images{index};
    image_path = fullfile('inputs', [image_name '.tif']);
    im0 = imread(image_path);

    % Compute the statistics
    params = textureColorAnalysis(im0, Nsc, Nor, Na);

    [Ny, Nx, ~] = size(im0);

    % The fifth mask entry disables convergence figures, which is useful for a
    % non-interactive cluster run and avoids legacy figure-handle arithmetic.
    cmask = [1; 1; 1; 1; 0];
    matlab_metamer = textureColorSynthesis(params, [Ny Nx seed], Niter, cmask);

    metamer_path = fullfile(metamer_dir, ['matlab_' image_name '.tif']);
    imwrite(matlab_metamer, metamer_path);
    save(fullfile(metamer_dir, ['matlab_' image_name '.mat']), ...
         'image_name', 'image_path', 'Nsc', 'Nor', 'Na', 'Niter', 'seed', ...
         'matlab_metamer', 'cmask', '-v7');

    close all;
    fprintf('Saved metamer to %s\n', metamer_path);
end
