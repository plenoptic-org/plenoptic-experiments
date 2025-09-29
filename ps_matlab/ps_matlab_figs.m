addpath(genpath('../matlabPyrTools'))
addpath(genpath('../textureSynth'))
figs = {"fig12a", "fig12b", "fig12c", "fig12d", "fig12e", "fig12f", "fig13a", "fig13b", "fig13c", "fig13d", "fig14a", "fig14b", "fig14c", "fig14d", "fig14e", "fig15a", "fig15b", "fig15c", "fig15d", "fig15e", "fig15f", "fig16a", "fig16b", "fig16c", "fig16d", "fig16e", "fig16f", "fig18a", "fig3a", "fig3b", "fig4a", "fig4b", "fig5a", "fig5b", "fig6a", "fig6b", "fig8a", "fig8b"};

for i=1:length(figs)
    fig = figs{i}
    if fig == "fig18a"
        ext = ".png";
    else
        ext = ".jpg";
    end
    % load from plenoptic's store of images
    im0 = imread(strcat('/mnt/home/wbroderick/.cache/plenoptic/portilla_simoncelli_images.tar.gz.untar/portilla_simoncelli_images/', fig, ext));
    im0 = double(im0(:,:,1));

    Nsc = 4; % Number of scales
    Nor = 4; % Number of orientations
    Na = 9;  % Spatial neighborhood is Na x Na coefficients
	         % It must be an odd number!

    params = textureAnalysis(im0, Nsc, Nor, Na);

    Niter = 100;	% Number of iterations of synthesis loop
    Nsx = 256;	% Size of synthetic image is Nsy x Nsx
    Nsy = 256;	% WARNING: Both dimensions must be multiple of 2^(Nsc+2)

    res = textureSynthesis(params, [Nsy Nsx], Niter);

    imwrite(res/255, strcat('/mnt/ceph/users/wbroderick/ps_matlab/', fig, '.png'));
end
