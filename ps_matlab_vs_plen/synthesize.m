function synthesize(figName, Niter, initImgName)
    addpath(genpath('../matlabPyrTools'));
    addpath(genpath('../textureSynth'));

    tic;

    initImg = imread(strcat('/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_speed/init_images/', initImgName, '.png'));
    initImg = double(initImg(:,:,1));

    if figName == "fig18a"
        ext = ".png";
    else
        ext = ".jpg";
    end

    im0 = imread(strcat('/mnt/home/wbroderick/.cache/plenoptic/portilla_simoncelli_images.tar.gz.untar/portilla_simoncelli_images/', figName, ext));
    im0 = double(im0(:,:,1));

    params = textureAnalysis(im0, 4, 4, 9);

    initToc = toc;

    res = textureSynthesis(params, initImg, Niter);

    synthToc = toc;

    save(strcat('/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_speed/matlab_results/', figName, "_", initImgName, "_iter-", string(Niter), '.mat'), "res", "initToc", "synthToc")
end
