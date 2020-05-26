close all;
clear all;
clc;

%Add MatConvNet
run './matconvnet/matlab/vl_setupnn'

%Load the pre-trained CNN
net = load('imagenet-caffe-alex.mat');

% disp('Preparing Training data');
% folderCat = './Updated_DogCat/Training/Cat/';
% folderDog = './Updated_DogCat/Training/Dog/';
%
% filesCat = dir(fullfile(folderCat, '*.jpg'));
% filesDog = dir(fullfile(folderDog, '*.jpg'));
%
% feats = zeros(length(filesCat) + length(filesDog), 4096);
% labels = zeros(length(filesCat) + length(filesDog), 1);
%
% %for cat - training
% for i = 1:length(filesCat)
%     disp(i);
%     filename = filesCat(i,1).name;
%     im = imread([folderCat filename]);
%     im_ = single(im) ;
%     im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;
%
%     for j = 1 : 3
%         im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
%     end
%
%     % run the CNN
%     res = vl_simplenn(net, im_) ;
%     feat = res(18).x;
%     feat = feat(:)';
%     feats(i,:) = feat;
%     labels(i) = 1;
% end
%
% % for dog - training
% for i = 1:length(filesDog)
%     disp(i);
%     filename = filesDog(i,1).name;
%     img = imread([folderDog filename]);
%     im = imread([folderCat filename]);
%     im_ = single(im) ;
%     im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;
%
%     for j = 1 : 3
%         im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
%     end
%
%     % run the CNN
%     res = vl_simplenn(net, im_) ;
%
%     feat = res(18).x;
%     feat = feat(:)';
%     feats(i,:) = feat;
%
%     feats(i + length(filesCat),:) = feat;
%     labels(i + length(filesCat)) = 2;
% end



% Preparing testing data
disp('Preparing testing data');

folderCat = './Datasets/Testing/Cat/';
folderDog = './Datasets/Testing/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

feats = zeros(length(filesCat) + length(filesDog), 4096);
labels = zeros(length(filesCat) + length(filesDog), 1);

groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

for i = 1 : length(filesCat)
    labels (i) =1;
end
%Testing on Cat
for i = 1 : length(filesCat)
    disp(i);
    filename = filesCat(i, 1).name;
    im = imread([folderCat filename]);
    im_ = single(im) ;
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;
    
    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    feat = res(18).x;
    feat = feat(:)';
    feats(i,:) = feat;
    
    groundtruthLabel(i) = 1;
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure ; clf; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
        net.meta.classes.description{best}, best, bestScore)) ;
    %figure, bar(featurevector);
end


%Testing on Dog
for i = 1 : length(filesDog)
    disp(i);
    filename = filesDog(i, 1).name;
    im = imread([folderDog filename]);
    im_ = single(im) ;
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;
    
    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    feat = res(18).x;
    feat = feat(:)';
    labels(i +length(filesCat)) = 2;
    feats(i + length(filesCat), :) = feat;
    groundtruthLabel(i + length(filesCat)) = 2;
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure ; clf; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
        net.meta.classes.description{best}, best, bestScore)) ;
    %    figure, bar(featurevector);
end

disp('Performing Testing');
accurateClassification = 0;

for i = 1 : size(feats,1)
    feat = feats(i, :);
    dists = distChiSq(feat, feats);
    [val, idx] = sort(dists);
    
    prediction = 0;
    k = 9;
    
    for n = 1 : k
        if(groundtruthLabel(i) == labels(idx(n)))
            prediction = prediction + 1;
        end
    end
    
    prediction = prediction / k;
    
    if(prediction > 0.5)
        accurateClassification = accurateClassification + 1;
    end
end

accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);