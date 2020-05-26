close all;
clear all;
clc;


disp('Preparing training data');

folderCat = './Flipped Image Dataset/Training/Cat/';
folderDog = './Flipped Image Dataset/Training/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

feats = zeros(length(filesCat) + length(filesDog), 256);
labels = zeros(length(filesCat) + length(filesDog),1);

disp('Training Cat Data');
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    
    img = imread([folderCat filename]);
    img = imresize(img,[256,256]);
    
    feat = lbp(img);
    feats(i,:) = feat;
    
    labels(i) = 1;
end

disp('Training Dog Data');
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    
    img = imread([folderDog filename]);
    img = imresize(img,[256,256]);

    feat = lbp(img);
    feats(i + length(filesCat),:) = feat;

    labels(i + length(filesCat)) = 2;
end



disp('Preparing testing data');
folderCat = './Flipped Image Dataset/Testing/Cat/';
folderDog = './Flipped Image Dataset/Testing/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

featsTest = zeros(length(filesCat) + length(filesDog), 256);
groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
prediction = 0;

disp('Testing Cat Data');
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = imresize(img,[256,256]);

    feat = lbp(img);
    featsTest(i,:) = feat;
    
    groundtruthLabel(i) = 1;
end

disp('Testing Dog Data');
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = imresize(img,[256,256]);

    feat = lbp(img);
    featsTest(i + length(filesCat),:) = feat;
    
    groundtruthLabel(i + length(filesCat)) = 2;
end


disp('Performing testing');
accurateClassification = 0;

for i = 1:size(featsTest,1)
    feat = featsTest(i,:);
    dists = distChiSq(feat,feats);
    [val, idx] = sort(dists);
    
    cat_count = 0; 
    dog_count =0 ;
    k = 1;
    
    for n = 1 : k
        if(labels(idx(n)) == 1)
            cat_count = cat_count + 1;
        elseif(labels(idx(n)) == 2)
            dog_count = dog_count + 1;    
        end
    end
    
    if(cat_count > dog_count)
        predictedLabel(i) = 1;
    else
        predictedLabel(i) = 2;
    end
    
    if(predictedLabel(i) == groundtruthLabel(i))
        accurateClassification = accurateClassification + 1;
    end

end

accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);