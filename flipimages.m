close all;
clear all;
clc;
folder = './DogCat/Training/Cat/';
files = dir(fullfile(folder,'*.jpg'));

for i = 1:length(files)
    disp(i);
    filename = files(i,1).name;
    img= imread([folder filename]);
    IR = flip(img,2);
    fname = sprintf('catf%d.jpg',i);
    imwrite(IR,fname);
end
%im = imread('dog.jpg');
% Ir = flip(im,2);

% subplot(1,2,1); imshow(im); title('Query');
% subplot(1,2,2); imshow(Ir); title('Result');

%IR = flip(im,2);
%imwrite(IR,'yy.jpg');

% for m = 1:10
%    % ...
%    fname = sprintf('name%d.jpg',m);
%    imwrite(IR,fname);
%    % ...
% end
