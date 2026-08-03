function smoke_test_metadata
root=tempname; mkdir(root); mkdir(fullfile(root,'images')); mkdir(fullfile(root,'out'));
imwrite(uint8(zeros(16,20,3)),fullfile(root,'images','a.png')); imwrite(uint8(ones(16,20,3)*255),fullfile(root,'images','b.png'));
items(1)=struct('source_index',1,'edit_image',{{'images/a.png','images/b.png'}});
items(2)=struct('source_index',2,'edit_image',{{'images\a.png','images\b.png','images/DO_NOT_READ_1.png','images/DO_NOT_READ_2.png'}});
items(3)=struct('source_index',3,'edit_image',{{'images/missing.png','images/b.png'}});
fid=fopen(fullfile(root,'metadata.json'),'w'); fprintf(fid,'%s',jsonencode(items)); fclose(fid);
infer_metadata('metadata',fullfile(root,'metadata.json'),'output_dir',fullfile(root,'out'));
m=jsondecode(fileread(fullfile(root,'out','inference_manifest.json')));
assert(m(1).success&&m(2).success&&~m(3).success); assert(all(size(imread(m(1).prediction),[1 2])==[16 20]));
disp('DSIFT metadata smoke: PASS');
end
