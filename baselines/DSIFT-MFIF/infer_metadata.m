% reference only, not used at runtime
function infer_metadata(varargin)
% Batch metadata.json entry point; keeps the author's DSIFT code untouched.
p=inputParser; addParameter(p,'metadata',''); addParameter(p,'output_dir','');
addParameter(p,'start_index',0); addParameter(p,'max_samples',-1); addParameter(p,'overwrite',0);
addParameter(p,'save_inputs',0); addParameter(p,'size_policy','error'); parse(p,varargin{:}); a=p.Results;
if isempty(a.metadata)||isempty(a.output_dir), error('metadata and output_dir are required'); end
if ~exist(a.output_dir,'dir'), mkdir(a.output_dir); end
metaPath=char(java.io.File(a.metadata).getCanonicalPath()); base=fileparts(metaPath);
items=jsondecode(fileread(metaPath)); if ~isstruct(items), error('metadata top level must be a list of objects'); end
n=numel(items); first=a.start_index+1; last=n; if a.max_samples>=0, last=min(n,first+a.max_samples-1); end
records=repmat(empty_record(),0,1); errfid=fopen(fullfile(a.output_dir,'errors.jsonl'),'w');
for k=first:last
    t=tic; item=items(k); rec=empty_record(); rec.index=k-1;
    try
        if ~isfield(item,'edit_image')||numel(item.edit_image)<2, error('edit_image must contain at least two paths'); end
        % Deliberately read only entries 1 and 2. focus_a/focus_b are ignored.
        pa=resolve_path(item.edit_image{1},base); pb=resolve_path(item.edit_image{2},base);
        rec.source_a=pa; rec.source_b=pb; if isfield(item,'image'), rec.gt=resolve_path(item.image,base); end
        rec.sample_id=make_id(item,k-1); pred=fullfile(a.output_dir,[rec.sample_id '_pred.png']); rec.prediction=pred;
        if exist(pred,'file')&&~logical(a.overwrite), rec.success=true; rec.error='skipped_existing'; records(end+1)=rec; continue; end
        if ~exist(pa,'file'), error('A image not found: %s',pa); end; if ~exist(pb,'file'), error('B image not found: %s',pb); end
        A=to_rgb(imread(pa)); B=to_rgb(imread(pb)); rec.original_width=size(A,2); rec.original_height=size(A,1); originalA=A;
        if any([size(A,1) size(A,2)]~=[size(B,1) size(B,2)])
            switch a.size_policy
                case 'error', error('A/B size mismatch');
                case 'resize_b_to_a', B=imresize(B,[size(A,1) size(A,2)],'bicubic');
                case 'center_crop_common'
                    h=min(size(A,1),size(B,1)); w=min(size(A,2),size(B,2)); A=center_crop(A,h,w); B=center_crop(B,h,w);
                otherwise, error('unknown size_policy');
            end
        end
        F=DSIFT_Fusion(A,B,48,8,1);
        if size(F,1)~=rec.original_height||size(F,2)~=rec.original_width
            canvas=originalA; y=floor((size(canvas,1)-size(F,1))/2)+1; x=floor((size(canvas,2)-size(F,2))/2)+1;
            canvas(y:y+size(F,1)-1,x:x+size(F,2)-1,:)=F; F=canvas;
        end
        imwrite(F,pred,'png'); if logical(a.save_inputs), imwrite(A,fullfile(a.output_dir,[rec.sample_id '_input_a.png'])); imwrite(B,fullfile(a.output_dir,[rec.sample_id '_input_b.png'])); end
        rec.success=true;
    catch ME
        rec.error=[class(ME) ': ' ME.message]; fprintf(errfid,'%s\n',jsonencode(rec));
    end
    rec.runtime_seconds=toc(t); records(end+1)=rec;
end
fclose(errfid); fid=fopen(fullfile(a.output_dir,'inference_manifest.json'),'w'); fprintf(fid,'%s',jsonencode(records,'PrettyPrint',true)); fclose(fid);
writetable(struct2table(records),fullfile(a.output_dir,'inference_manifest.csv'));
cfg=a; cfg.metadata=metaPath; cfg.method='DSIFT-MFIF'; cfg.checkpoint_loaded=[]; fid=fopen(fullfile(a.output_dir,'run_config.json'),'w'); fprintf(fid,'%s',jsonencode(cfg,'PrettyPrint',true)); fclose(fid);
end

function r=empty_record(), r=struct('index',0,'sample_id','','source_a','','source_b','','gt','','prediction','','original_width',[],'original_height',[],'runtime_seconds',0,'success',false,'error',''); end
function p=resolve_path(raw,base)
raw=char(raw); raw=strrep(raw,'\',filesep); raw=strrep(raw,'/',filesep);
if ispc, absolute=~isempty(regexp(raw,'^[A-Za-z]:[\\/]','once'))||startsWith(raw,'\\'); else, absolute=startsWith(raw,filesep); end
if absolute, p=raw; else, p=fullfile(base,raw); end
end
function x=to_rgb(x), if ndims(x)==2, x=repmat(x,[1 1 3]); elseif size(x,3)>=4, x=x(:,:,1:3); end; if ~isa(x,'uint8'), x=im2uint8(x); end; end
function x=center_crop(x,h,w), y=floor((size(x,1)-h)/2)+1; z=floor((size(x,2)-w)/2)+1; x=x(y:y+h-1,z:z+w-1,:); end
function id=make_id(item,index)
if isfield(item,'source_index')&&~isempty(item.source_index), raw=num2str(item.source_index); elseif isfield(item,'image')&&~isempty(item.image), [~,raw]=fileparts(char(item.image)); elseif isfield(item,'edit_image'), [~,raw]=fileparts(char(item.edit_image{1})); else, raw=num2str(index); end
suffixes={'_target','_gt','_a','_src','_source','_input'}; low=lower(raw); for i=1:numel(suffixes), s=suffixes{i}; if endsWith(low,s), raw=extractBefore(raw,strlength(raw)-strlength(s)+1); break; end; end
raw=regexprep(char(raw),'[^0-9A-Za-z._-]+','_'); if all(isstrprop(raw,'digit')), id=sprintf('%06d',str2double(raw)); else, id=raw; end
end
