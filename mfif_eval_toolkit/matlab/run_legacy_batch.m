% reference only, not used at runtime
function run_legacy_batch(job_csv, out_csv, tpami_root, objective_root, metrics_csv)
%RUN_LEGACY_BATCH Evaluate published no-GT image-fusion metrics in batch.
%
% The path order intentionally prefers the TPAMI MFIF-Metrics functions
% when duplicate names exist, while Objective-evaluation provides QSF,
% QAB/F, alternate Qabf and source-MS-SSIM utilities.

    if ~isfolder(fullfile(tpami_root, 'fusion-metrics'))
        error('TPAMI fusion-metrics directory missing: %s', tpami_root);
    end
    if ~isfolder(objective_root)
        error('Objective-evaluation directory missing: %s', objective_root);
    end

    addpath(genpath(objective_root), '-end');
    addpath(genpath(fullfile(tpami_root, 'fusion-metrics')), '-begin');

    jobs = readtable(job_csv, 'TextType', 'string');
    metrics = split(string(metrics_csv), ',');
    metrics = metrics(strlength(metrics) > 0);
    n = height(jobs);
    result = table(jobs.row_id, 'VariableNames', {'row_id'});
    for k = 1:numel(metrics)
        result.(char(metrics(k))) = nan(n, 1);
    end
    legacy_error = strings(n, 1);

    for i = 1:n
        try
            A = read_gray_double(jobs.source_a(i));
            B = read_gray_double(jobs.source_b(i));
            F = read_gray_double(jobs.fused(i));
            if ~isequal(size(A), size(B), size(F))
                error('Shape mismatch A=%s B=%s F=%s', mat2str(size(A)), mat2str(size(B)), mat2str(size(F)));
            end

            messages = strings(0,1);
            for k = 1:numel(metrics)
                metric = char(metrics(k));
                try
                    value = compute_metric(metric, A, B, F);
                    if ~isscalar(value)
                        value = mean(value(:), 'omitnan');
                    end
                    result.(metric)(i) = double(value);
                catch ME_metric
                    messages(end+1,1) = string(metric) + ": " + string(ME_metric.message); %#ok<AGROW>
                end
            end
            legacy_error(i) = strjoin(messages, ' | ');
        catch ME
            legacy_error(i) = "load/preprocess: " + string(ME.message);
        end
    end
    result.legacy_error = legacy_error;
    writetable(result, out_csv);
end

function image = read_gray_double(path_value)
    image = imread(char(path_value));
    if ndims(image) == 3
        image = rgb2gray(image);
    end
    image = double(image); % Match TPAMI Metric_calculation.m: uint8 -> double in [0,255].
end

function value = compute_metric(metric, A, B, F)
    switch lower(metric)
        case 'qmi'
            value = metricMI(A, B, F, 1);
        case 'qsf'
            value = metricZheng(A, B, F);
        case 'qs'
            value = metricPeilla(A, B, F, 1);
        case 'qcb'
            value = metricChenBlum(A, B, F);
        case 'qabf'
            value = Qabf(A, B, F);
        case 'qabf_analysis'
            value = analysis_Qabf(A, B, F);
        case 'qncie'
            value = metricWang(A, B, F);
        case 'qg'
            value = metricXydeas(A, B, F);
        case 'qp'
            value = metricZhao(A, B, F);
        case 'qe'
            value = metricPeilla(A, B, F, 3);
        case 'qviff'
            value = VIFF_Public(A, B, F);
        case 'ms_ssim_src'
            value = analysis_MSSSIM(A, B, F);
        otherwise
            error('Unsupported MATLAB metric: %s', metric);
    end
end
