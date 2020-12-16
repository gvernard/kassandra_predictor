function A_hat_all = train_predict_for_george % comment lines 1, 6 & 8 in order to make it fast
%
% Basic code.
% 

T = readtable('OxCGRT_latest.csv');

load('all_indices.mat');
NoCountries = length(idx_all);

% a subset of countries (mostly from the western world)
% p_test = -1+[9, 10, 13, 17, 21, 58, 59, 60, 71, 72, 73, 76, 82, 83, 85, 87, 90:94, 99, 106, ...
%           108, 111, 115, 116, 119, 124, 154, 163, 170, 171, 185, 186, 187, 205:257];

% various hyper-parameters
thres = 0.1; % for OMP
K = 5;  % used for the definition of the target variable
Llog = 4; % used for the definition of the target variable

Lx = 7; % moving average window

Lc = 9; % smoothness factor. Hihger value implies more smoothness
h = conv(ones(Lc,1)/Lc,ones(Lc,1)/Lc); % coefficients of the smoothing filter

all_names = {};
A_hat_all = zeros(NoCountries, 12); % 1+11

for p = 1:NoCountries
    % indices in the database, T
    idx = idx_all{p};
    idx = idx(1):idx(end)-2;
    idx_ = idx(1):idx(end)-1;

    
    % PREPROCESSING (X is an auxiliary matrix)
    X = T.ConfirmedCases(idx);

    % remove NANs
    X(isnan(X)) = 0;

    % filtfilt is like filter without linear phase
    % filter is like convolution with the length of the output being equal
    % to the length of the input signal
    X = diff(filtfilt(ones(Lx,1)/Lx, 1, X));

    X = [X, T.C1_SchoolClosing(idx_)];
    X = [X, T.C2_WorkplaceClosing(idx_)];
    X = [X, T.C3_CancelPublicEvents(idx_)];
    X = [X, T.C4_RestrictionsOnGatherings(idx_)];
    X = [X, T.C5_ClosePublicTransport(idx_)];
    X = [X, T.C6_StayAtHomeRequirements(idx_)];
    X = [X, T.C7_RestrictionsOnInternalMovement(idx_)];
    X = [X, T.H1_PublicInformationCampaigns(idx_)];
    X = [X, T.H2_TestingPolicy(idx_)];
    X = [X, T.H3_ContactTracing(idx_)];
    X = [X, T.H6_FacialCoverings(idx_)];

    [N, L] = size(X); % 'Length of time-series' x 'Number of interventions'

    % remove NANs
    for i = 2:N
        for j = 1:L
            if isnan(X(i,j))
                X(i,j)=X(i-1,j);
            end
        end
    end
    
    
    % DATASET CONSTRUCTION
    % ground truth for new cases
    y = X(2:end, 1);
    
    % target variable
    z = log(Llog+X(:,1));
    z = filter([1, zeros(1,K-1), -1], 1, z)/K;
    
    % feature/variable/predictors matrix/table
    Psi = ones(N,1);
    for i = 2:L
        X_int = X(:,i)/10;
        
        X_int = filter(h,1,X_int);
        
        Psi = [Psi, X_int]; % add the feature
        
%         plot(X_, 'b');hold on;
%         plot(X_int, 'k');hold off;
%         pause;
    end
    
    %%% TRAINING
    % split time-series into training and testing sets
    idx_tr = 1:320;
    idx_te = 321:N-1;
    
    % OMP is a lightspeed-fast, greedy feature selection algorithm
    [a_hat, ~] = omp(z(idx_tr), Psi(idx_tr,:), L, thres, false(1, size(Psi,2)));
    
    % save the coefficient vector
    A_hat_all(p,:) = real(a_hat); % real part should not be necessary

    
    % PREDICTIONS
    z_hat = Psi(idx_te,:)*a_hat'; % predicted rate of the log(new cases)
    
    y_hat = zeros(length(idx_te),1); % predicted new cases
    y_hat(1) = (Llog+y(idx_tr(end)))*exp(z_hat(i))-Llog;
    for i = 2:length(idx_te)
        y_hat(i) = (Llog+y_hat(i-1))*exp(z_hat(i))-Llog;
    end

    
    % PLOTTING
%     plot(idx_tr, y(idx_tr), 'b');hold on;
%     plot(idx_te, y(idx_te), 'b--');
%     plot(idx_te, y_hat, 'r--');hold off;
%     legend('Trainset - True', 'Testset - True', 'Testset - Predicted', 'Location', 'NorthWest');
    country_name = T.CountryName(idx(1));
    region_name = T.RegionName(idx(1));
%     title([num2str(p) '.  ' country_name{1} '   ' region_name{1}]);
%     set(gca, 'Fontsize', 14);
% %     pause;

    % names for all countries and regions
    if isempty(region_name{1})
        all_names{p} = country_name{1};
    else
        all_names{p} = [country_name{1} '__' region_name{1}];
    end

end

S = array2table(A_hat_all);
S.Name = all_names';

% save table into a csv file
writetable(S, 'model_coef.csv');



