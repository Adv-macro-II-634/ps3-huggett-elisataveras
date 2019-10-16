clear
close all

% PARAMETERS
beta = .994; % discount factor 
sigma = 1.5; % coefficient of risk aversion
b = 0.5; % replacement ratio (unemployment benefits)
y_s = [1, b]; % endowment in employment states
PI = [.97 .03; .5 .5]; % transition matrix
PI_t = PI ^ 100;

% ASSET VECTOR
a_lo = -2; %lower bound of grid points
a_hi = 5;%upper bound of grid points
num_a = 401;

a = linspace(a_lo, a_hi, num_a); % asset (row) vector

% INITIAL GUESS FOR q
q_min = 0;
q_max = 1;
q_guess = (q_min + q_max) / 2;

n=1;
aggsav = 1 ;
while abs(aggsav) >= 0.01 ;
    % CURRENT RETURN (UTILITY) FUNCTION
    cons = bsxfun(@minus, a', q_guess * a);
    cons = bsxfun(@plus, cons, permute(y_s, [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
  
% negative consumption is not possible -> make it irrelevant by assigning
% it very large negative utility
ret(cons < 0) = -Inf;


    % INITIAL VALUE FUNCTION GUESS
    v_guess = zeros(2, num_a);
    % VALUE FUNCTION ITERATION
    v_tol = 1;
    i=1;
 while v_tol >.0001;
    % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE
    value_fn = ret + beta * ...
        repmat(permute((PI * v_guess), [3 2 1]), [num_a 1 1]);
    
       % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
    [vfn, pol_indx] = max(value_fn, [], 2);
    vfn = permute(vfn,[3 1 2]);
    
        % what is the distance between current guess and value function
    v_tol = max(abs(v_guess(:) - vfn(:)));
    
    % if distance is larger than tolerance, update current guess and
    % continue, otherwise exit the loop
    v_guess = vfn;
    i=i+1;
 end

 pol_indx = permute(pol_indx, [3 1 2]);
    % KEEP DECSISION RULE        
    g = a(pol_indx); % policy function
    
    
    % Initial distribution
    % uniformely distributed values
    %each will have the probability of 1/n:
    Mu = (1/(2*num_a))*ones(2,num_a);
    %need to do this until is stationary distribution 
    Mu_tol   = 1;
    while Mu_tol > 1e-8;   
    % ITERATE OVER DISTRIBUTIONS
    [emp_ind, a_ind] = find(Mu > 0); % find non-zero indices
        
          MuNew = zeros(size(Mu));
    for ii = 1:length(emp_ind)
        apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); % which a prime does the policy fn prescribe?
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
            (PI(emp_ind(ii), :) * Mu(emp_ind(ii), a_ind(ii)))';
    end
        
        Mu_tol   = max(abs(MuNew(:)-Mu(:)));
        Mu= MuNew;
        
    end
   
    MarketClearing=sum(sum(Mu.*g));
    
    % now I need to update the information    
    if MarketClearing>= 0.01% price is too low ; too much savinf
        q_min = (q_min + q_max) / 2; %new q_min must be higher than before   
    else
        q_max = (q_min + q_max) / 2;   
    end 
    q_guess = (q_min + q_max) / 2;
    aggsav = MarketClearing;
    n=n+1;
    %save the price
    q=q_guess;
    
    disp('market-clearing; iteration; price')
    disp(full([MarketClearing n q]))
end


%Lorenz Curve
%calculate the income
 income_E = a+1;
 income_U = a+b;
 %multiply by distribution
  Income = [income_E income_U].*Mu(:)';
  %summing total income and sorting income
  TotalIncome=sum(Income(:));
 [Income sort] = sort(Income);
 %get the percentage 
 perc = Income/TotalIncome;
 %population percentage given by the stationary distribution
 pop = Mu(:)';
 pop = pop(sort);
 %cumulative sum population percentage and percentage of income
 for i = 2:length(Income)
     perc(i) = perc(i)+perc(i-1);
     pop(i) = pop(i)+pop(i-1);
 end
 
 plot([0 1],[0 1],'r',pop,perc,'--','Linewidth',1);
 title('Lorenz Curve')
 xlabel('Cumulative Share from Lowest to Highest Income') 
ylabel('Cumulative Share of Income')
legend({'Equality line','Lorenz Curve'},'Location','southeast')



%extra credit 
% calculate expected util
% in constant consumption 
c=PI_t(1,1)*1+ PI_t(1,2)*b;
ret = (c .^ (1-sigma)) ./ (1 - sigma);
W=(ret*(1/(1-beta)));

 %create matrix with WE and WU
 WFB=[W*ones(1,num_a);
        W*ones(1,num_a)];
    
% need to find value function value for each chosen asset:

vf_chosen=vfn(pol_indx);

%find the index of those that has higher value function given their policy
%function
 [emp_indVF, a_indVF] = find(WFB>vf_chosen); % find non-zero indices
 
  distInc = zeros(size(Mu)); 
      for ii = 1:length(emp_indVF)
         distInc(ii) = Mu(emp_indVF(ii), a_indVF(ii));
      end   
    %percentage of population that are better off 
    sum(distInc(:))
 
% calculate lambda:

lambda=(WFB./vf_chosen).^(1/1-sigma)-1;

 Welfaregain=sum(sum(Mu.*lambda));
 

 