clear
close all

% PARAMETERS
beta = .9932; % discount factor 
sigma = 1.5; % coefficient of risk aversion
b = 0.5; % replacement ratio (unemployment benefits)
y_s = [1, b]; % endowment in employment states
PI = [.97 .03; .5 .5]; % transition matrix


% ASSET VECTOR
a_lo = -2; %lower bound of grid points
a_hi = 5;%upper bound of grid points
num_a = 10;

a = linspace(a_lo, a_hi, num_a); % asset (row) vector

% INITIAL GUESS FOR q
q_min = 0.98;
q_max = 1;
q_guess = (q_min + q_max) / 2;


%to do the calculations
N = length(a);
m = length(y_s);
c_s=zeros(m*N,N);
s = 1:m*N;
s = reshape(s,N,m)';



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
 while v_tol >.0001;
    % compute the utility value for all possible combinations of k and k':
    value_fn = ret + beta * ...
        repmat(permute((PI * v_guess), [3 2 1]), [num_a 1 1]);
    
    
      % find the optimal k' for every k:
    [vfn, pol_indx] = max(value_fn, [], 2);
    vfn = shiftdim(vfn,2);
    
        % what is the distance between current guess and value function
    v_tol = max(abs( vfn(:) - v_guess(:) ));
    
    % if distance is larger than tolerance, update current guess and
    % continue, otherwise exit the loop
    v_guess = vfn;
    
end

    % KEEP DECSISION RULE
    pol_fn = a(pol_indx);
    
    % Initial distribution
    %guess an initial dstribution over s and a
    % I am guessing this means dsitribution of people 
    % zeros
    Mu= rand(N*m,N*m,1); 
    
     MuNew = zeros(size(Mu));
    for i = 1:m
        MuNew(s(i,:),:)= kron(PI(i,:),sparse(1:N,pol_indx(s(i,:)),ones(1,N),N,N));
    end
    MuNew = MuNew^1000;
    mu = MuNew(1,:);
    %This calculates the invariant distributions
    err = dot(kron(ones(1,m),a),mu);
  
    % now I need to update
    
    if err>= 0.01% price is too low ; too much savinf
        q_min = q_guess; %new q_min must be higher than before   
    else
        q_max = q_guess;   
    end 
    aggsav = q_max-q_min;
    
    
   
    end


 