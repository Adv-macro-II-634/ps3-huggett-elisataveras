clear 
close all
%%%% Set up parameters
alpha = 0.35;
beta = 0.994;
delta = 0.025;
sigma = 1.5;
S= [0.97 0.03; 0.5 0.5];
S_sta = S ^ 100;
b = 0.5; % replacement ratio (unemployment benefits)
Y =  [1, b];

%%%% Set up discretized state space
a_min = -2;
a_max = 5;
num_a = 10; % number of points in the grid for k

a = linspace(a_min, a_max, num_a);

% INITIAL GUESS FOR q
q_min = 0.98;
q_max = 1;
q_guess = (q_min + q_max) / 2;


aggsav = 1 ;
while abs(aggsav) >= 0.01 ;

a_mat = repmat(a', [1 num_a]); % this will be useful in a bit

%%%% Set up consumption and return function
    c = bsxfun(@minus, a', q_guess * a);
    c = bsxfun(@plus, c, permute(Y, [1 3 2]));
ret = (c.^ (1 - sigma))./ (1 - sigma); % return function high

% value function guess 
v_guess = zeros(2,num_a);
tol=1 ; 
tic
while tol > .0001  %& its < maxits
    % compute the utility value for all possible combinations of k and k':
    value_fn = ret + beta * ...
        repmat(permute((S * v_guess), [3 2 1]), [num_a 1 1]);
    
    
      % find the optimal k' for every k:
    [vfn, pol_indx] = max(value_fn, [], 2);
    vfn = shiftdim(vfn,2);
    
        % what is the distance between current guess and value function
    d = max(abs( vfn(:) - v_guess(:) ));
    
    % if distance is larger than tolerance, update current guess and
    % continue, otherwise exit the loop
    v_guess = vfn;
    
end

    % Initial distribution
    Mu= rand(T_sim, 1); % uniform distribution over s and A 

  
    % ITERATE OVER DISTRIBUTIONS
    [emp_ind, a_ind, mass] = find(Mu > 0); % find non-zero indices
    
    MuNew = zeros(size(Mu));
    for ii = 1:length(emp_ind)
        apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); % which a prime does the policy fn prescribe?
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
            (PI(emp_ind(ii), :) * mass)';
    end
 
        
    end


 