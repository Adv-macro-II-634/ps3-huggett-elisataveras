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
    vfn = shiftdim(vfn,2);
    
        % what is the distance between current guess and value function
    v_tol = max(abs( vfn(:) - v_guess(:)));
    
    % if distance is larger than tolerance, update current guess and
    % continue, otherwise exit the loop
    v_guess = vfn;
    i=i+1;
 end

 pol_indx = permute(pol_indx, [3 1 2]);
    % KEEP DECSISION RULE        
    g = a(pol_indx); % policy function
    polh=pol_indx(1,:);
    poll=pol_indx(2,:);
    
    
    % Initial distribution
    % uniformely distributed values
    %each will have the probability of:
      Mu = (ones(2*num_a,1))/(2*num_a);
    
    
    posa = zeros(num_a,num_a); posb = posa;

        for i = 1:num_a

        posa(i,polh(1,i)) = 1; posb(i,poll(1,i)) = 1;

        end

    T = [PI(1,1)*posa,PI(1,2)*posa;

        PI(2,1)*posb,PI(2,2)*posb];
    

      %need to do this until is stationary distribution 
    diff   = 100;
    while diff > .0001

        MuNew     = T'*Mu;

        diff   = max(abs(MuNew-Mu));

        Mu       = MuNew;

    end
    
    
    
    MarketClearing=[a,a]*Mu;
    
       %Mu= rand(N*m,N*m,1); 
        % ITERATE OVER DISTRIBUTIONS
      % [emp_ind, a_ind, mass] = find(Mu > 0); % find non-zero indices
    
      %     for ii = 1:length(emp_ind)
       %    apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); % which a prime does the policy fn prescribe?
       %    MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
       %        (PI(emp_ind(ii), :)* mass)';
       
        %need to do this until is stationary distribution 
        % if norm(MuNew-Mu) < 10^-10
       % break
            %end
             %Mu= MuNew;
        %end

      %  end
        
        
   
    % T = zeros(size(Mu));
   % for i = 1:m
   %      T(s(i,:),:)= kron(PI(i,:),sparse(1:N,pol_indx(s(i,:)),ones(1,N),N,N));
   % end
    
    % T^n 
   %  T =  T^1000;
   % mu =  T(1,:);
    %T^n*mu
   % err = dot(kron(ones(1,m),a),mu);
  
    % now I need to update
    q=q_guess;
    if MarketClearing>= 0.01% price is too low ; too much savinf
        q_min = q_guess; %new q_min must be higher than before   
    else
        q_max = q_guess;   
    end 
    aggsav = MarketClearing;
    
    n=n+1
end
    
    disp('price]')
    disp(full(q))

 