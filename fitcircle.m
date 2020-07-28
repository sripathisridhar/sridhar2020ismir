% Sripathi Sridhar
% Helix PART 2
% Linear least squares circle fit to a set of 2d points
% [1] Coope

function [x, r, error] = fitcircle(A)

%     A = [0.7,4.0; 3.3,4.7; 5.6,4.0; 7.5,1.3; 6.4,-1.1; 4.4,-3.0; 0.3,-2.5; -1.1,1.3]';
    [n,m] = size(A);
    
    y = [A', ones(m,1)]\sum(A.*A)';
    
    x = 0.5 * y(1:n);
    r = sqrt(y(n+1) + x'*x);
    
    error = 0;
    for i = 1:m
        error = error + abs(r - sqrt(abs(sum(x.*x - A(:,1).*A(:,1)))));
    end
    error = error/m;
    
end
