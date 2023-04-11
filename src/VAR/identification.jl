
struct CholeskyVAR <: AbstractIdenticationMethod end
function identify!(var::VAR{E}, identification_method::CholeskyVAR) where {E<:BayesianEstimated}
    
    
