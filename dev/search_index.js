var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MacroEconometrics","category":"page"},{"location":"#MacroEconometrics","page":"Home","title":"MacroEconometrics","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MacroEconometrics.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MacroEconometrics]","category":"page"},{"location":"#MacroEconometrics.BayesianEstimated","page":"Home","title":"MacroEconometrics.BayesianEstimated","text":"Since conjugacy is incredible rare and often not actually plausible (the priors are not plausible) in actual work, we will assume that all Bayesian estimated quantities are just samples, and can thus be put in an array. Additional meta data, such as warning information during sampling, etc can be put in the metadata field.\n\nBehaviour\n\nMost basic operations will directly be forwarded to the value field. As such, the type can be used like any other array, including multiplication with other arrays, iteration, etc.\n\nFields\n\nvalue::Array{T}: The actual values of the estimated quantity. Should be an Array; If multiple chains have been used, then the chains should be stacked along the third dimension\nmetadata::M: Any additional data that one wishes to save relating to the estimation. This could be warnings from the sampling algorithms, etc.\n\n\n\n\n\n","category":"type"},{"location":"#MacroEconometrics.Estimated","page":"Home","title":"MacroEconometrics.Estimated","text":"The idea is to abstract he model from the way it has been estimated. In it's most essential form, a model is just a struct, with the same structure no matter how is has been estimated. The only time that estimation matter is when statistics about estimated quantities are asked for. Thus, any statistic that needs to be estimated in a model should be of type Estimated. Estimated itself can then be either Frequentist or Bayesian.\n\n\n\n\n\n","category":"type"},{"location":"#MacroEconometrics.FixedEstimated","page":"Home","title":"MacroEconometrics.FixedEstimated","text":"\"   Fixing the estimation.\n\nSometimes we want to fix a parameter to a specific value. Although this parameter is no-longer technically estimated, it is still an estimable quantity and thus is still a subtype of Estimated. \n\nFields\n\n-value::Array{T}: The value of the quantity\n\n\n\n\n\n","category":"type"},{"location":"#MacroEconometrics.VAR","page":"Home","title":"MacroEconometrics.VAR","text":"VAR model\n\nEvery VAR model is of the form\n\ny_t = b_0 + B_1 y_t-1 +  + B_p y_t-p + varepsilon_t\n\nwhere B_i is n times n and b_0 is an n times 1 vector. For the disturbance terms, varepsilon_t we generally assume varepsilon_t sim N(0 Sigma). This can therefore be equivelently written as \n\ny_t = b_0 + B tildey_t-1 + varepsilon_t\n\nwhere B is now n times np and tildey_t-1=(y_t-1  y_t-p).\n\nGeneral Information\n\nVAR models are only approprite for regularly spaced observations. We therefore check whether this is the case here\n\nFields\n\nn::Int: Number of variables in the VAR.\np::Int: Numbe of lags; See the equations above.\nB::E<:Estimated: Lag coefficient matrix; See the equations above; Should be a subtype of Estimated. \nb0::E<:Estimated: Intercept vector; See equations above.\nΣ::E<:Estimated: Covariance matrix of disturbance terms; See equation above.\ndata::TSFrame: A time series data frame containing the data used for the VAR. Observations should be regularly spaced.\n\n\n\n\n\n","category":"type"}]
}
