data{
    int n_types;
    int n_props;
    int n_events;
    int lengths[n_props];
    real times[n_events];
    int types[n_events];
    int the_type;
    real<lower=0> height_lambda;
    real<lower=0> decay_lambda;
    real<lower=0> baseline_lambda;
}

parameters{
    real<lower=0> heights[n_types];
    real<lower=0> decays[n_types];
//    real bases[n_types];
    real<lower=0> baseline;
}

transformed parameters{

}

model{
    // temp variables
    real fake_lambda;
    real actual_lambda;
    real prob;
    int z;
    int c;

    // prior
    for (i in 1:n_types){
        heights[i] ~ exponential(height_lambda);
	decays[i] ~ exponential(decay_lambda);
    }
    baseline ~ exponential(baseline_lambda);

    // likelihood

    c <- 0;

    for (p in 1:n_props){
    	for(i in 1:lengths[p]){
	    // calculate lambda at event time due to previous events.
	    // for now, incorrectly assume that probably event is 1 is lambda value sent through saturating function
	    // for now, pretending that i, k index position within the property
	    fake_lambda <- baseline;
	    for(k in 1:(i-1)){
	    	//fake_lambda <- fake_lambda + bases[types[c+i]] + heights[types[c+k]] * exp(-1.0 * (times[c+i] - times[c+k]) / decays[types[c+k]]);
		fake_lambda <- fake_lambda + heights[types[c+k]] * exp(-1.0 * (times[c+i] - times[c+k]) / decays[types[c+k]]); 
	    }
	    actual_lambda <- 1*(1-(1/log(2))*log(1+exp(-4*fake_lambda)));
	    prob <- actual_lambda;
	    // make indicator representing if observed event is the_type (violent crime)
	    if(types[c+i] != the_type){
		// no event
		z <- 0;
	    }
	    else{
	        z <- 1;    
	    }
	    z ~ bernoulli(prob);
	}
	c <- c + lengths[p];
    }
}