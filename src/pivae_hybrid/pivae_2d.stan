functions {
  vector layer(vector x, matrix W, vector B) {
    return W' * x + B;
  }
  vector generator_stan(vector z, matrix W1, matrix W2, matrix W3,
                        vector B1, vector B2, vector B3) {
    return layer(tanh(layer(tanh(layer(z, W1, B1)), W2, B2)), W3, B3);
  }
}
data {
  int<lower=1> p;
  int<lower=1> p1;
  int<lower=1> p2;
  int<lower=1> beta_dim;
  int<lower=1> n_train;
  matrix[p, p1] W1;
  vector[p1] B1;
  matrix[p1, p2] W2;
  vector[p2] B2;
  matrix[p2, beta_dim] W3;
  vector[beta_dim] B3;
  // Exact QR sufficient statistics from training [Phi, y] only.
  matrix[beta_dim, beta_dim] R;
  vector[beta_dim] qty;
  real<lower=0> residual_sse;
  real<lower=0> sigma_prior_mean;
  real<lower=0> sigma_prior_sd;
}
parameters {
  vector[p] z;
  // The original sigma2 was a standard deviation, despite its name.
  real<lower=0.001> sigma;
}
transformed parameters {
  vector[beta_dim] f = generator_stan(z, W1, W2, W3, B1, B2, B3);
}
model {
  z ~ std_normal();
  sigma ~ normal(sigma_prior_mean, sigma_prior_sd);
  // Equal to sum(normal_lpdf(y_train | Phi_train * f, sigma)),
  // up to a parameter-independent constant; uses every fitting row.
  target += -n_train * log(sigma)
            -(dot_self(R * f - qty) + residual_sse) / (2 * square(sigma));
}
