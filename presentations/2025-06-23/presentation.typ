#import "@preview/diatypst:0.6.0": *

#set math.equation(numbering: "(1)")

#show: slides.with(
    title: "DDPM",
    date: "23.06.2025",
    authors: "Rishi Vora",
    ratio: 16 / 10,
    toc: false,
)

= DDPMs

== Concept

The idea behind Denoising Diffusion Probabilistic Models (or Diffusion Models for short) is to destroy the training data by adding noise, and then learn to reverse this process to generate new data.

Let us consider the MNIST dataset, which consists of images of handwritten digits from 0 to 9. There exists some manifold in the high-dimensional space of images, that contains all such images and the MNIST dataset must be a submanifold of it. Our aim is to sample new images from this manifold, but the problem is that we do not know the algebraic form of this manifold because it is too complex.

So we will take a point on the submanifold and add noise to it iteratively, to push it outside the submanifold. Then we will train a model to learn to remove the noise step by step, to get back into the manifold.

== Forward Process

We start from a initial image $x_0$ and corrupt it as follows:

$
    x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) epsilon.alt_t
$

Here, $epsilon.alt_t$ is sampled from an independent standard Gaussian distribution $epsilon.alt_t tilde cal(N) (0, I)$ and the parameter $beta_t in (0, 1)$ controls the amount of noise added. The term $sqrt(1 - beta_t) x_(t-1)$ retains a fraction of the previous state.

The transition density of getting a particular $x_t$ given some $x_(t-1)$ is given by

$
    q(x_t | x_(t-1)) = cal(N) (x_t | sqrt(1 - beta_t) x_(t-1), beta_t I)
$

where $cal(N) (x | mu, sigma^2)$ is the probability density at $x$.

The process is repeated for $T$ steps. $beta_1, dots, beta_T$ is the variance schedule.

#pagebreak()

By reparameterizing

$
              alpha_t & = 1 - beta_t \
    overline(alpha)_t & = product_(i=1)^t (1 - beta_i)
$

we can write

$
    x_t & = sqrt(alpha_t) x_(t-1) + sqrt(1 - alpha_t) epsilon.alt_t \
    & = sqrt(alpha_t) (sqrt(alpha_(t-1)) x_(t-2) + sqrt(1 - alpha_(t-1)) epsilon.alt_(t-1)) + sqrt(1 - alpha_t) epsilon.alt_t \
    & = sqrt(alpha_t) sqrt(alpha_(t-1)) x_(t-2) + sqrt(alpha_t) sqrt(1 - alpha_(t-1)) epsilon.alt_(t-1) + sqrt(1 - alpha_t) epsilon.alt_t \
    & = sqrt(alpha_t) sqrt(alpha_(t-1)) x_(t-2) + sqrt(1 - alpha_t alpha_(t-1)) epsilon.alt_(t, t-1) quad (because a epsilon.alt_1 + b epsilon.alt_2 = sqrt(a^2 + b^2) epsilon.alt_3) \
    & dots.v \
    x_t & = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) overline(epsilon.alt)_t
$ <atn>

This way we can go to any arbitrary time step $t$ for a given $x_0$. The transition probability becomes

$
    q(x_t | x_0) = cal(N) (x_t | sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) I)
$

Since all $alpha_i$ are in $(0, 1)$, for large enough $T$, $q (x_t | x_0) arrow.r cal(N) (x_T | 0, I)$, and so we get pure Gaussian noise.

== Backward Process

Now we would like to find the probability density of the reverse process i.e. $q(x_(t-1) | x_t)$. More specifically, for training we want the model to learn to generate $x_0$, so we want to find $q(x_(t-1) | x_t, x_0)$.

By product rule, we can write

$
                 P (A,B,C) & = P (A | B, C) P (B | C) P (C) \
                           & = P (B,A,C) = P (B | A, C) P (A | C) P (C) \
    therefore P (A | B, C) & = (P (B | A, C) P (A | C)) / (P (B | C))
$

So

$
    q(x_(t-1) | x_t, x_0) = (q(x_t | x_(t-1), x_0) q(x_(t-1) | x_0)) / q(x_t | x_0)
$

#pagebreak()

Now

$
    q(x_t | x_(t-1), x_0) = q(x_t | x_(t-1)) quad (because "Markovian process") \
    therefore q(x_(t-1) | x_t, x_0) = (cal(N) (x_t | sqrt(alpha_t) x_(t-1), (1 - alpha_t) I) cal(N) (x_(t-1) | sqrt(overline(alpha)_(t-1)) x_0, (1 - overline(alpha)_(t-1)) I)) / (cal(N) (x_t | sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) I))
$

For a multivariate isotropic Gaussian distribution, the probability density at $x in bb(R)^d$ is given by

$
    cal(N) (x | mu, sigma^2 I) = 1 / (sigma (2 pi)^(1/d)) exp (- 1/(2 sigma^2) (x - mu)^T (x - mu))
$ <migd>

#pagebreak()

Plugging this in and ignoring the coefficient, we get

$
    q(x_(t-1) | x_t, x_0) = exp (& -1/2 ((x_t - sqrt(alpha_t) x_(t-1))^T (x_t - sqrt(alpha_t) x_(t-1)))/(1 - alpha_t) \ & - 1/2 ((x_(t-1) - sqrt(overline(alpha)_(t-1)) x_0)^T (x_(t-1) - sqrt(overline(alpha)_(t-1)) x_0))/(1 - overline(alpha)_(t-1)) \ & + 1/2 ((x_t - sqrt(overline(alpha)_t) x_0)^T (x_t - sqrt(overline(alpha)_t) x_0))/(1 - overline(alpha)_t))
$

We would like to express this in the form of a Gaussian distribution $cal(N) (x | tilde(mu) (x_t, x_0), tilde(sigma)^2 (t) I)$ (so as to "subtract" the Gaussian noise we added earlier), so we need to find the mean and variance.

#pagebreak()

Looking at @migd, we can find $tilde(sigma)^2$ by finding the coefficient of $x_(t-1)^T x_(t-1)$, which is

$
      & 1/2 (alpha_t)/(1 - alpha_t) + 1/2 (1)/(1 - overline(alpha)_(t-1)) \
    = & 1/2 (1 - overline(alpha)_(t))/((1 - alpha_t)(1 - overline(alpha)_(t-1)))
$

$
    therefore tilde(sigma)^2 (t) = ((1 - alpha_t)(1 - overline(alpha)_(t-1)))/(1 - overline(alpha)_(t))
$

This has no dependence on $x_i$ and hence can be written as

$
    tilde(sigma)^2 (t) = tilde(beta) (t) = (1 - overline(alpha)_(t-1))/(1 - overline(alpha)_(t)) beta_t
$

Going by the original paper @ddpm, we assume this to be a constant rather than varying with $t$.

#pagebreak()

Looking at @migd again, we can find $tilde(mu) (x_t, x_0)$ by finding the vector $v$ multiplied to $x_(t-1)^T$, which is

$
    (tilde(mu) (x_t, x_0))/(tilde(sigma)^2 (t)) = v = (sqrt(alpha_t) x_t)/(1 - alpha_t) + (sqrt(overline(alpha)_(t-1)) x_0)/(1 - overline(alpha)_(t-1))
$

$
    therefore tilde(mu) (x_t, x_0) = (sqrt(alpha_(t)) (1 - overline(alpha)_(t-1)) x_t + sqrt(overline(alpha)_(t-1)) (1 - alpha_t) x_0)/(1 - overline(alpha)_(t))
$

This will only depend on $x_t$ because we know $x_t (x_0, overline(epsilon.alt)_t)$ (by @atn)

$
    therefore tilde(mu) (x_t) = 1/(sqrt(overline(alpha)_(t))) (x_t - (1 - alpha_t)/(sqrt(1 - overline(alpha)_t)) overline(epsilon.alt)_t)
$

Therefore, to get $cal(N) (x_(t-1) | tilde(mu) (x_t), tilde(beta))$ we need to estimate the original noise that corrupted the image. We estimate this using a neural network and so we need to define a loss function for training.

If our estimated noise is $epsilon.alt_theta (x_t, t)$, then the loss function is

$
    L_t = E_(x_0, epsilon.alt_t) [bar.v.double overline(epsilon.alt)_t - epsilon.alt_theta (x_t, t) bar.v.double^2]
$

We train a UNet model as per @ddpm. It takes as input the noisy image $x_t$ and the time step $t$, and outputs the estimated noise $epsilon.alt_theta (x_t, t)$.

#line(length: 20%, start: (40%, 10%))

The derivations were based on @higham2023.

= Bibliography

==

#bibliography("./references.bib")
