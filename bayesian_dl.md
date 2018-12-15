
These are notes preparing for my talk in Nepal.

# Outline

First, let's write a big outline.

I. Introduction
  A. The view from 20,000 feet
      1. Why do we run ML systems?
      2. To provide compressed representations, for ourselves and for our machines
      3. These representations allow us to make sense of the world, and act in a
         way that aligns with our values. Exampel with pesticides.
      4. This is much more than classification! As cool as it can be to have
         accurate classifiers and image generators,
  B. Learning Objectives
      1. Illuminate the path between historical foundations and modern research
         in bayesian deep learning
      2. Add a few more models and algorithms to your catalog of simple
         examples, which you can refer to whenever you read a new abstract paper
      3. Understand underlying math and code for a few example algorithms, so
         you can use them in your own applications
  C. The value of deep learning is that it attempts to automatically learn these sorts of
features from the overall class labels.
ow we will learn
      1. Draw pictures. Fancy math is usually just a way of capturing some
         geometric intutition.
      2. Toy datasets: It's much easier to understand a complicated algorithm
         watch it in action on a simple datset.
II. Let's agree on the basics
  A00. Example datasets
      1. The iris dataset -- succinct representation of types of flowers
         (species), and further, automatic
      2. Images dataset -- this is a toy example, but there are tons of examples
         showing computer vision is important. E.g., classifying different types
         of pests in images.
      3. Isn't images way more complicated than iris? Sort of! One is
         high-dimensional, which means many methods won't work well (for reasons
         that are beyond the scope of this essay). But don't ever forget that at
         the end of the day we're still working with vectors, it's just that
         some are length 2 and others are length 256.
  A. Every single machine learning algorithm ever
      1. Model space, search strategy, objective function
  B. Specialization to bayesian machine learning
      1. Model class, algorithm, inference mechanism
  C. Deep learning vs. Bayes
      1. Formal inference vs. ad hoc point estimates. E.g., let's have reliable
         estimates of our uncertainty in a disease diagnosis. Or, which examples
         a human should inspect by hand.
      2. Complex functions vs. simple models
      3. Modular optimization components vs. fragile, hand-constructed sampling
         mechanisms
  D. Bayesian inference
      1. The philosophical description, and some geometry / probability
      2. Math basics: Beta bernoulli
      3. Some examples from papers
  E. Deep learning
      1. Typical recipe: recursive nonlinearities, stochastic gradient descent
      2. Basics: Stacked autoencoder
III. The world before the variational autoencoder
  A. Variational inference
      1. Derivation of the ELBO
      2. But the true posterior is way too complicated, we can't possibly do
         this maximization
      3. Let's relax this requirement: Mean-Field
  B. For the gaussian mixture model
      1. Show this on the Iris dataset
  C. Stochastic variational inference
      1. What if we had had millions of points? It would have taken forever just
         to make one update of the means.
      2. Distinction between local and global factors
          a. In principle, even from a few points, we can learn a bit about the global factors
          b. Let's update the local factors a few at a time
          c. Then let's take a step to optimize the global factors
      3. Example application on the Iris dataset
      4. Example application on the images dataset
IV. The Variational AutoEncoder
  A. What about more flexible likelihoods?
      1. There is a superficial connection between autoencoders and bayesian
         inference. p(x | z) and p(z | x). Can we make this connection more
         formal?
  B. Problem: we can't possibly find the posteriors, even after making a
  mean-field approximation.
      1. How are you possibly going to see how the objective changes when I
         change the parameters in my posterior slightly?
      2. Naive answer: Sample objective after changing parameters phi slightly
         in a many directions. Take step in that makes you descend most quickly.
         Problem: You're never going to be able to do that many steps of this
         procedure.
      3. An answer: Reparametrization. Decouple the posterior parameters from
         the noise mechanism. You can evaluate the gradient directly.
  C. To avoid many local updates: amortization
      1. This is actually less flexible than the original q(z | x) from standard
         mean-field inference
      2. But it's way more modular and doesn't seem to suffer much
  D. Example
      1. The iris dataset
      2. The images dataset
V. Normalizing Flows
  A. Okay, so we were working with mean-field factorization
      1. What if you could do something a bit more rich?
      2. How can we turn a simple density into a more complicated one?
  B. Composition! This is the same trick as for generic deep learning.
      1. Aside: The change of variables formula
         a. Write the formula
         b. Show the picture
  C. Examples
      1. Picture of planar flow
      2. Picture of radial flow
      3. Example code
VI. Wrapping up
   A. What have we learned: Show some of the pictures
   B. Where can you go from here (point them to some of the workshops)


## Some History

Here are two classic papers from 1990,

1. Sampling Based Approaches to Calculating Marginal Densities
2. Handwritten Digit Recognition with a Back-Propagation Network

The first paper is notable because it showed how Bayesian Inference could be
used beyond problems for which integration formulas were available. It led to a
renaissance in Bayesian statistics, because people realized that a generic
computational module could replace tedious mathematical derivations when
attempting to do formal inference (those familiar with statistics might notice
parallels with the bootstrap).

The second paper is one of the first showing how neural networks could be used
to do image classification on digits. It was an early demonstration that
automatically learning good representations during modeling not only let you
avoid the tedious work of hand-crafting features on a problem-by-problem basis,
it could actually lead to improved results.

These papers seem pretty different, even though they both are about working with
data and involved computation to automate otherwise tedious, expert-labor
intensive processes. Not only were they intended for completely different
communities (statistics vs. machine learning), the proposed systems provide very
different types of value. The value of the Bayesian paradigm that Gelfand and
Smith made psosible was a probabilistic description of a system under study,
meaning that you have a sense of the mechanisms that generated your data (and a
guess about your uncertainty in inference). On the other hand, it assumed you
had already derived meaningful features and knew how to specify an explicit
probability model. The neural framework, on the other hand, requires minimal
explicit formulation, making it possible obtain accurate predictions from
essentially raw input data. That said, it falls far short of providing
formal inference -- it doesn't automatically give any idea of the uncertainty in
predictions, let alone the uncertainty associated with resulting estimation.

The tension manifest here would be articulated a few years later in Leo
Brieman's "Two Cultures" essay. The essay was contraversial -- just take a look
at the associated discussion papers. It's too bad that this was the case: in an
alternate universe, the tension might have lead to a concerted effort to get the
best of both cultures: representation learning in an inferential framework.

Fast forwards twenty years, and it seems like this reconciliation might finally
be happening, in a field called Bayesian Deep Learning.

## Why do we do ML?

There is more to ML than handwritten digit classification (though I'm sure the
post office is very happy about our progress there). The previous section
alluded to some of the main goals of the field: we would like more or less
automatic systems that help us (or our machines) make sense of the world and
action in a way that aligns with our values (the sensing + acting dichotomy is
one I first heard from Zoubin Ghahramani on a panel at NeurIPS 2018).

If the goal is to facilitate human decision-making, the exercise is called
intelligence augmentation. One nice example of this type of work is [reference],
which showed how to quickly count the white flies on cassava leaves, which is
otherwise an extremely time-consuming and tedious activity.

I would reserve the term artificial intelligence for the case that we want the
machine to do the decision-making for us. A canonical example is the choice of
which advertisement to show along with search engine results (can you imagine if
there were a human choosing ads each time someone searched google??). Of course,
in reality, the line can be blurry (e.g,. systems that handle most cases
automatically, but ask people to give feedback on a few difficult ones), and in
any case there's point quibbling about semantics.

In either case, it is important to have elements from both of the two cultures,

  1. We want our systems to be running real time and interfacing with the world.
    We don't want to limit ourselves to problems whose outcome are histograms
    that we can present to policy or scientific experts.
  2. We need some probabilistic description (many these days would argue causal,
    as well) of (a) the system producing the data (b) the various representations
    and predictions made by the system. These descriptions give us (and our
    machines) much more to work off of, when we have to actually act in the real
    world.
  3. We want to work as little as possible, because we are, by and large, very
    lazy. That means minimal data preprocessing, feature extraction, model
    specification, and hyperparameter tuning.

Alright, enough philosophizing for now.

## Learning Objectives

The main objective of this tutorial is to leave you equipped with enough of a
foundation in bayesian ML and deep learning to feel comfortable reading recent
literature in the subject. E.g., I want you to be able to critically assess the
contributions made by the papers at this workshop. To get there, I propose three
subgoals,

  1. Be able to trace out some milestones between historical foundations
      (variational inference) and modern research (normalizing flows) in
      bayesian deep learning.
  2. Add a few more models and algorithms to your personal catalog of simple
      examples, which you can refer to whenever you read a new papers.
  3. Understand underlying math and code for a few example algorithms, so you
      can use them in your own applications.

We'll make use of a few learning devices in order to reach these goals. First,
we'll try to draw lots of pictures, with the hope that fancy math is geometric
intuition in disguise. Second, we'll make use of toy datasets. While this risks
losing sight of the broader context, it's usually much easier to understand a
complicated algorithm by watching it in action on simple datasets.

## Example Datasets

We'll run our experiments on two datasets:
[Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) and
[CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10). The Iris dataset is a
classic dataset (from 1936!) giving measurements on characteristics of Iris
flowers (lengths and widths of both sepals and petals). There are 50 samples for
each of three species. So, overall, there are 150 rows and 4 columns. Since
there are only four features, we can plot each of them against each other, see
the figure below, generated by [this
notebook](https://www.kaggle.com/xuhewen/iris-dataset-visualization-and-machine-learning).

[Iris](figure/iris_scatter.png)

This doesn't seem like much by modern standards, but it was enough to motivate
R. A. Fisher's development of an early classification system: Linear
Discriminant Analysis. The idea is that given the four measurements on some new
flower, you would be able to automatically tell which species it came from.

CIFAR-10 is an dataset made up of thousands of small (32 x 32 pixels) labeled
images. It was introduced in a PhD thesis in 2009, and includes 60,000
hand-labeled images from each of 10 classes. Some examples are shown below.

[cifar10](figure/cifar10.png)

At first, this might seem completely different from the Iris dataset. But
stepping back, the problems presented by these datasets are actually quite
conceptually similar. Pixels in these images are like feature measurements in
Iris, and in the end, the goal is to determine a class label for each new sample
(an image) given a large collection of measurements about it (the 32 x 32 x 3 = 3072
numbers given the intensity for red, green, and blue colors for each pixel in
that image). In the Iris problem, we have use 4 measurements to make a decision
about which of 3 classes a new flower belongs, and we have 150 examples on which
to design our system. For CIFAR, when presented with a new image (think of it
as 3072 numbers), we need to decide which of 10 classes it belongs to, and we
have 60,000 examples on which to base our rule.

As similar as these problems seem, throwing Linear Discriminant Analysis at this
problem is not a good idea, though. There are two related issues,

1. 3072 measurements is many more than 4, and there might be much more
   complicated correlation structures than what we saw in the Iris dataset.
2. While the four measurements in the Iris dataset have been chosen to provide
   the most relevant description of the flowers, raw pixel values don't mean
   much on their own.

To address these problems, we might imagine creating new features for each
image, like "lives in a forest" or "is used for traveling," which would really
help with classifying the CIFAR images. Unfortunately this would be very time
consuming to do, and you would have to design new features for each image
classification task.

The value of deep learning is that it attempts to automatically learn these
sorts of features directly from the overall class labels.

>>> Exercise: Plot a scatterplot of the red intensity value of the top-left pixel of
a random sample of 1000 images in CIFAR against its neighbor immediately to
the right. How correlated are they? What about the top-left and bottom-right
pixels?

## Questions to ask every algorithm you meet

It can be intimidating to approach unfamiliar algorithms, which are often hidden
behind a thicket of abstract notation or unreadable code. While it's often our
first instinct to try to dive into every little detail and make sense of
everything, it is useful to first try to answer a few high-level questions. Here
are a few that I heard from Jerry Friedman, though everyone comes up with their
own list with time,

1. What is the model class?

  In most problems, you are attempting to approximate structure in some
  complicated dataset using some simple(ish) functions. The resulting fitted
  functions are supposed to provide a simple description of a complex dataset or
  let you make statements about unseen samples (e.g., what species of flower is
  this new sample)?

  Different methods only consider certain possible functions. For example,
  linear regression only allows linear functions from ${mtex`x`} to ${mtex`y`},
  while logistic regression is restricted sigmoid functions. Regression trees
  learn piecewise functions, split across a partition of ${mtex`x`}. Even deep
  learning, for all the attention it gets, will only be able to fit particular
  functions, depending on its particular architecture.

  This encompasses both supervised and unsupervised problems, if (a) you think
  of unsupervised problems as supervised problems where good labels just happen
  to be missing or (b) you imagine summarizing your dataset by some probability
  density function.

2. What is the evaluation criteria?

  How are you going to tell if one fitted function is better than another?
  Often, this is made explicit in terms of an objective or likelihood function.
  But often, the criteria are implicit (think about partial least squares, for
  example).

  This question is linked to a deeper question of whether the algorithm's
  evaluation criteria reflects real-world utility. For example, if you were
  trying to detect fraudulent credit card transactions, what might matter is not
  necessarily the overall accuracy of the predictions, but the accuracy of the
  100 cases flagged as most likely to be fraudulent, since this may be as many
  as you would be able to inspect at a time, anyways.

3. What is the search strategy?

  Even if two algorithms agree on the types of functions they are trying to fit
  and the criteria that will be used to tell if one is better than another, they
  might differ in the way they choose to search through the space of possible
  functions. For example, one might look at all available datapoints before
  deciding to update the current approximation, while another might make lots of
  small changes after looking at a few datapoints at a time (as in the
  difference between gradient and stochastic gradient descent methods).

In a bayesian setting, there is a parallel way of decomposing methods according
to the underlying model, inference, and algorithm -- this is a view that often
comes up in discussions of bayesian statistics, but which I think is most
clearly articulated in this talk by Shakir Mohamed.

1. What is the underlying model?

  We need to have some way of encoding existing beliefs about the world, along
  with the processes that lead to the data that we've collected. This is the
  foundation that we'll use to revise our description of the world.

  In practice, this is usually done by specifying a probability distribution over
  the observed data, along with a prior over plausible distributions, before ever
  seeing any data. As in the "model class" question earlier, these choices
  restrict the space of possible probabilistic approximations to the world that we
  will ever be able to find, even in the best case.

2. What is the inference criterion?

  We may be interested in various types of probabilistic descriptions. For
  example, we may be happy with a mode from the posterior, or we may want
  arbitrary samples from it. Alternatively, we may be happy with certain
  approximations to the posterior, if we think the statistical-computational
  efficiency tradeoff is reasonable.

3. What is the algorithm?

  For a given model, there are often many avenues towards any inferential goal.
  For example, you may have your choice between different types of MCMC
  or optimization procedures.

## Getting the best of both worlds

In the introduction, we made a philosophical case that it would be nice if the
Bayesian and Deep Learning perspectives could be made more compatible. Let's
consider some of the more practical implications here,

1. Point estimates vs. Posteriors: Traditional deep learning methods only give
   you predictions and parameter estimates, without any evaluation of
   uncertainty for either.
2. Complex functions vs. Simple Models: Traditional Bayesian analysis is usually
   constrained to relatively simple function classes, like Generalized Linear
   Models, where conjugate priors are available or MCMC can be trusted to
   converge in a reasonable amount of time.
3. Modular vs. hand-crafted algorithms: Deep learning methods tend to be made
   from isolated, modular components, that can be easily transferred from
   application to application, while Bayesian inference procedures often need to
   be derived on a problem-by-problem basis.

Before describing approaches to unify these perspectives, let's first consider
one pedagogical example from each field: the beta-binomial model for estimating
coin flipping probabilities and a feedforwards convolutional network for
classifying handwritten digits. These are provided in the scripts
beta_binomial.py and mnist.py.

For the coin flipping problem, we'll are given the task of identifying the
potential bias in a coin, after watching ${mtex`n`} flips. In the usual Bayesian
way, we define a prior distribution over ${mtex`u`} encoding our initial beliefs
about the heads probability, along with a likelihood model for the outcome of
i.i.d. coin flips conditional on the true underlying value. Since we assume the
flips are independent and the probability is constant across all flips, a
binomial distribution makes sense for this likelihood, ${mtex`x \vert u \sim
\Bin\left(x \vert n, u)`}. For the prior, we may be inclined to think that the
underlying probability is probably near 1/2, since we don't think it was
tampered with to mislead us for any reason. We could represent this with any
distribution that places more mass around 1/2, but a particularly convenient way
to do this is to set ${mtex`u \sim \Beta\left(u \vert 10, 10\right)`}, since
it turns out that this distribution is conjugate to the binomial likelihood.
This means that Bayes rule,

${mtex`p\left(u \vert x\right) = \frac{p\left(x \vert u\right)}p\left(u \right)}{p\left(x\right)}`}

can be worked out analytically, by computing the integral ${mtex`p\left(x\right)
= \int_{0}^{1} p\left(x \vert u\right)p\left(u\right)`}. Most discussions of
this model would subject you to the actual calculation at this point, and as a
matter of culture, I recommend you try it. In fact,

*Exercise: Show that ${mtex`u \vert x \sim \Beta\left(u \vert 10 + x, 10 +
n - x\right)`}, first by doing the integral in the denominator of Bayes rule and
then alternatively noticing that the unnormalized function ${mtex`p\left(u\vert
x\right) \propto p\left(x \vert u\right)p\left(u\right)`} has a familiar beta
density form.*

That said, I think it's more informative to actually write some code and look at
some pictures. A histogram from many draws from the prior histogram is plotted
below. You can fiddle with the two sliders to see how changing the two
parameters of the prior beta distribution would capture different prior beliefs.

In your prior, you would draw a probability from this histogram and then flip a
coin with that probability of coming up heads for the rest of your the draws.

In the probabilitic programming language `pyro`, you can describe the prior using
the following lines of code,

sql(`
  # define the hyperparameters that control the beta prior
  alpha0 = torch.tensor(10.0)
  beta0 = torch.tensor(10.0)

  # sample f from the beta prior
  p = pyro.sample("heads_prob", dist.Beta(alpha0, beta0))
`)

and then the likelihood (a sequence of Bernoulli coin flips) using the following
(somewhat obtuse) syntax

sql(`
with pyro.iarange("data", len(data)):
    pyro.sample(
        "obs",
        dist.Bernoulli(p).expand_by(data.shape).independent(1),
        obs=data
    )
`)
The most important lines are the last two, which specify that every element of
this data vector is from a Bernoulli with probability ${mtex`p`} drawn from the
prior. The `iarange` wrapper makes sure operations can be done in parallel.

Suppose we have observed 30 draws from the coin, and we observe 20 ones and 10
zeros. We know in this case that the posterior is computable by integration (in
this case it's ${mtex`\Bet\left(30, 20\right)`}), but as a prelude to the more
complex situations to come, where the posterior isn't available in closed form,
what can we do?

The usual solution is to find an approximation to the posterior. In pyro, this
is referred to as using "guides." The guides define a family of candidate
posteriors, and it's the language's job to find the member of the family that
best represents the truth. In this example, we let the guides be the family of
all possible beta posteriors. This is a natural choice, since the *true*
posterior lies within this family, so if the search procedure is effective, it
should be possible to find the true posterior.

The guide is specified in the following function. The first part defines the
parameters of the approximation distribution, which we will need to search over.
The last line is what tells pyro that the approximation distribution will be a
beta.

sql(`
def guide(data):
    # tell pyro what the parameters to search over are
    alpha_q = pyro.param(
        "alpha_q",
        torch.tensor(15.0), # initialization value
        constraint=constraints.positive
    )
    beta_q = pyro.param(
        "beta_q",
        torch.tensor(15.0),
        constraint=constraints.positive
    )

    # define approximation distribution
    pyro.sample("heads_prob", dist.Beta(alpha_q, beta_q))
`)

We'll skip over the details of how the search is done.


But at the end of the day, the package identifies a ${mtex`\Beta\left(30.093, 15.415\right)`}
as the best approximation to the true posterior. This is a bit off from the
truth, even if all you cared about was the posterior mean (${mtex`\frac{20 + 10}{30 +
20}`}). But perhaps this is the price to pay for getting an answer immediately,
without doing any calculations by hand.

The complete code for this experiment is given in [this
script](https://github.com/krisrs1128/naami_tutorials/blob/master/notebooks/betabinomial.py).

## Latent Variables


This is maybe easiest to understand with an example. Consider a Gaussian
mixture model, where the data ${mtex`x_i`} are assumed to have been drawn from
the following process,

mtex_block`
\begin{aligned}
\mu_{k} &\sim \Gsn\left(0, \Sigma_{0}\right) \\
z_i &\sim \Cat\left(z_i \vert p\right) \\
x_i \vert z_i = k &\sim \Gsn\left(x_i \vert \mu_{k}, \Sigma_k\right)
\end{aligned}`

where ${mtex`p`} is some probability over ${mtex`K`} categories.

That is, we have ${mtex`K`} underlying Gaussians, each with a different mean
${mtex`\mu_k`} and covariance ${mtex`\Sigma_k`}. The different means are drawn
from some common prior. When we sample a new datapoint ${mtex`x_i`}, we first
pick some index between 1 and ${mtex`K`} at random, according to the
probabilities ${mtex`p`}, and then we draw from the Gaussian distribution given
by the index that we've just picked from.

Here's an example of the kind of histogram you would observe for a
one-dimensional version, where ${mtex`K = 3`}. There is a narrow gaussian near
the center, and two wide ones on either end.

If we knew which of the underlying gaussians any new data point belonged to
(that is, if we knew it's ${mtex`z_i`}), we would be able to write down its
exact (gaussian) density. The "complete" loglikelihood when we observe both
${mtex`x_i`} and ${mtex`z_i`} for each sample has the following form,

mtex_block`
\begin{aligned}
\log p\left(x, z\right) &= \log \left[\prod_{i = 1}^{n} \prod_{k = 1}^{K} \left(p_{k}\Gsn\left(x_i \vert \mu_k, \Sigma_k\right)^{\indic{z_i = k}}\right)\right]
&= \sum_{i = 1}^{n} \sum_{k = 1}^{K} \indic{z_i = k} \log p_{k} \log \Gsn\left(x_i \vert \mu_k, \Sigma_k\right)
&= \sum_{i = 1}^{n} \sum_{k = 1}^{K} \indic{z_i = k} \log p_{k} \left[-\frac{D}{2}\log\left(2\pi) - \log \left|\Sigma_k\right| - \frac{1}{2} \left(x_i - \mu_k\right)^{T}\Sigma^{-1}_{k} \left(x_i - \mu_k)
\end{aligned}
`

While this might look a little messy to those who aren't battled-scarred by
statistics, it's actually pretty nice, because you can run all these operations
in a computer and get probabilities for any configurations of the ${mtex`x_i`}
and ${mtex`z_i`}'s that you are interested in.

Unfortunately, we often don't have direct access to the underlying
${mtex`z_i`}'s which generated the process, and this complicates the likelihood
evaluation. The reason is that we need to sum the above expression over all
possible configurations of the underlying ${mtex`z_i`}'s,

mtex_block`
\log p\left(x\right) &= \log \sum_{z} p\left(x, z\right) dz
`
so we can't go from steps two to three in the original simplification for the
complete loglikelihood: the summation prevents from turning the log of the
product into the sum of logs. Even worse news -- while we could evaluate the
(nonlogged) likelihood for any particular ${mtex`z_i, x_i`} configuration,
summing over all ${mtex`n^{K}`} possible configurations of the ${mtex`z_{i}`}'s
becomes completely untenable for even moderate values of ${mtex`n`} and
${mtex`K`}.

Note that this also prevents us from computing the posterior ${mtex`p\left(z
\vert x\right)`}, because the intractable summation appears in the denominator
in Bayes' rule,

mtex_block`
\begin{aligned}
p\left(z \vert x\right) &= \frac{p\left(x \vert z\right)p\left(z\right)}{p\left(x\right)} \\
&= \frac{p\left(x \vert z\right)p\left(z\right)}{\sum_{z} p\left(x, z\right)} \\
\end{aligned}

## The Evidence Lower Bound

Somewhat counterintuitively, we're going to approach this problem by introducing
additional complexity. But viewed another way, this complexity creates
additional degrees of freedom: sometimes when you get stuck at the end of a
blind alley, it helps to open as many door as possible.

The idea is to consider new density ${mtex`q\left(z \vert x\right)`}, which lies
in some large (but hopefully not intractable) family ${mtex`Q`} (in our previous
example, ${mtex`Q`} was the family of beta distributions). Note that while I'm
writing the density ${mtex`q`} as conditional on ${mtex`x`}, this is not a
requirement -- we could simply ignore ${mtex`x`} when coming up with a density
${mtex`q`} over ${mtex`z`}.


Now, since ${mtex`p\left(x\right)`} doesn't involve ${mtex`z`}, and ${mtex`q`}
is a density over ${mtex`z`}'s, we can trivially write ${mtex`\log
p\left(x\right) = \Esubarg{q}{\log p\left(x\right)}`}. Then, rearranging Bayes'
rule in the second line and multiplying and dividing by ${mtex`q`} in the next
line,
we obtain the (more complex, but flexible) expression,

mtex_block`
\begin{aligned}
\log p\left(x\right) &= \Esubarg{q}{\log p\left(x\right)} \\
&= \Esubarg{q}{\log \frac{p\left(x, z\right)}{p\left(z \vert x\right)}} \\
\Esubarg{q}{\log \frac{p\left(x, z\right)}{q\left(z \vert x\right)} \frac{q\left(z \vert x\right)}{p\left(z \vert x\right)}} \\
&= \Esubarg{q}{\log p\left(x \vert z\right)} - D_{KL}\left(q\left(z \vert x\right) \vert \vert p\left(z\right)\right) + D_{KL}\left(q\left(z \vert x\right) \vert \vert p\left(z \vert x\right)\right)
\end{aligned}
`

This is an equality worth studying in details. Think of it by mentally varying
${mtex`q`}, some distribution of the latent variables given the observations
${mtex`x`} -- in this way, it's like a posterior, though note that it can be any
density in ${mtex`Q`}.

The first term is often called a "reconstruction error". It measures how
probable the observed ${mtex`x`} would be under different configurations of
${mtex`z`}, after we average over many draws of ${mtex`z`} from ${mtex`q`}.
For example, in the mixture of gaussians case, if we had a distribution ${`q`}
that made sure ${mtex`z`}'s clustered according to the mixture peaks, then the
reconstruction error would be much better then if we were guessing latent
mixture assignments at random.

The second term is a measure of the discrepancy between the posterior-like
${mtex`q`} and the prior over the latent variables. You can think of it like a
regularization term -- it is large when ${mtex`q`} is far from the prior, which
usually means it's getting more complicated.

The last term formalizes what we mean when we say ${mtex`q`} is
"posterior-like." When ${mtex`q`} is exactly the posterior, then this term is
zero.

The evidence lower bound ("ELBO," evidence is just a fancy way of referring to
${mtex`\log p\left(x\right)`}) is the inequality that emerges when you remember
that the KL-divergence is always larger than zero. Specifically, when we drop
the last KL term, we observe

mtex_block`
\log p\left(x\right) \geq \Esubarg{q}{\log p\left(x \vert z\right)} - D_{KL}\left(q\left(z \vert x\right) \vert \vert p\left(z\right)\right)
`

The reason we might begin to care about something like this is given in the next
section. The main point to keep in mind though is that we can express the
loglikehood over observed data, ${mtex`\log p\left(x\right)`} (an almost
puritanically simple expression), in terms of expectations and divergences
involving latent variables distributed according to arbitrary ${mtex`q`}'s
(which though more complex, affords us an amazing amount of flexibility).

## Variational Inference

The basic idea of variational inference is that posterior inference can be
framed as an optimization problem. That is, if we had a good way of searching
through the space of probability densities and measuring their ability to true
the true posterior, we could try to find a reasonably good approximate
posterior, even when the true posterior can't be written in closed form.

If this idea feels like deja vu, it's because we've already seen it, during our
discussion of \`guides\` in pyro. There, we parameterized a family of densities,
through the \`guide\` function, and through magic, we obtained a setting of the
guide that was reasonably close to the -- in our case analytically known -- true
posterior. What seemed like magic was actually (stochastic) variational
inference.

To practically implement the variational inference idea, we need to figure out
two things,

  1. What densities ${mtex`Q`} can we actually work with?
  2. How are we going to measure the quality of an approximation?

The first question is usually solved on a case-by-case basis, though there are a
few generic recipes, like the mean-field or structured-mean field
approximations. We'll see some examples of how approximating families are chosen
below.

For the second question, there is a neat answer in terms of the ELBO that we
derived above. The idea is that a choice of ${mtex`q`} that makes the lower
bound on the evidence tight is probably a good one, indeed from the original
equality, we see that the size of the gap between ${\log p \left(x\right)} and
the ELBO is exactly

mtex_block`
D_{KL}\left(q\left(z \vert x\right) \vert \vert p\left(z \vert x\right)\right)
`

which is the distance between the approximation and the true posterior,
according to the KL-divergence.

There is some interesting research that attempts to use different measures (the
KL divergence in particular is notorious for issues like zero-forcing, which has
implications on variance estimates), but for now let's stick to this
ELBO-based measure.

It turns out that if you choose to use a mean-field family for point (1) and the
ELBO for (2), then it's easy to write down an explicit optimization procedure to
find (a locally optimal) ${mtex`q`}, from the potentially large initial family
${mtex`Q`}. The strategy is outlined below, but it has a bit more math than the
rest of the sections, and it's fine if you choose to skip it.

### Mean-Field Variational Inference

The premise of mean-field variational inference is that assuming independence
among all coordinates in the approximation ${mtex`q`} makes computation so much
more straightforwards that it's worth any decrease in approximation quality.
That is, if we have ${mtex`n`} latent variables ${mtex`z`}, then instead of
considering arbitrary densities ${mtex`q\left(z_1, \dots, z_n\right)`}, we'll
only search over those of the form

mtex_block`
q\left(z_1, \dots, z_n\right) = \prod_{i = 1}^{n} q_{i}\left(z_i\right).
`

Geometrically, it means that in some ${mtex`n`}-dimensional space, we're taking
the product of axis-aligned densities (tilted ellipses are ruled out, for
example).

Notice that, compared to our derivation of the ELBO, we've dropped all
dependence on ${mtex`x_i`}. While the notation is more concise, realize that
this is actually a larger class of densities -- the set of densities over
${mtex`z`} is larger than the set of conditional densities over ${mtex`z`},
conditional on ${mtex`x`}. So the derivation we provide here is the most general
possible, given the factorization assumption.

The nice thing about this assumption is that it lets you derive a coordinate
ascent algorithm in closed form. Note that this makes no other assumption on the
form of ${mtex`q`} (for example, we're not requiring each component to be in an
exponential family). Specifically, we fix all the densities except for
the $i^{th}$ one, and then task ourselves with finding the $q_{i}$ that
maximizes the ELBO. It turns out that the optimal choice, among all possible
one-dimensional densities, is given by

mtex_block`
q\left(z_i\right) \propto \exp{\Esubarg{q_-i}{\log p\left(x, z_i, z_{-i}\right)}}
`

where ${mtex`q_{-i}`} refers to the product density over all but the
${mtex`i^{th}`} latent variable (similarly ${mtex`z_{-i}` refers to the vector
of all but ${mtex`z_i`}}). Just as a sanity check, note that this is actually a
density over ${mtex`z_i`} -- all the other latent variables get integrated out
by the expectation. Second, if it weren't for the expectation lying in the
middle, the exp and log would have cancelled out. So informally, this is like
a joint density after averaging over all but the ${mtex`i^{th}`} term -- in this
way, it has a flavor of gibbs sampling.

The derivation of this update rule actually isn't so complicated. The approach
is to rewrite the ELBO after ignoring all terms that are constant in
${mtex`q_i`}. First, we'll rewrite our earlier ELBO into an equivalent
expression, which is a more transparent function of only expectations, rather
than a mix of expectations and KLs. [We're also replacing ${mtex`q\left(z \vert
x\right)}` by the more general ${mtex`q\left(z\right)`}],

mtex_block`
\begin{aligned}
\log p\left(x\right) \geq
\Esubarg{q}{\log p\left(x, \vert z\right)} - D_{KL}\left( q\left(z\right) \vert p\left(z\right)\right) \\
&= \Esubarg{q}{\log \frac{p\left(x, z\right)}{p\left(z\right)}} - \Esubarg{q}{\log \frac{q\left(z\right)}{p\left(z\right)}} \\
&= \Esubarg{q}{\log p\left(x, z\right)} - \Esubarg{q}{\log q\left(z\right)}
\end{aligned}
`

As an aside: This expression has a nice interpretation, just like the previous
ELBO terms. The first expression is a variant of the expected reconstruction
error, but considering configurations of ${mtex`x_i`} and ${mtex`z_i`} jointly.
The second expression is the entropy of ${mtex`q`}, which serves as a sort of
regularizer. So the idea is to maximize the reconstruction likelihood while
having as much entropy in ${mtex`q`} as possible.

Back to the derivation. Let's consider the terms of this expression that are
functions of ${mtex`q_i`}, and remove additive constants (using the notation
${mtex`\over{c}{=}`} to refer to equality up to constants),

mtex_block`
\begin{aligned}
\log p\left(x\right) &\geq \Esubarg{q}{\log p\left(x, z\right)} - \Esubarg{q}{\log q\left(z\right)} \\
&= \Esubarg{q_i}{\Esubarg{q_{-i} \vert i}{\log p\left(x, z_i, z_{-i})}} - \sum \Esubarg{q_j}{\log q_j\left(z_j\right)}
\end{aligned}
`
Of course, to maximize a negative KL, we can set it to zero, since the smallest
a KL divergence can be is zero. To do this, just set ${mtex`q_i`} to the
expression on the right-hand side, which is indeed the update density we
specified before. (you might have noticed the slight of hand in writing the
density on the right without the normalizing constant... I realize I've done
that, but thought the notation was getting messy enough as it is.)

Stepping back, this seems pretty nice. As long as we can evaluate the
expectation in the update formula (which we usually can, since we are looking at
a joint over ${mtex`\left(x, z\right)`} and not an intractable posterior), we
have a relatively automatic way of improving our current joint
${mtex`q\left(z_1, \dots, z_n\right) = \prod_i q_i\left(z_i\right)`}, increasing
our lower bound by updating one ${mtex`q_i`} at a time.

## Stochastic Variational Inference

There is a form of variational inference that appears a lot in practice, which
we haven't talked about yet, but which si important in understanding

## Variational Autoencoders

## Miscellaneous References

I don't blame you for wanting to learn more about this subject, here is a random
collection of interesting discussions aligned with what we talked about here,

1. You may have heard that Bayesian Deep Learning is (or isn't) the [most
   brilliant thing ever](https://www.youtube.com/watch?v=HumFmLu3CJ8).
2.
