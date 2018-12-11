
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
  C. HThe value of deep learning is that it attempts to automatically learn these sorts of
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


# Some History

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
the figure below, generated by[this
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

## Miscellaneous References

I don't blame you for wanting to learn more about this subject, here is a random
collection of interesting discussions aligned with what we talked about here,

1. You may have heard that Bayesian Deep Learning is (or isn't) the [most
   brilliant thing ever](https://www.youtube.com/watch?v=HumFmLu3CJ8).
2.
