---
layout: post
title: 'Deconstructing PyMC3 Part II'
date: 2019-11-22 14:16 +0530
categories: pymc3 deconstruction
author: suriya
---

## Random Variables and Prior Distributions

In PyMC3, model are defined in terms of random variables. The random variables can be observed variables (observations are available) or unobserved variables, with prior distributions associated with them. They should always be defined within the context of a model. 

In the example below, we add an unobserved random variable `x`, with a normal prior distribution parameterized by mean `mu` and standard deviation `sigma`.

{% highlight python %}
mu, sigma = 2, 10.
with Model() as model:
  # model definition
  x = Normal('x', mu, sd=sigma)
{% endhighlight %}


<img src="/images/posts/pymc3/02-distributions/dist.png" class="align-left" alt="Instance Creation" width="170"/>

In this blog post, we'll examine the process involved in creating a random variable and adding it to the model. Lets fire up `ipdb`.


The call to `Normal` takes us to the (grand)parent of `Normal` class in `pymc3/distributions/distribution.py`.  `Distribution.__new__` is called. This method creates an instance of `Normal` class and attaches it to a random variable and returns the same.

<br />


{% highlight python %}
class Distribution:
  """Statistical distribution"""
  def __new__(cls, name, *args, **kwargs):
    ...
    model = Model.get_context()  # :: get model from context
    ...                          # :: throw exception if not in context
    data = kwargs.pop('observed', None)  # :: see if there's data associated
    cls.data = data  # :: attach it to class
    total_size = kwargs.pop('total_size', None)  # :: get dim of distribution
    dist = cls.dist(*args, **kwargs)  # <- creates an instance of Normal
    return model.Var(name, dist, data, total_size)  # :: model.Var creates an RV
{% endhighlight %}

The class method `dist` creates an instance of `Normal`. The call  `__init__` triggers `Normal.__init__`. 

{% highlight python %}
@classmethod
def dist(cls, *args, **kwargs):
  dist = object.__new__(cls)
  dist.__init__(*args, **kwargs)
  return dist
{% endhighlight %}

`Normal.__init__` manages parameters of normal distribution `mu` and `sigma`. Different forms of sigma - `tau` and `variance` are created. We check if sigma has negative support. Since we don't define sigma as a distribution, this check doesn't make sense. Then, we place a call to the parent `Continuous.__init__`. 

{% highlight python %}
class Normal(Continuous):

  def __init__(self, mu=0, sigma=None, tau=None, sd=None, **kwargs):
    if sd is not None:
      sigma = sd
    tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)  # :: get tau from sigma
    self.sigma = self.sd = tt.as_tensor_variable(sigma)  # :: set sigma
    self.tau = tt.as_tensor_variable(tau)  # make everything a theano tensor variable
    mu = tt.as_tensor_variable(floatX(mu))
    self.mean = self.median = self.mode = self.mu = mu
    self.variance = 1. / self.tau  # :: sd^2

    assert_negative_support(sigma, 'sigma', 'Normal')  # :: check if negative support
    assert_negative_support(tau, 'tau', 'Normal')      # ::  exists 
    # as sigma is not a distribution in our case, there will be no support
    super().__init__(**kwargs)  # <- Call to Continuous.__init__
{% endhighlight %}

Not much happens in `Continuous.__init__`.  In fact, the `Continuous` class has no more methods. I believe this is purely an aesthetic choice, to explicitly model the Continuous/Discrete duality. `Distribution.__init__` is called in the end.

{% highlight python %}
class Continuous(Distribution):
  """Base class for continuous distributions"""
  def __init__(self, shape=(), dtype=None, defaults=('median', 'mean', 'mode'),
               *args, **kwargs):
    if dtype is None:
      dtype = theano.config.floatX
    super().__init__(shape, dtype, defaults=defaults, *args, **kwargs)
{% endhighlight %}

`Distribution.__init__` sets the `shape`, `dtype` and `type` (`TensorType`) of `Normal` instance. We don't know where `defaults = ('mean', 'median', 'mode')` come into play. 

{% highlight python %}
class Distribution:

  def __init__(self, shape, dtype, testval=None, defaults=(),
      transform=None, broadcastable=None):
    self.shape = np.atleast_1d(shape)  # :: shape = array([]) <- ()
		...
    self.dtype = dtype
    self.type = TensorType(self.dtype, self.shape, broadcastable)
    self.testval = testval      # :: What is the role of `testval`
    self.defaults = defaults    # :: ( 'mean', 'median', 'mode' )
    self.transform = transform  # :: What is a transform? Must dig into this later!
{% endhighlight %}

This ends the creation of `Normal` instance. `Distribution.dist` returns the instance. `__new__` creates a free random variable by calling `model.Var` method of `Model`. 

{% highlight python %}
class Distribution:
  """Statistical distribution"""
  def __new__(cls, name, *args, **kwargs):
		...
    dist = cls.dist(*args, **kwargs)  # :: method creates an instance of `Normal`
    return model.Var(name, dist, data, total_size)  # <- creates and returns a FreeRV
{% endhighlight %}

If there isn't any data associated, we create a FreeRV with prior distribution `dist` (`Normal`). We add it to the list of free variables in the model. We also add the random variable as an attribute of the model so we can access it as an attribute, like `model.x`. 

{% highlight python %}
def Var(self, name, dist, data=None, total_size=None):
  """Create and add (un)observed random variable to the model with an
  appropriate prior distribution.
  """
  ...
  if data is None:
    if getattr(dist, "transform", None) is None:      # :: What is a transform?
      with self:  # :: create FreeRV in model context
        var = FreeRV(name=name, distribution=dist,    # <- call to FreeRV.__init__
                     total_size=total_size, model=self)
	...
  self.free_RVs.append(var)      # :: add to the list of free variables
  self.add_random_variable(var)  # :: adds random variable as an attribute of model
  return var                     # ::  so we can access it by name. eg : `model.x`
{% endhighlight %}

{% highlight python %}
def add_random_variable(self, var):
  """Add a random variable to the named variables of the model."""
  if self.named_vars.tree_contains(var.name):
    raise ValueError("Variable name {} already exists.".format(var.name))
  self.named_vars[var.name] = var  # :: add to a list of named variables
  if not hasattr(self, self.name_of(var.name)):  
    setattr(self, self.name_of(var.name), var)   # add as an attribute
{% endhighlight %}

Now lets take a look at the `FreeRV` class. `FreeRV` inherits `Factor` and `PyMC3Variable`. `Factor` contains "*Common functionality for objects with a log probability density associated with them*". `PyMC3Variable` is a wrapper over theano's `TensorVariable` class. This means a `FreeRV` instance can behave like a tensor and support tensor operations as well as support distribution-like behaviour, i.e., we could sample from it. We copy the attributes of the normal distribution instance into  `FreeRV` - size, shape, etc,. `distribution.logp` creates a theano expression for `logp` which could be evaluated for any value.

{% highlight python %}
class FreeRV(Factor, PyMC3Variable):
  """Unobserved random variable that a model is specified in terms of."""

  def __init__(self, name=None, distribution=None, total_size=None,
               model=None, ...):
    if type is None:  # :: set type of distribution as type of FreeRV
      type = distribution.type  # :: TensorType
    super().__init__(type, owner, index, name)  # Call to Factor -> TensorVariable

    if distribution is not None:
      self.dshape = tuple(distribution.shape)
      self.dsize = int(np.prod(distribution.shape))
      self.distribution = distribution
      self.tag.test_value = np.ones(  # :: no idea what's happening here or why!
        distribution.shape, distribution.dtype) * distribution.default()  # :: TODO
      self.logp_elemwiset = distribution.logp(self)
      self.total_size = total_size
      self.model = model
      # scale = 1. if total_size is None <- scalar variable
      self.scaling = _get_scaling(total_size, self.shape, self.ndim)
      incorporate_methods(source=distribution, destination=self,  # copy the method
                          methods=['random'],                     # `random` to the
                          wrapper=InstanceMethod)                 # FreeRV
      # so we could sample from FreeRV
{% endhighlight %}

{% highlight python %}
def _get_scaling(total_size, shape, ndim):
  """Gets scaling constant for logp"""
  if total_size is None:
    coef = floatX(1)
  ...
  return tt.as_tensor(floatX(coef))
{% endhighlight %}

`logp` calculation happens in `Normal.logp` method. `bound` method in `pymc3/distributions/dist_math.py`, evaluates the expression given as first argument and returns the resulting theano expression if the condition presented as second argument is satisfied. If not it returns negative infinity.

{% highlight python %}
def logp(self, value):
  """Calculate log-probability of Normal distribution at specified value."""
	...
  return bound((-tau * (value - mu)**2 + tt.log(tau / np.pi / 2.)) / 2.,
               sigma > 0)
{% endhighlight %}

{% highlight python %}
def bound(logp, *conditions, **kwargs):
  """Bounds a log probability density with several conditions."""
	...
  return tt.switch(alltrue(conditions), logp, -np.inf)
{% endhighlight %}

## Reconstruction

I've created a [github repo](https://github.com/suriyadeepan/deconstructing-pymc3) to maintain the reconstructed code. Check out the [code](https://github.com/suriyadeepan/deconstructing-pymc3/tree/master/reconstruction/02-distributions) constructed from our examination in this blog post.

In summary, we have looked at how PyMC3 creates and manages random variables of a model. We have learned that random variables (`FreeRV`) are associated with prior distributions and they can exhibit distribution-like behaviour and theano tensor-like behaviour. In the next blog, we'll find out how random sampling is implemented in distributions and how we can evaluate (`logp`) arbitrary samples under a given distribution.
