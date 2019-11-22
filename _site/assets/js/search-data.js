var store = [{
        "title": "Deconstructing PyMC3 : Part I",
        "excerpt":"  Models in PyMC3 are centered around the Model class. It has references to all random variables (RVs) and computes the model logp and its gradients. Usually, you would instantiate it as part of a with context, as follows: with Model() as model:  # model definition  pass What happens behind the scenes while line #1 is executed is all we care about in this blog post. This takes us to pymc3/model.py. The class Model extends the classes Context, Factor, WithMemoization and is instantiated via the metaclass InitContextMeta.  When line #1 is executed, the metaclass InitContextMeta’s __call__ method is called which executes __init__ method of Model’s instance in its context. The metaclass consists only of the __call__ method.  class InitContextMeta(type):    def __call__(cls, *args, **kwargs):    instance = cls.__new__(cls, *args, **kwargs)  # &lt;&lt;    ...cls points to Model class. Model.__new__ is called which creates an instance of Model. After which the parent instance is resolved and stored at self._parent. This feature exists to support Model nesting.  class Model(Context, Factor, WithMemoization, metaclass=InitContextMeta):\t  def __new__(cls, *args, **kwargs):        # resolves the parent instance        instance = super().__new__(cls)        if kwargs.get('model') is not None:  # parent explicitly set by user            instance._parent = kwargs.get('model')        elif cls.get_contexts():  # &lt;&lt;            instance._parent = cls.get_contexts()[-1]        else:            instance._parent = None The context manager syntax of Model is derived from Context class. Let us take a look. class Context:    contexts = threading.local()  # :: contexts is a thread-local object                                  # :: items stored will be hidden from other      \t\t\t\t\t\t\t\t\t\t\t\t\t\t# :: threads ::    def __enter__(self):        type(self).get_contexts().append(self) # :: add instance to context stack ::\t\t\t\t...        return self    def __exit__(self, typ, value, traceback):        type(self).get_contexts().pop()  #  :: remove instance from context stack ::\t\t\t\t...            @classmethod    def get_contexts(cls):  # :: return the thread-local stack ::        if not hasattr(cls.contexts, 'stack'):            cls.contexts.stack = []  # :: create a stack if one isn't available ::        return cls.contexts.stack    @classmethod    def get_context(cls):\t\t\t...      return cls.get_contexts()[-1]  # :: return the deepest context in the stack ::Notice that get_contexts is a @classmethod. Class methods take cls as their first argument. Since Model is a child of Context, Context.get_contexts is called, which returns an empty stack.  class Model(Context, Factor, WithMemoization, metaclass=InitContextMeta):\t  def __new__(cls, *args, **kwargs):        # resolves the parent instance        instance = super().__new__(cls)        if kwargs.get('model') is not None:            instance._parent = kwargs.get('model')        elif cls.get_contexts():            instance._parent = cls.get_contexts()[-1]        else:            instance._parent = None  # &lt;&lt; parent instance is `None`        ...        return instanceWe are back to InitContextMeta.__call__ method. class InitContextMeta(type):    def __call__(cls, *args, **kwargs):        instance = cls.__new__(cls, *args, **kwargs)        with instance:  # &lt;&lt; appends context            instance.__init__(*args, **kwargs)        return instancewith instance: invokes Context.__enter__ which simply adds the instance to context stack. Note that type(self) resolves to Model class. def __enter__(self):  type(self).get_contexts().append(self)  # :: add instance to thread-local stack ::  return selfModel.__init__ method is called within context of instance which creates a bunch of empty variables. self.named_vars = treedict()self.free_RVs = treelist()self.observed_RVs = treelist()self.deterministics = treelist()self.potentials = treelist()self.missing_values = treelist()Note that if there exists a parent instance self._parent, the parent’s variables are added to the instance’s variables as parents. self.named_vars = treedict(parent=self.parent.named_vars)...We exit the context with instance: invoking the __exit__ method which removes the instance from contexts stack.  This takes us back to InitContextMeta.__call__ where the Model instance is finally returned. with Model() as model:  ...The use of context manager syntax here takes us again to Context.__enter__.  Model instance is added to context stack. Now we have an empty container for a model. Right now we don’t care about the details of data structures treelist and treedict. Reconstruction Let us reconstruct the Model class with the information available so far. import threadingclass Context:  contexts = threading.local()  # thread-local storage  def __enter__(self):    cls = type(self)  # get a handle to Class `Context`    contexts = cls.get_contexts()  # call @classmethod `get_contexts`    contexts.append(self)  # add instance to contexts    return self  def __exit__(self, typ, value, traceback):    cls = type(self)  # get a handle to Class `Context`    contexts = cls.get_contexts()  # call @classmethod `get_contexts`    contexts.pop()  # remove instance from contexts stack  @classmethod  def get_contexts(cls):    if not hasattr(cls.contexts, 'stack'):  # does `Context.contexts.stack` exist?      cls.contexts.stack = []  # create and return an empty stack    return cls.contexts.stack  @classmethod  def get_context(cls):    contexts = cls.get_contexts()  # get all contexts    if len(contexts) == 0:      raise Exception(\"Context stack is empty!\")    return contexts[-1]  # return the deepest contextclass InitContextMeta(type):  \"\"\"Metaclass that runs Model.__init__ in its context\"\"\"  def __call__(cls, *args, **kwargs):    instance = cls.__new__(cls, *args, **kwargs)  # create an instance of `Model`    with instance:  # run __init__ in context      instance.__init__(*args, **kwargs)  # `Model.__init__`    return instanceclass Model(Context, metaclass=InitContextMeta):  def __new__(cls, *args, **kwargs):  # class method that creates an instance    instance = super().__new__(cls)    # resolve parent instance    if cls.get_contexts():  # if contexts stack isn't empty      instance._parent = cls.get_context()  # get the deepest context    else:      instance._parent = None    return instance  def __init__(self, name=''):    self.name = name    if self.parent is None:      self.named_vars = {}  # using python dict instead of pymc's treedict      self.free_RVs = []    # using python list instead of pymc's treelist      self.observed_RVs = []      self.deterministics = []      self.potentials = []      self.missing_values = []    else:      raise NotImplementedError('We dont care about this case yet!')  @property  def parent(self):    return self._parent    if __name__ == '__main__':  with Model() as model:    passWe were able to build an empty model. In the next post, we’ll deconstruct the process behind Model Definition -  creating and managing random variables within the context of the model. ","categories": ["pymc3","deconstruction"],
        "tags": [],
        "url": "http://localhost:4000/pymc3/deconstruction/2019/11/07/deconstructing-pymc3-part-i.html"
      },{
        "title": "Deconstructing PyMC3 Part II",
        "excerpt":"Random Variables and Prior Distributions In PyMC3, model are defined in terms of random variables. The random variables can be observed variables (observations are available) or unobserved variables, with prior distributions associated with them. They should always be defined within the context of a model. In the example below, we add an unobserved random variable x, with a normal prior distribution parameterized by mean mu and standard deviation sigma. mu, sigma = 2, 10.with Model() as model:  # model definition  x = Normal('x', mu, sd=sigma) In this blog post, we’ll examine the process involved in creating a random variable and adding it to the model. Lets fire up ipdb. The call to Normal takes us to the (grand)parent of Normal class in pymc3/distributions/distribution.py.  Distribution.__new__ is called. This method creates an instance of Normal class and attaches it to a random variable and returns the same.  class Distribution:  \"\"\"Statistical distribution\"\"\"  def __new__(cls, name, *args, **kwargs):    ...    model = Model.get_context()  # :: get model from context    ...                          # :: throw exception if not in context    data = kwargs.pop('observed', None)  # :: see if there's data associated    cls.data = data  # :: attach it to class    total_size = kwargs.pop('total_size', None)  # :: get dim of distribution    dist = cls.dist(*args, **kwargs)  # &lt;- creates an instance of Normal    return model.Var(name, dist, data, total_size)  # :: model.Var creates an RVThe class method dist creates an instance of Normal. The call  __init__ triggers Normal.__init__. @classmethoddef dist(cls, *args, **kwargs):  dist = object.__new__(cls)  dist.__init__(*args, **kwargs)  return distNormal.__init__ manages parameters of normal distribution mu and sigma. Different forms of sigma - tau and variance are created. We check if sigma has negative support. Since we don’t define sigma as a distribution, this check doesn’t make sense. Then, we place a call to the parent Continuous.__init__. class Normal(Continuous):  def __init__(self, mu=0, sigma=None, tau=None, sd=None, **kwargs):    if sd is not None:      sigma = sd    tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)  # :: get tau from sigma    self.sigma = self.sd = tt.as_tensor_variable(sigma)  # :: set sigma    self.tau = tt.as_tensor_variable(tau)  # make everything a theano tensor variable    mu = tt.as_tensor_variable(floatX(mu))    self.mean = self.median = self.mode = self.mu = mu    self.variance = 1. / self.tau  # :: sd^2    assert_negative_support(sigma, 'sigma', 'Normal')  # :: check if negative support    assert_negative_support(tau, 'tau', 'Normal')      # ::  exists     # as sigma is not a distribution in our case, there will be no support    super().__init__(**kwargs)  # &lt;- Call to Continuous.__init__Not much happens in Continuous.__init__.  In fact, the Continuous class has no more methods. I believe this is purely an aesthetic choice, to explicitly model the Continuous/Discrete duality. Distribution.__init__ is called in the end. class Continuous(Distribution):  \"\"\"Base class for continuous distributions\"\"\"  def __init__(self, shape=(), dtype=None, defaults=('median', 'mean', 'mode'),               *args, **kwargs):    if dtype is None:      dtype = theano.config.floatX    super().__init__(shape, dtype, defaults=defaults, *args, **kwargs)Distribution.__init__ sets the shape, dtype and type (TensorType) of Normal instance. We don’t know where defaults = ('mean', 'median', 'mode') come into play. class Distribution:  def __init__(self, shape, dtype, testval=None, defaults=(),      transform=None, broadcastable=None):    self.shape = np.atleast_1d(shape)  # :: shape = array([]) &lt;- ()\t\t...    self.dtype = dtype    self.type = TensorType(self.dtype, self.shape, broadcastable)    self.testval = testval      # :: What is the role of `testval`    self.defaults = defaults    # :: ( 'mean', 'median', 'mode' )    self.transform = transform  # :: What is a transform? Must dig into this later!This ends the creation of Normal instance. Distribution.dist returns the instance. __new__ creates a free random variable by calling model.Var method of Model. class Distribution:  \"\"\"Statistical distribution\"\"\"  def __new__(cls, name, *args, **kwargs):\t\t...    dist = cls.dist(*args, **kwargs)  # :: method creates an instance of `Normal`    return model.Var(name, dist, data, total_size)  # &lt;- creates and returns a FreeRVIf there isn’t any data associated, we create a FreeRV with prior distribution dist (Normal). We add it to the list of free variables in the model. We also add the random variable as an attribute of the model so we can access it as an attribute, like model.x. def Var(self, name, dist, data=None, total_size=None):  \"\"\"Create and add (un)observed random variable to the model with an  appropriate prior distribution.  \"\"\"  ...  if data is None:    if getattr(dist, \"transform\", None) is None:      # :: What is a transform?      with self:  # :: create FreeRV in model context        var = FreeRV(name=name, distribution=dist,    # &lt;- call to FreeRV.__init__                     total_size=total_size, model=self)\t...  self.free_RVs.append(var)      # :: add to the list of free variables  self.add_random_variable(var)  # :: adds random variable as an attribute of model  return var                     # ::  so we can access it by name. eg : `model.x`def add_random_variable(self, var):  \"\"\"Add a random variable to the named variables of the model.\"\"\"  if self.named_vars.tree_contains(var.name):    raise ValueError(\"Variable name {} already exists.\".format(var.name))  self.named_vars[var.name] = var  # :: add to a list of named variables  if not hasattr(self, self.name_of(var.name)):      setattr(self, self.name_of(var.name), var)   # add as an attributeNow lets take a look at the FreeRV class. FreeRV inherits Factor and PyMC3Variable. Factor contains “Common functionality for objects with a log probability density associated with them”. PyMC3Variable is a wrapper over theano’s TensorVariable class. This means a FreeRV instance can behave like a tensor and support tensor operations as well as support distribution-like behaviour, i.e., we could sample from it. We copy the attributes of the normal distribution instance into  FreeRV - size, shape, etc,. distribution.logp creates a theano expression for logp which could be evaluated for any value. class FreeRV(Factor, PyMC3Variable):  \"\"\"Unobserved random variable that a model is specified in terms of.\"\"\"  def __init__(self, name=None, distribution=None, total_size=None,               model=None, ...):    if type is None:  # :: set type of distribution as type of FreeRV      type = distribution.type  # :: TensorType    super().__init__(type, owner, index, name)  # Call to Factor -&gt; TensorVariable    if distribution is not None:      self.dshape = tuple(distribution.shape)      self.dsize = int(np.prod(distribution.shape))      self.distribution = distribution      self.tag.test_value = np.ones(  # :: no idea what's happening here or why!        distribution.shape, distribution.dtype) * distribution.default()  # :: TODO      self.logp_elemwiset = distribution.logp(self)      self.total_size = total_size      self.model = model      # scale = 1. if total_size is None &lt;- scalar variable      self.scaling = _get_scaling(total_size, self.shape, self.ndim)      incorporate_methods(source=distribution, destination=self,  # copy the method                          methods=['random'],                     # `random` to the                          wrapper=InstanceMethod)                 # FreeRV      # so we could sample from FreeRVdef _get_scaling(total_size, shape, ndim):  \"\"\"Gets scaling constant for logp\"\"\"  if total_size is None:    coef = floatX(1)  ...  return tt.as_tensor(floatX(coef))logp calculation happens in Normal.logp method. bound method in pymc3/distributions/dist_math.py, evaluates the expression given as first argument and returns the resulting theano expression if the condition presented as second argument is satisfied. If not it returns negative infinity. def logp(self, value):  \"\"\"Calculate log-probability of Normal distribution at specified value.\"\"\"\t...  return bound((-tau * (value - mu)**2 + tt.log(tau / np.pi / 2.)) / 2.,               sigma &gt; 0)def bound(logp, *conditions, **kwargs):  \"\"\"Bounds a log probability density with several conditions.\"\"\"\t...  return tt.switch(alltrue(conditions), logp, -np.inf)Reconstruction I’ve created a github repo to maintain the reconstructed code. Check out the code constructed from our examination in this blog post. In summary, we have looked at how PyMC3 creates and manages random variables of a model. We have learned that random variables (FreeRV) are associated with prior distributions and they can exhibit distribution-like behaviour and theano tensor-like behaviour. In the next blog, we’ll find out how random sampling is implemented in distributions and how we can evaluate (logp) arbitrary samples under a given distribution. ","categories": ["pymc3","deconstruction"],
        "tags": [],
        "url": "http://localhost:4000/pymc3/deconstruction/2019/11/22/deconstructing-pymc3-part-2.html"
      }]
