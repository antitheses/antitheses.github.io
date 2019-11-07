---
layout: post
title: 'Deconstructing PyMC3 : Part I'
date: 2019-11-07 1:38 +0530
categories: pymc3 deconstruction
author: suriya
---

> Models in PyMC3 are centered around the `Model` class. It has references to all random variables (RVs) and computes the model logp and its gradients. Usually, you would instantiate it as part of a with context, as follows:

{% highlight python %}
with Model() as model:
  # model definition
  pass
{% endhighlight %}

<img src="/images/posts/pymc3/model.png" class="align-left" alt="Instance Creation" width="400"/>

What happens behind the scenes while line `#1` is executed is all we care about in this blog post. This takes us to `pymc3/model.py`. The class `Model` extends the classes `Context`, `Factor`, `WithMemoization` and is instantiated via the metaclass `InitContextMeta`.

<br />

When line `#1` is executed, the metaclass `InitContextMeta`'s `__call__` method is called which executes `__init__` method of `Model`'s instance in its context. The metaclass consists only of the `__call__` method.

<img src="/images/posts/pymc3/instance-creation.png" class="align-center" alt="Instance Creation" width="400"/>

{% highlight python %}
class InitContextMeta(type):
  
  def __call__(cls, *args, **kwargs):
    instance = cls.__new__(cls, *args, **kwargs)  # <<
    ...
{% endhighlight %}


`cls` points to `Model` class. `Model.__new__` is called which creates an instance of `Model`. After which the parent instance is resolved and stored at `self._parent`. This feature exists to support Model nesting.


{% highlight python %}
 class Model(Context, Factor, WithMemoization, metaclass=InitContextMeta):
	  def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get('model') is not None:  # parent explicitly set by user
            instance._parent = kwargs.get('model')
        elif cls.get_contexts():  # <<
            instance._parent = cls.get_contexts()[-1]
        else:
            instance._parent = None 
{% endhighlight %}

The context manager syntax of `Model` is derived from `Context` class. Let us take a look.

{% highlight python %}
class Context:
    contexts = threading.local()  # :: contexts is a thread-local object
                                  # :: items stored will be hidden from other
      														# :: threads ::
    def __enter__(self):
        type(self).get_contexts().append(self) # :: add instance to context stack ::
				...
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()  #  :: remove instance from context stack ::
				...
        
    @classmethod
    def get_contexts(cls):  # :: return the thread-local stack ::
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []  # :: create a stack if one isn't available ::
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
			...
      return cls.get_contexts()[-1]  # :: return the deepest context in the stack ::
{% endhighlight %}

Notice that `get_contexts` is a `@classmethod`. Class methods take `cls` as their first argument. Since `Model` is a child of `Context`, `Context.get_contexts` is called, which returns an empty stack.

{% highlight python %}
 class Model(Context, Factor, WithMemoization, metaclass=InitContextMeta):
	  def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get('model') is not None:
            instance._parent = kwargs.get('model')
        elif cls.get_contexts():
            instance._parent = cls.get_contexts()[-1]
        else:
            instance._parent = None  # << parent instance is `None`
        ...
        return instance
{% endhighlight %}

We are back to `InitContextMeta.__call__` method. 

{% highlight python %}
class InitContextMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        with instance:  # << appends context
            instance.__init__(*args, **kwargs)
        return instance
{% endhighlight %}

`with instance:` invokes `Context.__enter__` which simply adds the instance to context stack. Note that `type(self)` resolves to `Model` class.

{% highlight python %}
def __enter__(self):
  type(self).get_contexts().append(self)  # :: add instance to thread-local stack ::
  return self
{% endhighlight %}

`Model.__init__` method is called within context of instance which creates a bunch of empty variables.

{% highlight python %}
self.named_vars = treedict()
self.free_RVs = treelist()
self.observed_RVs = treelist()
self.deterministics = treelist()
self.potentials = treelist()
self.missing_values = treelist()
{% endhighlight %}

Note that if there exists a parent instance `self._parent`, the parent's variables are added to the instance's variables as parents.

{% highlight python %}
self.named_vars = treedict(parent=self.parent.named_vars)
...
{% endhighlight %}

We exit the context `with instance:` invoking the `__exit__` method which removes the instance from contexts stack.  This takes us back to `InitContextMeta.__call__` where the `Model` instance is finally returned.

{% highlight python %}
with Model() as model:
  ...
{% endhighlight %}

The use of context manager syntax here takes us again to `Context.__enter__`.  `Model` instance is added to context stack. Now we have an empty container for a model. Right now we don't care about the details of data structures `treelist` and `treedict`.

## Reconstruction

Let us reconstruct the `Model` class with the information available so far.

{% highlight python %}
import threading


class Context:
  contexts = threading.local()  # thread-local storage

  def __enter__(self):
    cls = type(self)  # get a handle to Class `Context`
    contexts = cls.get_contexts()  # call @classmethod `get_contexts`
    contexts.append(self)  # add instance to contexts
    return self

  def __exit__(self, typ, value, traceback):
    cls = type(self)  # get a handle to Class `Context`
    contexts = cls.get_contexts()  # call @classmethod `get_contexts`
    contexts.pop()  # remove instance from contexts stack

  @classmethod
  def get_contexts(cls):
    if not hasattr(cls.contexts, 'stack'):  # does `Context.contexts.stack` exist?
      cls.contexts.stack = []  # create and return an empty stack
    return cls.contexts.stack

  @classmethod
  def get_context(cls):
    contexts = cls.get_contexts()  # get all contexts
    if len(contexts) == 0:
      raise Exception("Context stack is empty!")
    return contexts[-1]  # return the deepest context


class InitContextMeta(type):
  """Metaclass that runs Model.__init__ in its context"""
  def __call__(cls, *args, **kwargs):
    instance = cls.__new__(cls, *args, **kwargs)  # create an instance of `Model`
    with instance:  # run __init__ in context
      instance.__init__(*args, **kwargs)  # `Model.__init__`
    return instance


class Model(Context, metaclass=InitContextMeta):

  def __new__(cls, *args, **kwargs):  # class method that creates an instance
    instance = super().__new__(cls)
    # resolve parent instance
    if cls.get_contexts():  # if contexts stack isn't empty
      instance._parent = cls.get_context()  # get the deepest context
    else:
      instance._parent = None
    return instance

  def __init__(self, name=''):
    self.name = name
    if self.parent is None:
      self.named_vars = {}  # using python dict instead of pymc's treedict
      self.free_RVs = []    # using python list instead of pymc's treelist
      self.observed_RVs = []
      self.deterministics = []
      self.potentials = []
      self.missing_values = []
    else:
      raise NotImplementedError('We dont care about this case yet!')

  @property
  def parent(self):
    return self._parent
  
  
if __name__ == '__main__':
  with Model() as model:
    pass
{% endhighlight %}


We were able to build an empty model. In the next post, we'll deconstruct the process behind *Model Definition* -  creating and managing random variables within the context of the model.
