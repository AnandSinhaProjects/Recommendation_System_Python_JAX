# Recommendation Systems with Python and JAX

Person: Anand Sinha
Status: In progress
Tags: Lesson
Files: Recommendation%20Systems%20with%20Python%20and%20JAX%2025c61676103e43f8bcd44e93162e4238/Bryan_Bischof_Hector_Yee_-_Building_Recommendation_Systems_in_Python_and_JAX__Hands-On_Production_Systems_at_Scale-OReilly_Media_(2023).pdf

# Introduction

Ubiquity of any technology often prompts questions of how the technology works, why it has become so common, and if we can get in on the action. For recommendation systems, the *how* is quite complicated. We’ll need to understand the geometry of taste, and how only a little bit of interaction from a user can provide us a *GPS signal* in that abstract space. You’ll see how to quickly gather a great set of candidates and how to refine them to a cohesive set of recommendations.

We will formulate variants of the core problem to be solved by recommendation systems but, ultimately, the motivating problem framing is as follows:

---

## **Key Components of a Recommendation System**

We will identify and build on three core components of recommendation systems: the collector, ranker, and server.

### **Collector**

The collector’s role is to know what is in the collection of things that may be rec‐ ommended, and the necessary features or attributes of those things. Note that this collection is often a subset based on context or state.

### **Ranker**

The ranker’s role is to take the collection provided by the collector and order some or
all of its elements, according to a model for the context and user.

### **Server**

The server’s role is to take the ordered subset provided by the ranker, ensure that the
necessary data schema is satisfied—including essential business logic—and return the
requested number of recommendations.

---

## **Simplest Possible Recommenders**

We’ve established the components of a recommender, but to really make this practical, we need to see this in action. 

### **The Trivial Recommender**

The absolute simplest recommender is not very interesting but can still be demonstrated in the framework. It’s called *the trivial recommender* (*TR*) because it contains virtually no logic:

```python
import random

def get_trivial_recs(list):
    item_id = random.randint(0, (len(list)-1))
    return item_id

def get_availability(eglist):
    item_id = get_trivial_recs(eglist)
    if item_id != 0:
        return eglist[item_id]
    else :
        return None
    
get_availability([1,2,3,4,5,6,7,8])
```

Notice that this recommender may return either a specific item_id or None. Also observe that this recommender takes no arguments, and MAX_ITEM_INDEX is referencing a variable out of scope. Software principles ignored, let’s think about the three components:

### *Collector*

A random item_id is generated. The TR collects by checking the availability of item_id. We could argue that having access to item_id is also part of the collector’s responsibility. Conditional upon the availability, the collection of recommendable things is either [item_id] or None.

### *Ranker*

The TR ranks with a no-op; i.e., the ranking of 1 or 0 objects in a collection is the identity function on that collection, so we merely do nothing and move on to the next step.

### *Server*

The TR serves recommendations by its return statements. The only schema
that’s been specified in this example is that the return type is [List[str]].

***This recommender, which is not interesting or useful, provides a skeleton that we will add to as we develop further.***

## Most-Popular-Item Recommender

The *most-popular-item recommender* (MPIR) is the simplest recommender that contains any utility. An MPIR works just as it says; it returns the most popular items:

```python
#dict_test={'item1': 12, 'item2': 18, 'item3': 5, 'item4': 1, 'item5': 12, 'item6': 17, 'item7': 13, 'item8': 12, 'item9': 3, 'item10': 2}

import random

def generate_item_popularities():
  """Generates a dictionary of item names and their random number of appearances.

  Returns:
    A dictionary where keys are item names (strings) and values are the number
    of times each item appeared (integers between 1 and 20).
  """

  item_names = ["item1", "item2", "item3", "item4", "item5", "item6", "item7", 
                "item8", "item9", "item10","item11", "item12", "item13", "item14", "item15"]  # Add more item names as needed
  item_popularities = {}
  for item in item_names:
    appearances = random.randint(1, 20)
    item_popularities[item] = appearances
  return item_popularities

# Example usage
item = generate_item_popularities()
print(item)

#def get_item_popularities(ite):
#    return item.values()

max_num_recs = 10
def get_most_popular_recs(item):
    items_popularity_dict = item
    sorted_items = sorted(items_popularity_dict.items(), key= lambda item: item[1], reverse=True)
    return sorted_items[0:max_num_recs]

get_most_popular_recs(item)
```

This recommender attempts to return the *k* most popular items available. While simple, this is a useful recommender that serves as a great place to start when building a recommendation system.

***Collector*** → ***“item”*** is the place where we collect the data. Ideally the generate the item popularities would get the data.

***Ranker →*** Here we see our first simple ranker: ranking by sorting on values. Because the collector has organised our data such that the values of the dictionary are the counts, we use the Python built-in sorting function sorted. Note that we use key to indicate that we wish to sort by the second element of the tuples—in this case, equivalent to sorting by values—and we send the reverse flag to make our sort descending.

***Server →*** Finally, we need to satisfy our API schema, which is again provided via the return type Optional[List[str]]. This wants the return type to be the nullable list of item-identifier strings that we’re recommending, so we use a list comprehension to grab the first element of the tuples. But wait! Our function has this max_num_recs field—what might that be doing there? Of course, this is suggesting that our API schema is looking for no greater than max_num_recs in the response. We handle this via the slice operator, but note that our return is between 0 and max_num_recs results.

## What is JAX ?

JAX not be confused with AJAX is a framework for writing mathematical code in Python that is just-in-time (JIT) compiled. JIT compilation allows the same code to run on CPUs, GPUs, and TPUs. This makes it easy to write performant code that takes advantage of the parallel-processing power of vector processors.

Additionally, one of the design philosophies of JAX is to support tensors and gradi‐ ents as core concepts, making it an ideal tool for ML systems that utilize gradient- based learning on tensor-shaped data.

We import JAX’s version of NumPy as jnp to distinguish it from NumPy (np) by convention so that we know which version of a mathematical function we want to use. This is because sometimes we might want to run code on a vector processor like a GPU or TPU that we can use JAX for, or we might prefer to run some code on a CPU in NumPy.

By making functions pure and by making data immutable, JAX is able to make some guarantees to the underlying accelerated linear algebra (XLA) library that it uses to talk to GPUs. JAX guarantees that these functions applied to data can be run in parallel and have deterministic results without side effects, and thus XLA is able to compile these functions and make them run much faster than if they were run just on NumPy.
You can see that modifying one element in x results in an error. JAX would prefer that the array x is replaced rather than modified. One way to modify elements in an array is to do it in NumPy rather than JAX and convert NumPy arrays to JAX—for example, using jnp.array(np_array)—when the subsequent code needs to run fast on immutable data.

```python
import jax.numpy as jnp
import numpy as np

x = jnp.array([1.0,2.0,3.0], dtype=jnp.float32)
print(x)
print(x.shape)
```

As JAX is immutable the assignments dont work directly and rather we need to use either jnp.array(np.array) or use example.at(x).set(y) command

```python
x = jnp.array(np.array([3,4,4,5], dtype=float32)
#or 
x.at[::2].set(1)
```

### Indexing and Slicing a JNP

NumPy introduced indexing and slicing operations that allow us to access different parts of an array. In general, the notation follows a start:end:stride convention. The first element indicates where to start, the second indicates where to end (but not inclusive), and the stride indicates the number of elements to skip over. The syntax is similar to that of the Python range function.
Slicing allows us to access views of a tensor elegantly. Slicing and indexing are important skills to master, especially when we start to manipulate tensors in batches, which we typically do to make the most use of acceleration hardware.

```python
x = jnp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=jnp.int32)

#Print the whole matrix
print(x)
print('------------------------')
#print the first row
print(x[0])
print('------------------------')

#Print th last row
print(x[-1])
print('------------------------')

#Print the second column
print(x[:,1])
print('------------------------')

#Print every other element
print(x[::2, ::2])
print('------------------------')
```

### Broadcasting

When a binary operation such as addition or multiplication is applied to two tensors of different sizes, the tensor with axes of size 1 is lifted up in rank to match that of the larger-sized tensor. For example, if a tensor of shape (3,3) is multiplied by a tensor of shape (3,1), the rows of the second tensor are duplicated before the operation so that it looks like a tensor of shape (3,3):

```python
vec =  jnp.reshape(jnp.array([0.5,1.0,2.0]),[1,3])
print(vec)
print('------------------------')

y_vec_1 = vec * x
print(y_vec_1)
print('------------------------')

vec_rev =  jnp.reshape(jnp.array([0.5,1.0,2.0]),[3,1])
print(vec_rev)
print('------------------------')

y_vec_2 = x * vec_rev
print(y_vec_2)
print('------------------------')
```

The first case is the simplest, that of scalar multiplication. The scalar is multiplied throughout the matrix. In the second case, we have a vector of shape (3,1) multiply‐ ing the matrix. The first row is multiplied by 0.5, the second row is multiplied by 1.0, and the third row is multiplied by 2.0. However, if the vector has been reshaped to (1,3), the columns are multiplied by the successive entries of the vector instead.

### Random Numbers

Along with JAX’s philosophy of pure functions comes its particular way of han‐ dling random numbers. Because pure functions do not cause side effects, a random-number generator cannot modify the random number seed, unlike other random-number generators. Instead, JAX deals with random-number keys whose state is updated explicitly:

```python
import jax.random as random

key = random.PRNGKey(0)
x = random.uniform(key, shape=[3,3]
print(x)
```

JAX first requires you to create a random-number key from a seed. This key is then passed into random-number generation functions like uniform to create random numbers in the 0 to 1 range.

```python
key, subkey = random.split(key)
x = random.uniform(key, shape=[3,3])
print(x)
```

To create more random numbers, however, JAX requires that you split the key into two parts: a new key to generate other keys, and a subkey to generate new random numbers. This allows JAX to deterministically and reliably reproduce random num‐ bers even when many parallel operations are calling the random-number generator. We just split a key into as many parallel operations as needed, and the random numbers resulting are now randomly distributed but also reproducible.

### Just-in-Time Compilations

JAX starts to diverge from NumPy in terms of execution speed when we start using JIT compilation. JITing code—transforming the code to be compiled just in time— allows the same code to run on CPUs, GPUs, or TPUs:

```python

import jax
x = random.uniform(key, shape=[2048, 2048]) - 0.5

def my_function(x): x=x@x
	return jnp.maximum(0.0, x)

%timeit my_function(x).block_until_ready()

my_function_jitted = jax.jit(my_function)
%timeit my_function_jitted(x).block_until_ready()
```

The JITed code is not that much faster on a CPU but will be dramatically faster on a GPU or TPU backend. 

*Compilation also carries some overhead when the function is called the first time, which can skew the timing of the first call.*

Variable-length loops trigger frequent recompilations. The “Just-in-Time Compilation with JAX” documentation covers a lot of the nuances of getting functions to JIT compile.

## The User-Item Matrix

It’s extremely common to hear those who work on recommendation systems talk about matrices, and in particular the user-item matrix. While linear algebra is deep, both mathematically and as it applies to RecSys, we will begin with simple relationships.
