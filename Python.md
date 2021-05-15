# Some Python Questions

## 1. What is the difference between a list and a tuple ?

- A list is mutable, tuple is not
- Tuples can be hashed (as a key for a dictionary), lists cannot
- list is defined using [], tuple with ()

## 2. Hash of a tuple and Hash of a list

Tuple has a __hash__() method implemented for it whereas List does not. 

If a list were hashable, changing its elements would change the hash of the list hence breaking the contract.

Tuples are not always hashable in Python, because tuples may hold references to unhashable objects such as lists or dicts. A tuple is only hashable if all of its items are hashable as well.

## 3. Difference between an array and a linked list

An array is
- ordered collection of data
- assumes every element is of same size 
- entire array is stored in a contiguous block of memory
- can directly access an element of an array given the index

A linked list
- series of data with pointers 
- not stored in a contiguous block of memory
- can only access data elements in a sequential manner

## 4. Difference between iterator and iterable

An **iterator** is an object representing a stream of data.
- It returns the data one at a time
- It supports `__next__` and `__iter__` methods and raises `StopIteration exception` when it runs out of elements to return

- An **iterable**  supports `__iter__` method
- Every iterable is not an iterable - For e.g List is an iterable but not an iterator.

Both are stateful objects. Once you have consumed objects from it, it is gone.

## Why is range not an iterator ?

- We can loop over range objects without consuming it
- We **cannot** call `__next__` on a range object
- Unlike iterators range objects have length and index associated with them
- Unlike iterators can ask them if they contain things without changing their state
- Ranges can be called 'lazy sequences'

## Is Python interpreted or compiled ? 

In Python, the source code is compiled into a much simpler form called bytecode. These are instructions similar in spirit to CPU instructions, but instead of being executed by the CPU, they are executed by software called a virtual machine. (These are not VM’s that emulate entire operating systems, just a simplified CPU execution environment.)

An important aspect of Python’s compilation to bytecode is that it’s entirely implicit. You never invoke a compiler, you simply run a .py file. The Python implementation compiles the files as needed.

This is different than Java, for example, where you have to run the Java compiler to turn Java source code into compiled class files. For this reason, Java is often called a compiled language, while Python is called an interpreted language. But both compile to bytecode, and then both execute the bytecode with a software implementation of a virtual machine.

Ref : https://nedbatchelder.com/blog/201803/is_python_interpreted_or_compiled_yes.html

## What is the lambda operator?
Lambda operator or lambda function is used for creating small, one-time and anonymous function objects in Python. It can have any number of arguments, but it can have only one expression. It cannot contain any statements and it returns a function object which can be assigned to any variable. Mostly lambda functions are passed as parameters to a function which expects a function object as parameter like map, reduce, filter functions

## Create empty array using numpy

```np.empty([2, 2])```

## Sum of the digits of a number
```
number = 932
sumDigit = sum(int(digit) for digit in str(number))
print(sumDigit)

sumDigit1 = sum(map(int, str(number)))
print(sumDigit1)
```

## What is the difference between the list methods append and extend?
- Append adds its argument as a single element to the end of a list.  The length of the list itself will increase by one. 
- Extend iterates over its argument adding each element to the list, extending the list.

## Difference Between remove, del and pop in Python list:
- remove() deletes the matching element/object whereas del and pop removes the element at a specific index.
- del and pop deals with the index. The only difference between two is that- pop returns deleted the value from the list and del does not return anything.
- Pop is only method that returns the object.
- Remove is the only method that searches object (not index).

## Which is the best way to delete the element in List?
- If you want to delete a specific object in the list, use remove method.
- If you want to delete the object at a specific location (index) in the list, you can either use del or pop.
- Use the pop, if you want to delete and get the object at the specific location.

- Usage - <listname>.pop() - Eg - ```a.pop()```
- Usage - <listname>.pop(<index>) - Eg - ```a.pop(2)```
- Usage - del <listname>[index]  - Eg - ```a.delete(1)```
- Usage - <listname>.remove(<item>) - Eg - ```a.remove(1.1)```

## Read a random line in a file
```
import random
def read_random(fname):
    lines = open(fname).read().splitlines()
    return random.choice(lines)
lines = read_random('hello.txt')
print(lines)

```

## Given a string, write a Python program to split strings on Uppercase characters

```
import re
ini_str = 'PhilomathInfiniteCosmosSoulZZ'

res_list = [s for s in re.split('([A-Z][^A-Z]*)', ini_str) if s]

print(res_list)
```
**Output**

```['Philomath', 'Infinite', 'Cosmos', 'Soul', 'Z', 'Z']```

## Given a list of numbers and a variable k, where k is also a number, write a Python program using Numpy module, to find the number in a list which is closest to the given number k.

```
import numpy as np
def closest(lst, k):
    lst = np.asarray(lst)
    idx = (np.abs(lst - k)).argmin()
    return lst[idx]

lst = [3.64, 5.2, 9.42, 9.35, 8.5, 8]
k = 9.1
print(closest(lst, k))
```

**Output**

```9.35```

## Given a list, write a Python program to convert the given list to dictionary such that all the odd elements become the key, and even number elements become the value.
"""
```
def convert(lst):
    dct = {lst[i] : lst[i + 1] for i in range(0, len(lst), 2)}
    return dct

lst = [1, 'a', 2, 'b', 3, 'c']
print(convert(lst))
```

# Output: 

```{1: 'a', 2: 'b', 3: 'c'}```

## Given a list of integers and an integer variable K, write a python program to find all pairs in the list with given sum K.

```
def findPairs(lst, k):
    res = []
    while lst:
        num = lst.pop()
        diff = k - num
        if diff in lst:
            res.append((diff, num))
    return res

lst = [1, 5, 3, 7, 9]
K = 12
print(findPairs(lst, K))
```
**Output:** 
```[(3, 9), (5, 7)]```

## How will you Extract digits from given string

```
test_string = '1visio2builDer3'
print('Orig str', test_string)
# Matching all items that not digits and replacing them with ''
res = re.sub('\D', '', test_string)
print('Digits string is ', res)
```

**Output:** 
```Digits string is  123```

## Write a program to Split a string on last occurrence of delimiter

```
test_string = 'data, is, good, better, and best'
print('The original string : ' + str(test_string))
res = test_string.rsplit(', ', 1)
print('The post-split list at the last comma : ' + str(res))
```

**Output:** 
```['data, is, good, better', 'and best']```

# Else in For Loop

The ```else``` keyword in a ```for``` loop specifies a block of code to be executed when the loop is finished

```
for x in range(6):
  print(x)
else:
  print("Finally finished!")  
```

**Output**

```
0
1
2
3
4
5
Finally finished!
```  

## Get the string after occurrence of a given substring
```
test_string = 'Python is best for data science'
spl_word = 'best'
print('Orig str --', test_string)
print('To split on word  --', spl_word)
res = test_string.partition(spl_word)
last_item = res[-1]
print('Result -- ', res)
print('Last Item -- ', last_item)

print('-' * 30)
print('Case sensitiveness 1 ----------- ')
test_string = 'Python is best for data science'
spl_word = 'python'
print('Orig str --', test_string)
print('To split on word  --', spl_word)
res = test_string.partition(spl_word)
last_item = res[-1]
print('Result -- ', res)
print('Last Item -- ', last_item)

print('-' * 30)
print('Case sensitiveness 2 ----------- ')
test_string = 'Python is best for data science'
spl_word = 'Python'
print('Orig str --', test_string)
print('To split on word  --', spl_word)
res = test_string.partition(spl_word)
last_item = res[-1]
print('Result -- ', res)
print('Last Item -- ', last_item)
```
**Output**

```
Orig str -- Python is best for data science
To split on word  -- best
Result --  ('Python is ', 'best', ' for data science')
Last Item --   for data science
------------------------------
Case sensitiveness 1 ----------- 
Orig str -- Python is best for data science
To split on word  -- python
Result --  ('Python is best for data science', '', '')
Last Item --  
------------------------------
Case sensitiveness 2 ----------- 
Orig str -- Python is best for data science
To split on word  -- Python
Result --  ('', 'Python', ' is best for data science')
Last Item --   is best for data science
```

## Valid email

```
import re

regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
def isValidMail(email):
    if re.search(regex, email):
        print('Valid email --> ' , email)
    else:
        print('Invalid email --> ' , email)
        
email = 'testtest@example.com'
isValidMail(email)

email = 'hell.week.23@tension.com'
isValidMail(email)

email = 'this@tension..com'
isValidMail(email)

```

**Output**

```
Valid email -->  testtest@example.com
Valid email -->  hell.week.23@tension.com
Invalid email -->  this@tension..com
```

## Pad zeros to the right

```
test_string = 'GAP'
print('Orig str ', test_string)

# No. of zeros required
N = 4

# using rjust() add leading zeros
# string.rjust(length, fillchar)

res = test_string.rjust(N + len(test_string), '0')

print(res)

```

**Output**

```
Orig str  GAP
0000GAP
```
## 
