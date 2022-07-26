import tensorflow as tf


@tf.function
def MyAdd(a,b):
    """TODO: Docstring for add.

    :arg a: first arg 
    :arg b: second arg
    :returns: outcome

    """
    outcome = a+b
    return outcome

def MyMult(a,b):
    """docstring for MyMult"""
    outcome = a*b
    return outcome

    

a = tf.constant([[3,2],[3,2]], name="a")
b = tf.constant([[2,3],[3,2]], name="b")

outcome=MyAdd(a,b)
tf.print(outcome)

outcome=MyMult(a,b)
tf.print(outcome)






