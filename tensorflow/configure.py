import tensorflow as tf
import sys

if '__cxx11_abi_flag__' not in dir(tf):
    print("Cannot find the ABI version of TensorFlow.")
    print("Your TensorFlow version is too old. Please upgrade to at least TF v1.4.")
    sys.exit(1)

with open("tensorflow_config.txt", "w") as f:
    print("TensorFlow_ABI: {}".format(tf.__cxx11_abi_flag__))
    f.write("set(TensorFlow_ABI %i)\n" % tf.__cxx11_abi_flag__)
    print("TensorFlow_INCLUDE_DIRS: {}".format(tf.sysconfig.get_include()))
    f.write("set(TensorFlow_INCLUDE_DIRS \"%s\")\n" % tf.sysconfig.get_include())

