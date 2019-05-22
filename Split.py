import resnet
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from keras.models import Model
import keras
import numpy as np
from keras.layers import Input
import random
from keras import layers


resnet = resnet.ResNet152(weights='imagenet')
resnetLayers = [layer.name for layer in resnet.layers]


model1 = Model(inputs=resnet.input, outputs=resnet.get_layer('res4a_relu').output)
#model1.summary()


# this is the split point, i.e. the starting layer in our sub-model
starting_layer_name = 'res4a_relu'

# create a new input layer for our sub-model we want to construct
new_input = layers.Input(batch_shape=resnet.get_layer(starting_layer_name).get_input_shape_at(0))

layer_outputs = {}
def get_output_of_layer(layer):
    #print(layer.name)
    # if we have already applied this layer in its input(s),
    # just return the output
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    # if this is the starting layer, then apply it on the input tensor
    if layer.name == starting_layer_name:
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    # find all the connected layers which this layer
    # consumes their output
    prev_layers = []
    for node in layer._inbound_nodes:
        prev_layers.extend(node.inbound_layers)

    # get the output of connected layers
    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl)])

    # apply this layer on the collected outputs
    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out

# note that we start from the last layer of our desired sub-model.
# this layer could be any layer of the original model as long as it is
# reachable from the starting layer
new_output = get_output_of_layer(resnet.layers[-1])

# create the sub-model
model2 = Model(new_input, new_output)
