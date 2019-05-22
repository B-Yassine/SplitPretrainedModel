# SplitPretrainedResNetModel
Split pretrained ResNet model at any intermediate layer 


We need to find the connectivity of the layers and traverse that connectivity map to be able to construct a sub-model of the original model. 

    1/ Specify the last layer of your sub-model.
    2/ Start from that layer and find all the connected layers to it.
    3/ Get the output of those connected layers.
    4/ Apply the last layer on the collected output.

Step #3 implies a recursion: to get the output of connected layers (i.e. X), we first need to find their connected layers (i.e. Y), get their outputs (i.e. outputs of Y) and then apply them on those outputs (i.e. apply X on outputs of Y)


Thanks to this answer which can be found [here](https://stackoverflow.com/questions/56147685/how-to-split-a-keras-model-with-a-non-sequential-architecture-like-resnet-into)
