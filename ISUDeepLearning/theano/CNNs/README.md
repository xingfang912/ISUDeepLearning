This folder contains several convolutional neural networks (CNNs).
File name descriptions:
C stands for convolution. P stands for pooling. FC stands for fully connected.
For example, C_CP_FC.py includes a CNN that has its first layer as a convolutional layer (C), its second layer as a convolutional layer (C), its third layer as a pooling layer (P), and its fourth layer as a fully connected layer (FC).
We put CP together is because of the CPLayer class in CNNLib.py, meaning that the pooling layer can be turned off resulting a pure convolutional layer (C).
