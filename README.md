# nn-playground
 Have fun with neural networks

## What's done
* MNIST
  * Implementing neural networks using PyTorch
    * A linear network
      * Achieved 92% accuracy
    * A fully connected network
      * Achieved 97% accuracy
    * A convolutional neural network
      * Achieved 99% accuracy
  * Implementing handmade neural networks using NumPy
    * A fully connected network
      * Achieved 95% accuracy with a simple SGD
        * Momentum alone didn't really help
    * A convolutional neural network
      * Achieved 98% accuracy with a simple SGD
      * Took 22 hours for just 10 epochs!
      * The loss didn't decrease during the 1st epoch
        * Was the initialization bad?
      * Also tried misimplementing bw_conv2d(), which actually worked for some reason...
        ![](img/cnn_loss_plot.png)
