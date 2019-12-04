# nn-playground
 Have fun with neural networks

## What's done
* MNIST
  * A fully connected network with PyTorch
    * Achieved 97% accuracy
  * A convolutional neural network with PyTorch
    * Achieved 99% accuracy
  * A handmade FC network with NumPy
    * Achieved 95% accuracy with a simple SGD
      * Momentum alone didn't really help
  * A handmade CNN with NumPy
    * Achieved 98% accuracy with a simple SGD
    * Took 22 hours for just 10 epochs!
    * The loss didn't decrease during the 1st epoch
      * Was the initialization bad?
    * Also tried misimplementing bw_conv2d(), which actually worked for some reason...
      ![](img/cnn_loss_plot.png)
