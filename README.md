# Model Zoo

This repo is a collection of several deep learning models implemented using Pytorch according to corresponding papers.

Each folder contains the model, train and test scripts.  

## A list of all the models: 
(under construction)
- U-net implementation 
- U-net implementation (Non-cropped version)


## Some issues need to emphasize:

### Choice of Upsampling or Transposed convolution
- **Upsampling** can be seen as expanding the size by repeating the values to fill the extra space
- **transposed convolution** (de-convolution) is acutally a convolution and have parameters to learn
- In all the implementations, I used transposed convolution unless the paper has clearly pointed out the way they used to increase the dimensions.

