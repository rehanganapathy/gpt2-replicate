<h1>GPT2 Training and Architecture Reproduction</h1>

Followed along Karpathy's gpt2 reproduction, and there's a lot to learn here, really thankful. /n
This approach uses Pytorch, and is a decoder only architecture. Tokens are trained to get the logits (B,T,C). 
The activation function used here is GELU (Better than RELU cause it isn't 0 when the neuron isn't fired).
Parameters are shared at word embedding stage and at the classifcation stage(at the end).
Residuals have a slightly different connection.
Quantisation is really important as it affects training times, TF32 -> Bfloat16 makes a difference, but not so much in accuracy with the resources we have. 
Learning Rate decays in a cosine fashion after a warmp up. Weight decays are also used with the adam optimser. 
torch.compile() makes a huge difference to training time as data is not passed to the HMB, but instead kept on the GPU.
Adjust Batch Size according to your GPU and make sure it doesn't overload. 
Gradients are accumulated to match the batch size
Distributed Data Processing is used, you can use multiple GPUs and then average the gradients with dist.allReduce(), make sure the batches trained are different.
