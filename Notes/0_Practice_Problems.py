import torch

# 2. Create a random tensor with shape (7, 7).
# random_tensor = torch.rand(7, 7)
# print(random_tensor)
# tensor([[0.4725, 0.0065, 0.5666, 0.7778, 0.4371, 0.6908, 0.4814],
#         [0.1954, 0.4909, 0.1541, 0.2079, 0.4431, 0.2792, 0.1883],
#         [0.6512, 0.5668, 0.0183, 0.7770, 0.5992, 0.6755, 0.6765],
#         [0.4503, 0.9866, 0.7368, 0.6875, 0.7938, 0.5919, 0.1780],
#         [0.6481, 0.8040, 0.2991, 0.8610, 0.5971, 0.1144, 0.3673],
#         [0.8052, 0.3918, 0.8630, 0.4132, 0.0294, 0.5927, 0.5533],
#         [0.5815, 0.5757, 0.4160, 0.8461, 0.1768, 0.5684, 0.6259]])


# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
# random_tensor_B = torch.rand(1, 7)
# random_tensor_B = torch.transpose(random_tensor_B, 1, 0)
# print('B', random_tensor_B)
# B tensor([[0.3692],
#         [0.2324],
#         [0.1259],
#         [0.4726],
#         [0.3290],
#         [0.4377],
#         [0.0619]])
# answer_matmul = torch.matmul(random_tensor, random_tensor_B)
# print(answer_matmul)
# tensor([[1.3075],
#         [0.6355],
#         [1.1828],
#         [0.8477],
#         [1.0047],
#         [1.2596],
#         [0.8367]])

# 4. Set the random seed to 0 and do 2 & 3 over again.
# The output should be:
# (tensor([[1.8542],
#          [1.9611],
#          [2.2884],
#          [3.0481],
#          [1.7067],
#          [2.5290],
#          [1.7989]]), torch.Size([7, 1]))

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
rand_tens_A = torch.rand(7, 7)
rand_tens_B = torch.rand(1, 7)
rand_tens_B = torch.transpose(rand_tens_B, 1, 0)
# print(torch.matmul(rand_tens_A, rand_tens_B))
# tensor([[1.8542],
#         [1.9611],
#         [2.2884],
#         [3.0481],
#         [1.7067],
#         [2.5290],
#         [1.7989]])

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one)
# If there is, set the GPU random seed to 1234.

# torch.cuda.manual_seed(1234)


# 6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed). 
# The output should be something like:
# Device: cuda
# (tensor([[0.0290, 0.4019, 0.2598],
#          [0.3666, 0.0583, 0.7006]], device='cuda:0'),
#  tensor([[0.0518, 0.4681, 0.6738],
#          [0.3315, 0.7837, 0.5631]], device='cuda:0'))

torch.manual_seed(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"

A = torch.rand(2,3).to(device)

B = torch.rand(2, 3).to(device)
# print(A)
# tensor([[0.0290, 0.4019, 0.2598],
#         [0.3666, 0.0583, 0.7006]], device='cuda:0')
# print(B)
# tensor([[0.0518, 0.4681, 0.6738],
#         [0.3315, 0.7837, 0.5631]], device='cuda:0')

# 7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
# The output should look like:
#                    (tensor([[0.3647, 0.4709],
#                             [0.5184, 0.5617]], device='cuda:0'), torch.Size([2, 2]))
B = B.transpose(1, 0)
Answer = torch.mm(A, B)
# print(Answer, Answer.shape)
# tensor([[0.3647, 0.4709],
#         [0.5184, 0.5617]], device='cuda:0') torch.Size([2, 2])

# 8. Find the maximum and minimum values of the output of 7.
# print(Answer.min())
# tensor(0.3647, device='cuda:0')
# print(Answer.max())
# tensor(0.5617, device='cuda:0')

# 9. Find the maximum and minimum index values of the output of 7.
print(Answer.argmin())
# tensor(0, device='cuda:0')
print(Answer.argmax())
# tensor(3, device='cuda:0')

# 10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10).
# Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.

# The output should look like:
    # tensor([[[[0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297,
    #            0.3653, 0.8513]]]]) torch.Size([1, 1, 1, 10])
    # tensor([0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297, 0.3653,
    #         0.8513]) torch.Size([10])
    
torch.manual_seed(7)
Rand_A = torch.rand(1, 1, 1, 10)
Squeezed_A = Rand_A.squeeze()
print(Rand_A, Rand_A.shape)
# tensor([[[[0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297,
#            0.3653, 0.8513]]]]) torch.Size([1, 1, 1, 10])
print(Squeezed_A, Squeezed_A.shape)
# tensor([0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297, 0.3653,
#         0.8513]) torch.Size([10])

    