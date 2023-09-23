import torch

# SCALAR ===============================================================================================
# scalar = torch.sensor(7) = tensor(7) 
# scalar.ndim == 0

# VECTOR ===============================================================================================
# vector = torch.tensor([5, 5])
# vector.ndim == 1

# MATRIX ===============================================================================================
# MATRIX = torch.tensor([5, 5], [3, 3]) 
# MATRIX.ndim == 2

# TENSOR ===============================================================================================
# TENSOR = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# TENSOR.ndim == 3
# TENSOR.shape == tensor.size([1, 3, 3])

# RANDOM TENSOR EXAPMES ================================================================================
# random_tensor = torch.rand([3, 4, 4])
# tensor([[[0.8905, 0.2392, 0.5003, 0.2775],
#          [0.3179, 0.7080, 0.7229, 0.9193],
#          [0.7501, 0.6142, 0.6232, 0.7534],
#          [0.1705, 0.3215, 0.2057, 0.8828]],

#         [[0.8781, 0.8833, 0.2784, 0.5208],
#          [0.8234, 0.0729, 0.6915, 0.0085],
#          [0.9456, 0.5499, 0.5950, 0.8668],
#          [0.4829, 0.7054, 0.8021, 0.4857]],

#         [[0.2289, 0.4954, 0.0433, 0.9464],
#          [0.1805, 0.1813, 0.3178, 0.0964],
#          [0.3562, 0.7174, 0.9650, 0.1945],
#          [0.0970, 0.7967, 0.0517, 0.4727]]])
# print(random_tensor)

# NON RANDOM TESNSOR (GENERALLY FOR MASK)
# zeros = torch.zeros(4, 4)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
# print(zeros)
# random_tensor = torch.rand([3, 4, 4])
# print(zeros*random_tensor)
# tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],

#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],

#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

# RANDOM ONES TENSOR ========

# ones = torch.ones(4, 4)
# print(ones)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])

# RANGE =================================================
# print(torch.arange(0, 10))
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

# TENSORS LIKE ========================
# example = torch.arange(0, 10)
# TESNSOR_LIKE = torch.zeros_like(example)
# print(TESNSOR_LIKE)
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# TENSOR OPERATIONS IN BVUILT FUNCTIONS=================================================
#  Addition, Subtraction, multiplication, division, Matrix multiplication
# tensor = torch.tensor([1, 2, 3], [4, 5 ,6])

# print(torch.mul(tensor, 10))
# print(torch.add(tensor, 10))
# print(torch.sub(tensor, 10))
# print(torch.div(tensor, 10))

# MATRIX MULTIPLICATION (DOT PRODUCT) =================================
# tensor = torch.tensor([3, 4, 2])
# tensor2 = torch.tensor([[13, 9, 7, 15], [8, 7, 4, 6], [6, 4, 0, 3]])
# print(torch.matmul(tensor, tensor2))
# tensor([83, 63, 37, 75])
# ^
# 3 x 13 = 39, 4 x 8 = 32 = 61, 2 x 6 = 12 + 61 = 83 
# ^
# 3 x 9 = 27, 4 x 7 = 28 + 27 = 55, 2 x 4 = 8 + 55 = 63
# ^ AND SO ON

#RULES =================================================
# 1: the INNER DIMENSIONS MUST MATCH
# (1, '2') (DOT) ('2', 1) = success (2 and 2 are the same)
#  (2, '1') (DOT ) ('2', 1) = FAIL (1 and 2 are different)
# 2: the RESULT IS THE SHAPE OF THE OUTER DIMENSIONS
#  (1, '2') (DOT) ('2', 1) = (1, 1)
#  (3, 2) (DOT ) ('2', 3) = (3, 3)
# print(torch.matmul(torch.ones(1, 2), torch.rand(2, 1)))
# tensor([[0.8083]])

#  FIXING TENSOR MATMUL OF THE SAME SIZE ERROR (TRANSPOSE) ================================================
# tensor_A = torch.tensor([[1, 2], [3, 4], [5,6]])
# tensor_B = torch.tensor([[1, 2], [3, 4], [5,6]])
# torch.matmul(tensor_A, tensor_B) = ERROR (mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2))
# new_tensor_B = tensor_B.T
# ^ tensor([[1, 3, 5],
#           [2, 4, 6]])
# print(torch.matmul(tensor_A, new_tensor_B))
# tensor([[ 5, 11, 17],
#         [11, 25, 39],
#         [17, 39, 61]])

# FINDING MIN, MAX, MEAN, SUM, OF TENSORS (Aggergation) =================================
# A = torch.arange(0, 100, 15)
# #  MIN ======
# print(torch.min(A), A.min())
# # MAX ======
# print(torch.max(A), A.max())
# # MEAN (DOESNT WORK ON LONG DATATYPE (USE FLOAT32)) ======
# print(torch.mean(A.type(torch.float32)))
# # SUM ======
# print(torch.sum(A), A.sum())

 # RESHAPING, VIEWING, STACKING/SUPPRESSING =================================================
 
# tensor = torch.arange(10)

# RESHAPING --------------------------------

# print(tensor)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# tensor = tensor.reshape(2, 5)
# print(tensor)
# ^ tensor([[0, 1, 2, 3, 4],
#          [5, 6, 7, 8, 9]])
# print(tensor.shape)
# torch.Size([2, 5])

# VIEW (SHARES MEMEORY WITH ORIGINAL TENSOR) --------------------------------

# tensor_same = tensor.view(2, 5)
# tensor_same[0][0] = 5
# print(tensor)
# ^tensor([[5, 1, 2, 3, 4],
#          [5, 6, 7, 8, 9]])
# BEFORE IT WAS (1, 9) now (2, 5) because memory shared

# # Stack Tensors on top of each other --------------------------------
# stacked = torch.stack([tensor, tensor, tensor, tensor])
# # print(stacked)
# # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# stacked_dim1 = torch.stack([tensor, tensor, tensor, tensor], dim=1)
# # print(stacked_dim1)
# # tensor([[0, 0, 0, 0],
# #         [1, 1, 1, 1],
# #         [2, 2, 2, 2],
# #         [3, 3, 3, 3],
# #         [4, 4, 4, 4],
# #         [5, 5, 5, 5],
# #         [6, 6, 6, 6],
# #         [7, 7, 7, 7],
# #         [8, 8, 8, 8],
# #         [9, 9, 9, 9]])
# stacked_dimneg = torch.stack([tensor, tensor, tensor, tensor], dim=0)
# print(stacked_dimneg)

# SHIFTS ORDER OF DIMENSIONS (PERMUT) (SHARES SAME PLACE IN MEMORY!) =================================================================
# new_tensor  = torch.rand(size=(224, 224, 3))
# # print(new_tensor.shape)
# # torch.Size([224, 224, 3])
# # SHIFTS THE INDEX OF DIMENSIONS VALUE TO THAT SPOT IN THE LIST ('3' TO 0 INDEX AND 224 at the ENDS)
# shifted_tensor = new_tensor.permute(2, 0, 1)
# print(shifted_tensor.shape) 
# torch.Size([3, 224, 224])

# PUTTING A TENSOR / MODEL ON THE GPU =================================================================
# WORKS IF NO GPU IS AVAILABLE BY SWITCHING TO CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor([3, 3, 3])

#  TENSOR CURRENT ON CPU SEEN WITH TENSOR.DEVICE (DEFAULT CPU) ------
# print(tensor, tensor.device)
# tensor([3, 3, 3]), cpu

# Switch TENSOR to GPU (DEVICE IS CUDA IF AVALIABLE ON LINE 190) ----
tensor = tensor.to(device)
# print(tensor)
# tensor([3, 3, 3], device='cuda:0')

#  SWITCH TENSOR BACK TO CPUI (SOME CASES ONLY USE CPU) ---
# tensor = tensor.to('cpu')
# print(tensor.device)
# cpu
# OR USE --------------------------------
# tensor = tensor.cpu()
# print(tensor.device)
# cpu