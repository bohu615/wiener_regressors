from algorithms import *
from dataset import *

k = 0 # random seed

# works
print('running dataset F1')
x, desire = generate_F1()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, to_shuffle = True, seed=k, dataset='F1')
print('done!')

# works
print('running dataset F2')
x, desire = generate_F2()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, to_shuffle = True, seed=k, dataset='F2')
print('done!')

# works
print('running dataset F3')
x, desire = generate_F3()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, to_shuffle = True, seed=k, dataset='F3')
print('done!')

# works
print('running dataset CH')
x, desire = generate_CH()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, to_shuffle = True, D=2, seed=k, bs_QMI=600, dataset='CH', lr_MCA=0.01, lr_QMI=0.01)
print('done')

# works (PREVENT OVERFIT)
print('running dataset BH')
x, desire = generate_BH()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, to_shuffle = True, D=2, seed=k, prop=0.5, bs_QMI=600, dataset='BH', lr_MCA=0.01, lr_QMI=0.01)
print('done')

# works (PREVENT OVERFIT)
print('running dataset DB')
x, desire = generate_DB()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, to_shuffle = True, D=2, seed=k, bs_QMI=600, dataset='DB', lr_MCA=0.1, lr_QMI=0.01)
print('done')

#works
print('running dataset LORENZ')
x, desire = generate_LORENZ()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, D = 3, seed=k, bs_QMI=3000, dataset='LORENZ', QMI_if_unormalized = True)
print('done')

#works
print('running dataset LASER')
x, desire = generate_LASER()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, seed=k, dataset='LASER')
print('done')

print('running dataset SUN')
#works very well
x, desire = generate_SUN()
NN_output, MCA_output, QMI_output = run_dataset(x, desire, seed=k, dataset='SUN')
print('done')
