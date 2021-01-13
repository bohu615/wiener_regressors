import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.rcParams["figure.figsize"] = [6,4]

def plot_MSE_CC(NN_output_F1, MCA_output_F1, QMI_output_F1, dataset):
  plt.plot(NN_output_F1[0])
  plt.plot(MCA_output_F1[0])
  plt.plot(QMI_output_F1[0])

  plt.plot(NN_output_F1[1])
  plt.plot(MCA_output_F1[1])
  plt.plot(QMI_output_F1[1])

  plt.legend(['MSE/BP Training Curve', 'MCA Training Curve', 'QMI Training Curve', 
              'MSE/BP Testing Curve', 'MCA Testing Curve', 'QMI Testing Curve'])
  
  plt.grid(b=None)

  plt.xlabel('Iterations')
  plt.ylabel('MSE')
  plt.title('Dataset: '+dataset)
  
  plt.savefig('./figures/{0}_MSE.pdf'.format(dataset), bbox_inches='tight', dpi=300)
  #plt.show()
  plt.clf()
  plt.cla()
  plt.close()


  plt.plot(NN_output_F1[3])
  plt.plot(MCA_output_F1[3])
  plt.plot(QMI_output_F1[3])

  plt.plot(NN_output_F1[4])
  plt.plot(MCA_output_F1[4])
  plt.plot(QMI_output_F1[4])

  plt.legend(['MSE/BP Training Curve', 'MCA Training Curve', 'QMI Training Curve', 
              'MSE/BP Testing Curve', 'MCA Testing Curve', 'QMI Testing Curve'])
  
  plt.grid(b=None)

  plt.xlabel('Iterations')
  plt.ylabel('CC')
  plt.title('Dataset: '+dataset)

  plt.savefig('./figures/{0}_CC.pdf'.format(dataset), bbox_inches='tight', dpi=300)
  #plt.show()
  plt.clf()
  plt.cla()
  plt.close()


def read_outputs(seed=0, dataset='F1'):
  NN_output_F1 = np.load('./outputs/NN_output_seed{0}_dataset{1}.npy'.format(seed, dataset), allow_pickle=True)
  MCA_output_F1 = np.load('./outputs/MCA_output_seed{0}_dataset{1}.npy'.format(seed, dataset), allow_pickle=True)
  QMI_output_F1 = np.load('./outputs/QMI_output_seed{0}_dataset{1}.npy'.format(seed, dataset), allow_pickle=True)
  return NN_output_F1, MCA_output_F1, QMI_output_F1

dataset = ['F1', 'F2', 'F3', 'CH', 'BH', 'DB', 'LORENZ', 'LASER', 'SUN']
for set in dataset:
  NN_output_F1, MCA_output_F1, QMI_output_F1 = read_outputs(seed=0, dataset=set)
  plot_MSE_CC(NN_output_F1, MCA_output_F1, QMI_output_F1, set)