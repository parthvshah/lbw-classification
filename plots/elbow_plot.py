import matplotlib.pyplot as plt

plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [100.0,
                                            74.1020000000001,
                                            75.62999999999984, 
                                            76.25199999999955, 
                                            77.48299999999931, 
                                            79.34099999999889, 
                                            75.92199999999985, 
                                            78.03999999999922, 
                                            76.25799999999957, 
                                            77.17599999999923])
plt.xlabel('No. of neighbours')
plt.ylabel('Accuracy')
plt.savefig('elbow_avg_instances.jpg')