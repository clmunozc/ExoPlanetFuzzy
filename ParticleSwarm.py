#from exoNN import evaluate_NN
#from exoNN import initialize as init_nn
from generate_fis import evaluate_fuzzy, initialize_fuzzy
#from show_fuzzy import evaluate_fuzzy as test_fuzzy
#from show_fuzzy import initialize as init_test
from fstpso import FuzzyPSO
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize
from datetime import datetime
#from fuzzy import initialize
import sys
import warnings
import numpy as np
 
 
#warnings.simplefilter("ignore")

#ss=[2,1,1]
#types=[0,1,2]
n_exp = 3
dims = 24
variables = int(dims/2)
i=0
f = open("solutions_fuzzy_ex.txt", "a")
universe = [[0, 10]]*dims
for i in range(3):
    initialize_fuzzy(i)
    terminado = False
    completados = 0
    while not terminado:
        if completados>=3:
            terminado = True
            break
        f.write("solucion No. "+str(completados+1)+" para sistema difuso "+str(i)+"\n")
        try:
            print("Realizando optimizacion de sistema difuso "+str(i+1)+"...")
            FP = FuzzyPSO()
            FP.set_search_space(universe)	
            FP.set_fitness(evaluate_fuzzy)
            #FP.set_fitness(evaluate_SIDRA_grad_fuzzy)
            #FP.set_fitness(evaluate_EMG_fuzzy)
            #FP.set_fitness(evaluate_EMG_grad_fuzzy)
            t0 = datetime.now()
            result =  FP.solve_with_fstpso()
            #result = [[0, 0, 0, 0, 0, 0, 0],1]
            tf = datetime.now()
            tiempo= tf - t0
            tiempo=tiempo.total_seconds()
            print("Tiempo de ejecucion:" + str(tiempo/3600)+" horas.")
            print("Mejor solucion:")
            print(result[0])
            solution = str(np.array(result[0]))
            f.write(solution+"\n")
            print("Cuyo rendimiento llego a:")
            print(result[1])
            completados=completados+1
        except KeyboardInterrupt:
            sys.exit()
        except:
            print("Hubo un error en la optimizacion del sistema difuso, reintentando...")
f.close()
    
