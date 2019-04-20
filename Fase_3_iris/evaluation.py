#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def build_grid(vir_vir, set_set, ver_ver, vir_set, vir_ver, set_vir, set_ver, ver_vir, ver_set):
    
    grid = {}
    
    grid[0] = [vir_vir, vir_set, vir_ver]
    grid[1] = [set_vir, set_set, set_ver]
    grid[2] = [ver_vir, ver_set, ver_ver]
    
    return grid


# In[ ]:


def confusion_matrix(pred_label, real_label):
    
    vir_vir = 0
    set_set = 0
    ver_ver = 0
    
    vir_set = 0
    vir_ver = 0
    
    set_vir = 0
    set_ver = 0
    
    ver_vir = 0
    ver_set = 0
    
    for i in range(len(pred_label)):
        
        if(pred_label[i] == real_label[i]):
            if(pred_label[i] == 'Virginica'):
                vir_vir = vir_vir + 1
            elif(pred_label[i] == 'Setosa'):
                set_set = set_set + 1
            elif(pred_label[i] == 'Versicolor'):
                ver_ver = ver_ver + 1
        
        if(pred_label[i] == 'Virginica'):
            if(real_label[i] == 'Setosa'):
                vir_set = vir_set + 1
            elif(real_label[i] == 'Versicolor'):
                vir_ver = vir_ver + 1
                
        if(pred_label[i] == 'Setosa'):
            if(real_label[i] == 'Virginica'):
                set_vir = set_vir + 1
            elif(real_label[i] == 'Versicolor'):
                set_ver = set_ver + 1
                
        if(pred_label[i] == 'Versicolor'):
            if(real_label[i] == 'Virginica'):
                ver_vir = ver_vir + 1
            elif(real_label[i] == 'Setosa'):
                ver_set = ver_set + 1
    
    
    return build_grid(vir_vir, set_set, ver_ver, vir_set, vir_ver, set_vir, set_ver, ver_vir, ver_set)
    
    
def get_results(pred_label, real_label):
    
    matrix = confusion_matrix(pred_label, real_label)
        
    acc = accuracy(matrix, len(pred_label))
    rec = recall(matrix)
    prec = precision(matrix)
    
    return acc, rec, prec, matrix

def ind_information(acc, rec, prec, matrix):
    print(matrix[0])
    print(matrix[1])
    print(matrix[2])
    print("\n")
    
    print("Acurácia: " + str(acc))
    print("\n")

    print("Recall Virginica: " + str(rec['Virginica']))
    print("Recall Setosa: " + str(rec['Setosa']))
    print("Recall Versicolor: " + str(rec['Versicolor']))
    print("\n")

    print("Precisão Virginica: " + str(prec['Virginica']))
    print("Precisão Setosa: " + str(prec['Setosa']))
    print("Precisão Versicolor: " + str(prec['Versicolor']))
    print("\n")

def show_information(acc, rec, prec, sample):
    
    acc_mean = 0
    for i in range(len(sample.keys())):
        acc_mean = acc_mean + acc[i]
    acc_mean = acc_mean/len(sample.keys())

    rec_mean_vir = 0
    rec_mean_set = 0
    rec_mean_ver = 0
    for i in range(len(sample.keys())):
        rec_mean_vir = rec_mean_vir + rec[i]['Virginica']
        rec_mean_set = rec_mean_set + rec[i]['Setosa']
        rec_mean_ver = rec_mean_ver + rec[i]['Versicolor']
    rec_mean_vir = rec_mean_vir/len(sample.keys())
    rec_mean_set = rec_mean_set/len(sample.keys())
    rec_mean_ver = rec_mean_ver/len(sample.keys())

    prec_mean_vir = 0
    prec_mean_set = 0
    prec_mean_ver = 0
    for i in range(len(sample.keys())):
        prec_mean_vir = prec_mean_vir + prec[i]['Virginica']
        prec_mean_set = prec_mean_set + prec[i]['Setosa']
        prec_mean_ver = prec_mean_ver + prec[i]['Versicolor']
    prec_mean_vir = prec_mean_vir/len(sample.keys())
    prec_mean_set = prec_mean_set/len(sample.keys())
    prec_mean_ver = prec_mean_ver/len(sample.keys())

    print("Média Acurácia: " + str(acc_mean))
    print('\n')
    print("Média Recall Virginica: " + str(rec_mean_vir))
    print("Média Recall Setosa: " + str(rec_mean_set))
    print("Média Recall Versicolor: " + str(rec_mean_ver))
    print('\n')
    print("Média Precisão Virginica: " + str(prec_mean_vir))
    print("Média Precisão Setosa: " + str(prec_mean_set))
    print("Média Precisão Setosa: " + str(prec_mean_ver))
    
    
    
    
def loo_result(pred_label, real_label):
    
    if(pred_label == real_label):
        return True
    
    return False
        

# In[ ]:


def accuracy(matrix, tam):
    
    results = (matrix[0][0] + matrix[1][1] + matrix[2][2])/tam
    
    return results


# In[ ]:


def precision(matrix):
    
    results = {}
    
    results['Virginica'] = matrix[0][0] / (matrix[0][1] + matrix[0][2] + matrix[0][0])
    results['Setosa'] = matrix[1][1] / (matrix[1][0] + matrix[1][1] + matrix[1][2])
    results['Versicolor'] = matrix[2][2] / (matrix[2][0] + matrix[2][2] + matrix[2][1])
    
    return results


# In[ ]:


def recall(matrix):
    
    results = {}
    
    results['Virginica'] = matrix[0][0] / (matrix[1][0] + matrix[2][0] + matrix[0][0])
    results['Setosa'] = matrix[1][1] / (matrix[0][1] + matrix[2][1] +  matrix[1][1])
    results['Versicolor'] = matrix[2][2] / (matrix[0][2] + matrix[1][2] + matrix[2][2])
    
    return results


# In[ ]:





# In[ ]:




