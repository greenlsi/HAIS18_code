import numpy as np
from scipy import stats
import math


def classification_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def classification_binary_calculator(predictions, labels):
    tpos = 0
    tneg = 0
    fpos = 0
    fneg = 0
    brierScoreSum = []
    for predict, label in zip(predictions, labels):

        brierScoreSum.append((predict-label)*(predict-label))
        if np.argmax(predict):
            if np.argmax(predict) == np.argmax(label):
                tpos = tpos + 1
            else:
                fpos = fpos + 1
        else:
            if np.argmax(predict) == np.argmax(label):
                tneg = tneg + 1
            else:
                fneg = fneg + 1

    return tpos,tneg,fpos,fneg,brierScoreSum


def classification_binary_metrics(predictions, labels):
    tpos, tneg, fpos, fneg, bs = classification_binary_calculator(predictions, labels)
    metric = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1', 'E1', 'Informedness', 'Markedness', 'MCC',
              'DP', 'CP', 'Brier Score', 'b', 'Capacity of IDS']
    value = []
    temp = [tneg, tpos, fpos, fneg]

    print(temp)

    if np.nonzero(temp)[0].shape[0] < 3:
        print('LOG_ERROR: Model working wrong. Review data set structure')
        value = [-1]*len(metric)
        return value, metric

    # Sensitivity
    sens = float(tpos) / (tpos + fneg)
    value.append(sens)
    # Specificity
    spec = float(tneg) / (tneg + fpos)
    value.append(spec)
    # PPV
    ppv = float(tpos) / (tpos+fpos)
    value.append(ppv)
    # NPV
    npv = float(tneg) / (tneg+fneg)
    value.append(npv)
    # Accuracy
    acc = float(tpos + tneg) / (tpos + tneg + fpos + fneg)
    value.append(acc)
    # F1
    if sens==0 or ppv == 0:
        f1 = 0
    else:
        f1 = stats.hmean(np.array([sens, ppv]))
    value.append(f1)
    # E1
    e1 = (2*tneg)/(2*tneg+fpos+fneg)
    value.append(e1)
    # Informedless
    info = sens + spec - 1
    value.append(info)
    # Markedless
    mark = ppv+npv-1
    value.append(mark)
    # MCC
    mcc = np.mean(np.array([mark, info]))
    value.append(mcc)
    # DP
    if npv==0 or ppv==0 or sens==0:
        dp=0
    else:
        dp = stats.hmean(np.array([sens, ppv, npv]))
    value.append(dp)
    # CP
    if f1==0 or e1==0:
        cp=0
    else:
        cp = stats.hmean(np.array([f1,e1]))
    value.append(cp)
    # Brier Score
    bs = np.sum(bs)/len(bs)
    value.append(bs)
    # Capacity of IDS
    alpha = 1 - spec
    beta = 1 - sens
    b = float(tpos + fneg) / (tpos + tneg + fpos + fneg)

    #Completar, log base 2 y metodo para cambiar metric de 0 a 1 si hubiera
    if ppv == 0:
        ppv=1
    if npv==0:
        npv=1
    try:
        if ppv==1:
            cap = 1 - (((b * beta * math.log(1 - npv, 2)) + (
            (1 - b) * (1 - alpha) * math.log(npv, 2)) + ((1 - b) * alpha * math.log(0.000000000000000000001, 2)))
                       / ((b * math.log(b, 2)) + ((1 - b) * math.log(1 - b, 2))))
        elif npv==1:
            cap = 1 - (((b * (1 - beta) * math.log(ppv, 2)) + (b * beta * math.log(0.000000000000000000001, 2)) + ((1 - b) * alpha * math.log(1 - ppv, 2)))
                       / ((b * math.log(b, 2)) + ((1 - b) * math.log(1 - b, 2))))
        else:
            cap = 1 - (((b*(1-beta)*math.log(ppv, 2)) + (b*beta*math.log(1-npv, 2)) + ((1-b)*(1-alpha)*math.log(npv, 2)) + ((1-b)*alpha*math.log(1-ppv, 2)))
                /((b*math.log(b, 2)) + ((1-b)*math.log(1-b, 2))))
        value.append(b)
        value.append(cap)
    except ValueError:
        cap=1
        value.append(b)
        value.append(cap)

    return value, metric

def classification_confusion_matrix(predictions, labels):
    conf_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    for index in range(predictions.shape[0]):
        i = np.argmax(labels[index])
        j = np.argmax(predictions[index])
        conf_matrix[i,j]=conf_matrix[i,j]+1
    return conf_matrix

def classification_cohen_kappa(conf_matrix):
    No=0 #Diagonal
    Nt=0 #Total
    sumfil=np.zeros((conf_matrix.shape[0]))
    sumcol=np.zeros((conf_matrix.shape[1]))

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            Nt = Nt + conf_matrix[i,j]
            sumfil[i] = sumfil[i] + conf_matrix[i,j]
            sumcol[j] = sumcol[j] + conf_matrix[i,j]
            if i==j:
                No = No + conf_matrix[i,j]

    sumfc = 0
    for i in range(sumfil.shape[0]):
        sumfc= sumfc + sumfil[i]*sumcol[i]

    k = (No-sumfc/Nt)/(Nt-sumfc/Nt)
    return k, No, Nt