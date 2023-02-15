import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def plot_multi_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          text='',
                          cmap=plt.cm.Blues,
                          output = "D:/Videos/Ceilling/video_1/TrainAndTest/"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    cm_multi = multilabel_confusion_matrix(np.array(y_true), np.array(y_pred))
    # Only use the labels that appear in the data

    result_all = []
    for id_mat,cm in enumerate(cm_multi):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Not normalized confusion matrix')
        tn = cm[0, 0]
        tp = cm[1, 1]
        fn = cm[1, 0]
        fp = cm[0, 1]
        rec = tp / (tp + fn)*1.0
        acc = (tp + tn) / (tp + tn + fn + fp)*1.0
        prec = tp / (tp + fp)*1.0
        f1 = 2.0*prec*rec/(prec + rec)
        result_all.append([classes[id_mat], str(prec), str(rec), str(f1), str(acc)])
        print(classes[id_mat], prec, rec, f1, acc)
        #print(cm)
    
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=['False', 'True'], yticklabels=['False', 'True'],
               title=title + ' for target class ' + classes[id_mat],
               ylabel='GT label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
    
        plt.savefig(output + '/confusion_matrix_{}_{}.png'.format(id_mat, text))
        plt.close()
    with open(output + '/results.txt', 'w') as f:
        f.write("Target\tPrecision\tRecall\tF1\tAccuracy\n")
        for r in result_all:
            f.write('\t'.join(r) + '\n')

    return ax
    

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          text='',
                          cmap=plt.cm.Blues,
                          output = "D:/Videos/Ceilling/video_1/TrainAndTest/"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    result_all = []
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [classes[i] for i in unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    """tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    rec = tp / (tp + fn)*1.0
    acc = (tp + tn) / (tp + tn + fn + fp)*1.0
    prec = tp / (tp + fp)*1.0
    f1 = 2.0*prec*rec/(prec + rec)"""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    result_all.append([str(prec), str(rec), str(f1), str(acc)])
    print(prec, rec, f1, acc)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(output + '/confusion_matrix_{}.png'.format(text))

    with open(output + '/results_{}.txt'.format(text), 'w') as f:
        f.write("Precision\tRecall\tF1\tAccuracy\n")
        for r in result_all:
            f.write('\t'.join(r) + '\n')

    return ax