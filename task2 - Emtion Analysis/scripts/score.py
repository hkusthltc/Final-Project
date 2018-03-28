from scipy.stats import pearsonr 
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_intensity(y_preds, y_true, print_result=True, plot_fig=False):
    if print_result:
        print("pearson correlation %.4f %.4f" % pearsonr(y_preds, y_true)) 
    if plot_fig:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(y_preds, y_true)
        plt.title("")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    return pearsonr(y_preds, y_true)[0]

def evaluate_classification(y_preds, y_true, print_result=True, plot_fig=False):
    acc = accuracy_score(y_true, y_preds)
    if print_result:
        print("Accuracy %.4f" % acc)
    if plot_fig:
        plot_confusion_matrix(confusion_matrix(y_true, y_preds), [0, 1, 2, 3])
    return acc

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'
                          ):
    import matplotlib.pyplot as plt
    import itertools
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__=='__main__':
    # example for classification
    y_preds = [0,1,2,3]
    y_true = [0,0,0,0]
    evaluate_classification(y_preds, y_true, True, False)

    # example for regression
    y_preds = [1,2,3,4]
    y_true = [1.1,2.2,3.3,3.9]
    evaluate_intensity(y_preds, y_true, True, False)