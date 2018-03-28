from measures import moses_multi_bleu
import numpy as np

def bleu_compute(hyp, ref, print_bleu=True):
    bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True) 
    if print_bleu:
        print("bleu score: ", bleu_score)
    return bleu_score

if __name__=="__main__":
    # examples
    y = ["When I was little , I thought my country was the best on the planet , and I grew up singing a song called ; Nothing To Envy"]
    y_pred = ["When I was little , I thought my country was the best on the planet , and I grew up singing a song called ; Nothing To Envy"]
    bleu_compute(y_pred, y)

    y = ["When I was little , I thought my country was the best on the planet , and I grew up singing a song called ; Nothing To Envy"]
    y_pred = ["was little , country was the best ,grew up singing a song Envy"]
    bleu_compute(y_pred, y)