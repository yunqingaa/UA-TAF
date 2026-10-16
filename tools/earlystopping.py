import os
import numpy as np
import torch as th

class EarlyStopping:
    """
    Early stopping based on Macro-F1.
    It saves Accuracy, Macro-P, Macro-R, Macro-F1,
    and per-class Precision / Recall / F1.
    """

    def __init__(self, patience=10, verbose=False, delta=0.0):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if delta < 0:
            raise ValueError("delta must be non-negative")

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.val_loss_min = np.inf

        self.accs = 0.0
        self.macro_p = 0.0
        self.macro_r = 0.0
        self.macro_f1 = 0.0

        self.prec = [0.0, 0.0, 0.0, 0.0]
        self.rec = [0.0, 0.0, 0.0, 0.0]
        self.f1s = [0.0, 0.0, 0.0, 0.0]

        self.F1 = 0.0
        self.F2 = 0.0
        self.F3 = 0.0
        self.F4 = 0.0

    def __call__(self, val_loss, accs, macro_p, macro_r, macro_f1,
                 prec, rec, f1s, model, modelname, dataname):

        # Use Macro-F1 as the early stopping criterion
        score = macro_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss, accs, macro_p, macro_r, macro_f1,
                prec, rec, f1s, model, modelname, dataname
            )

        elif score <= self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(
                val_loss, accs, macro_p, macro_r, macro_f1,
                prec, rec, f1s, model, modelname, dataname
            )
            self.counter = 0

    def save_checkpoint(self, val_loss, accs, macro_p, macro_r, macro_f1,
                        prec, rec, f1s, model, modelname, dataname):

        if self.verbose:
            print(
                "Validation Macro-F1 increased. "
                "Saving model ... Macro-F1: {:.4f}".format(macro_f1)
            )

        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        th.save(
            model.state_dict(),
            os.path.join(
                checkpoint_dir,
                "{}_{}.m".format(modelname, dataname)
            )
        )

        self.val_loss_min = val_loss

        self.accs = float(accs)
        self.macro_p = float(macro_p)
        self.macro_r = float(macro_r)
        self.macro_f1 = float(macro_f1)

        self.prec = self._pad_metric(prec)
        self.rec = self._pad_metric(rec)
        self.f1s = self._pad_metric(f1s)

        self.F1, self.F2, self.F3, self.F4 = self.f1s

    @staticmethod
    def _pad_metric(values, size=4):
        values = [float(value) for value in values]
        return (values + [0.0] * size)[:size]


class MacroF1EarlyStopping:
    """Minimal Macro-F1 early stopping for scripts that only need a checkpoint."""

    def __init__(self, patience=10, verbose=False, delta=0.0):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if delta < 0:
            raise ValueError("delta must be non-negative")

        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.delta = float(delta)
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, macro_f1, model, modelname, dataname):
        score = float(macro_f1)

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model, modelname, dataname, score)
            return

        self.counter += 1
        if self.verbose:
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model, modelname, dataname, macro_f1):
        if self.verbose:
            print(
                "Validation Macro-F1 increased. "
                "Saving model ... Macro-F1: {:.4f}".format(macro_f1)
            )

        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        th.save(
            model.state_dict(),
            os.path.join(
                checkpoint_dir,
                "{}_{}.m".format(modelname, dataname)
            )
        )
