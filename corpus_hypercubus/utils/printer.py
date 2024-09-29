
# __all__=["Quest", "Printing_press"]
# class Handler:
#     def __init__(self):
#         quest_board = {}

#     def register_quest

# class Quest(object):
#     def __init__(self):
#         self._party = []

#     def __iadd__(self, member):
#         self._party.append(member)
#         return self

#     def __isub__(self, member):
#         self._party.remove(member)
#         return self

#     def __call__(self, *args, **kwargs):
#         for member in self._party:
#             member(*args, **kwargs)

class PrintingPress:
    def clear(n=1):
        ANSI_LINE_UP = '\033[1A'
        ANSI_LINE_CLEAR ='\x1b[2K'
        for i in range(n):
            print(ANSI_LINE_UP, end=ANSI_LINE_CLEAR)

    def training(batch_size:int, lr:float, loss:float, updates:int)->int:
        print(f"=+=+=+=+=+=+=+=+=+=+=+=+=o\nBatch_Size: {batch_size:d}\nLearning_Rate: {lr:.8f}\n.........................o\nTraining_Loss: {loss:.4f}\nUpdates: {updates:.2f}\n=+=+=+=+=+=+=+=+=+=+=+=+=o")
        return 7

    def validation(epoch:int, val_loss:float, val_acc:float)->int:
        print(f"\nEpoch:{epoch:d}\n=+=+=+=+=+=+=+=+=+=+=+=+=o\nValidate_Loss: {val_loss:.4f}\nValidate_Accuracy: {val_acc:.4f}\n=+=+=+=+=+=+=+=+=+=+=+=+=o")
        return 6

