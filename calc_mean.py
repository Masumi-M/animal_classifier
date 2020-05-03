import numpy as np
import pickle
import matplotlib.pyplot as plt
import parameter

database_path = parameter.database_path
cross_num = parameter.cross_num
epoch_num = parameter.epoch_num

def main():
    hist_all_acc = np.zeros((5, epoch_num))
    hist_all_acc[:,:] = np.nan

    for i_cross in range(cross_num):
        database_path_cross = database_path + "/cross" + str(i_cross + 1)
        print(database_path_cross)
        hist_file = open(database_path_cross + "/history.pkl", "rb")
        hist_data = pickle.load(hist_file)
        hist_file.close()
        print(hist_data['accuracy'])
        hist_all_acc[i_cross, len(hist_data['accuracy'])] = np.array(hist_data['accuracy'])
    
    hist_all_acc_mean = np.nanmean(hist_all_acc, axis=0)
    print(hist_all_acc_mean)

    # hist_visualize(hist_data, database_path)

def hist_visualize(hist_data, database_path):
    acc = hist_data['accuracy']
    loss = hist_data['loss']
    val_acc = hist_data['val_accuracy']
    val_loss = hist_data['val_loss']

    epochs = range(len(acc))

    fig = plt.figure(figsize=(7, 4))

    # Accuracy
    ax_acc = fig.add_subplot(1, 2, 1)
    ax_acc.plot(epochs, acc, label='Training')
    ax_acc.plot(epochs, val_acc, label='Validation')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy Score')
    ax_acc.set_ylim([0, 1])
    ax_acc.set_title('Accuracy')
    ax_acc.legend(loc='best')
    ax_acc.grid()

    # Loss
    ax_loss = fig.add_subplot(1, 2, 2)
    ax_loss.plot(epochs, loss, label='Training')
    ax_loss.plot(epochs, val_loss, label='Validation')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss Score')
    ax_loss.set_ylim([0, 1])
    ax_loss.set_title('Loss')
    ax_loss.legend(loc='best')
    ax_loss.grid()

    # Save and Show
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.15,
                        top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig(database_path + '/train_result.png')
    # plt.show()
    return

if __name__ == "__main__":
    main()
