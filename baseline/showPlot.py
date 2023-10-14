import matplotlib.pyplot as plt

def parse_and_plot(text):
    lines = text.split('\n')
    epochs = []
    train_loss = []
    train_cer = []
    val_loss = []
    val_cer = []

    for line in lines:
        if line:
            parts = line.split(', ')
            epochs.append(int(parts[0].split('=')[1]))
            train_loss.append(float(parts[1].split('=')[1]))
            train_cer.append(float(parts[2].split('=')[1]))
            val_loss.append(float(parts[3].split('=')[1]))
            val_cer.append(float(parts[4].split('=')[1]))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss, color=color, label='Train Loss')
    ax1.plot(epochs, val_loss, color=color, linestyle='dashed', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 3])

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('CER', color=color)
    ax2.plot(epochs, train_cer, color=color, label='Train CER')
    ax2.plot(epochs, val_cer, color=color, linestyle='dashed', label='Validation CER')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1])

    fig.tight_layout()
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.xticks(range(min(epochs), max(epochs)+1))

    plt.show()

if __name__ == '__main__':
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    text = '\n'.join(lines)
    parse_and_plot(text)
