import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def continuous_visualizations(data, column):
    sns.set(style="whitegrid")

    # Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="HeartDisease", y=column, data=data)
    plt.title(f"Box Plot: {column} vs. HeartDisease")
    plt.xlabel("HeartDisease")
    plt.ylabel(column)
    plt.show()

    # Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="HeartDisease", y=column, data=data)
    plt.title(f"Violin Plot: {column} vs. HeartDisease")
    plt.xlabel("HeartDisease")
    plt.ylabel(column)
    plt.show()

    # KDE Plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data, x=column, hue="HeartDisease", common_norm=False)
    plt.title(f"KDE Plot: {column} vs. HeartDisease")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=column, hue="HeartDisease", element="step", stat="density", common_norm=False)
    plt.title(f"Histogram: {column} vs. HeartDisease")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.show()


def categorical_visualizations(data, column):
    sns.set(style="whitegrid")

    #Bar Plot 
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=column, hue="HeartDisease")
    plt.title(f"Bar Plot: {column} vs. HeartDisease")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.legend(title="HeartDisease")
    plt.show()
    
    value_counts = data.groupby(column)["HeartDisease"].value_counts(normalize=True).unstack()
    print(value_counts)


def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes

    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        if isinstance(value, np.ma.core.MaskedArray):
            value = value.data[0]
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):

    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):

    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    plt.xlim( (0, AUC.shape[1]) )

    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.colorbar(c)

    show_values(c)

    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):

    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []

    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
